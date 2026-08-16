import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from .data import load_dataset
from .geometry import estimate_polygon, pad_to_square
from .nn import build_bbox_model, build_class_model, parse_scale
from .results import Results

DEFAULT_NAMES = {0: "Rock", 1: "Paper", 2: "Scissors"}


class Kestrel:
    """A YOLO-style wrapper around Kestrel's two Keras heads, run as a cascade:

        class_model(image) -> predicted class
        bbox_model(image, predicted class) -> predicted box

        model = Kestrel("kestrel26n.yaml")   # new, untrained
        model = Kestrel("kestrel26n.pt")      # load saved weights
        model.train(data="rps.yaml", epochs=3)   # data= is a standard YOLO dataset yaml
        model.val()
        model("bus.jpg")
        model.export(format="onnx")
    """

    def __init__(self, model="kestrel26n.yaml", task="detect"):
        self.task = task
        self.names = dict(DEFAULT_NAMES)
        self.cfg = {}
        self.scale = "n"
        self.bbox_model = None
        self.class_model = None
        self.train_args = {}
        self.metrics = None
        self._bbox_stats = None   # avg width/height/center_y of hand boxes, learned in train()
        self._val_cache = None
        self.class_values = {}

        model = str(model)
        if model.endswith((".yaml", ".yml")):
            self._new(model)
        elif model.endswith(".pt"):
            self._load(model)
        else:
            raise ValueError(
                f"Unsupported model source '{model}'. Use a .yaml path to build a new model "
                "or a .pt path to load a saved one."
            )

    # ------------------------------------------------------------------ build
    def _new(self, cfg_path):
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
        else:
            default_path = Path(__file__).parent / "cfg" / "default.yaml"
            with open(default_path) as f:
                cfg = yaml.safe_load(f) or {}

        self.cfg = cfg
        if cfg.get("names"):
            names = cfg["names"]
            self.names = (
                dict(enumerate(names))
                if isinstance(names, list)
                else {int(k): v for k, v in names.items()}
            )
            self.cfg["names"] = self.names
            self.cfg["nc"] = len(self.names)
        self._refresh_class_values()

        self.scale = parse_scale(cfg_path)
        self.class_model = build_class_model(cfg, self.scale)
        self.bbox_model = build_bbox_model(cfg, self.scale)

    def _refresh_class_values(self):
        """Spread the available class IDs across the classifier's full tanh range."""
        class_ids = sorted(int(k) for k in self.names)
        if not class_ids:
            self.class_values = {}
            return
        values = [0.0] if len(class_ids) == 1 else np.linspace(
            -1.0, 1.0, num=len(class_ids), dtype=np.float32)

        self.class_values = {
            cls_id: float(val)
            for cls_id, val in zip(class_ids, values)
        }

    def _ensure_arch(self, imgsz, names):
        """(Re)build the two heads if the auto-detected dataset geometry/classes
        differ from what the models were last built with -- mirrors how a bare
        architecture .yaml only becomes fully-specified once real data is seen."""
        cur_imgsz = tuple(self.cfg.get("imgsz", (160, 120)))
        if tuple(imgsz) != cur_imgsz or (names and names != self.names):
            self.cfg["imgsz"] = list(imgsz)
            if names:
                self.names = names
                self.cfg["names"] = self.names
                self.cfg["nc"] = len(self.names)
                self._refresh_class_values()
            self.class_model = build_class_model(self.cfg, self.scale)
            self.bbox_model = build_bbox_model(self.cfg, self.scale)

    # -------------------------------------------------------------- load/save
    def _load(self, pt_path):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Model weights not found: {pt_path}")
        tmp = Path(f".kestrel_load_{os.getpid()}")
        try:
            self._extract_from_load(pt_path, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _extract_from_load(self, pt_path, tmp):
        with zipfile.ZipFile(pt_path, "r") as zf:
            zf.extractall(path=tmp)
        import tensorflow as tf
        self.class_model = tf.keras.models.load_model(tmp / "class_model.keras")
        self.bbox_model = tf.keras.models.load_model(tmp / "bbox_model.keras")
        with open(tmp / "meta.json") as f:
            meta = json.load(f)
        self.names = {int(k): v for k, v in meta.get("names", DEFAULT_NAMES).items()}
        self._bbox_stats = meta.get("bbox_stats")
        self.train_args = meta.get("train_args", {})
        self.cfg = meta.get("cfg", {})
        self.scale = meta.get("scale", "n")
        self.cfg["names"] = self.names
        self.cfg["nc"] = len(self.names)
        self._refresh_class_values()

    def save(self, path="kestrel_model.pt"):
        """Bundle both Keras models + metadata into a single .pt-style zip file."""
        tmp = Path(f".kestrel_save_{os.getpid()}")
        tmp.mkdir(exist_ok=True)
        try:
            self.class_model.save(tmp / "class_model.keras")
            self.bbox_model.save(tmp / "bbox_model.keras")
            meta = {
                "names": self.names,
                "bbox_stats": self._bbox_stats,
                "train_args": self.train_args,
                "cfg": self.cfg,
                "scale": self.scale,
            }
            with open(tmp / "meta.json", "w") as f:
                json.dump(meta, f)
            with zipfile.ZipFile(path, "w") as zf:
                for p in tmp.iterdir():
                    zf.write(p, arcname=p.name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return path

    # -------------------------------------------------------------------- train
    def train(self, data, epochs=100, batch_size=64, imgsz=None, val_split=0.2, **fit_kwargs):
        """`data` is a standard YOLO dataset yaml (path/train/val/nc/names).

        Image size and class list are auto-detected from the dataset itself
        the same way YOLO does -- pass `imgsz=` explicitly to override that.
        (The architecture .yaml's `imgsz` is only a fallback for building/
        predicting before any training has happened; it never blocks
        auto-detection here.)
        """
        (Xtr, yb_tr, yc_tr,
         Xval, yb_val, yc_val,
         names, detected_imgsz) = load_dataset(data, imgsz=imgsz, val_split=val_split)

        self._ensure_arch(detected_imgsz, names)

        # 1) class model trains first, directly on the full frame.
        yc_tr_val = self._idx_to_value(yc_tr)
        self.class_model.compile(optimizer="adam", loss="mse")
        hist_class = self.class_model.fit(Xtr, yc_tr_val, epochs=epochs, batch_size=batch_size, **fit_kwargs)

        # 2) bbox model trains second, conditioned on the (true, teacher-forced) class value.
        self.bbox_model.compile(optimizer="adam", loss="mse")
        hist_bbox = self.bbox_model.fit(
            [Xtr, yc_tr_val], yb_tr[:, 0],  # regress xmin only -- this is the model's single output
            epochs=epochs, batch_size=batch_size, **fit_kwargs,
        )

        if len(yb_tr):
            self._bbox_stats = {
                "avg_center_y": float(yb_tr[:, 1].mean()),
                "avg_width": float((yb_tr[:, 2] - yb_tr[:, 0]).mean()),
                "avg_height": float((yb_tr[:, 3] - (2 * yb_tr[:, 1] - yb_tr[:, 3])).mean()),
            }
        self.train_args = dict(data=data, epochs=epochs, batch_size=batch_size, imgsz=list(detected_imgsz))
        self._val_cache = (Xval, yb_val, yc_val)

        self.metrics = {
            "class_loss": float(hist_class.history["loss"][-1]),
            "bbox_loss": float(hist_bbox.history["loss"][-1]),
        }
        return self.metrics

    def _idx_to_value(self, y_class_idx):
        return np.array([self.class_values.get(int(v), 0.0) for v in y_class_idx], dtype=np.float32)

    def _value_to_idx(self, v):
        if not self.class_values:
            return 0

        class_ids = np.array(sorted(self.class_values), dtype=np.int64)
        class_values = np.array([self.class_values[i] for i in class_ids], dtype=np.float32)
        return int(class_ids[np.argmin(np.abs(class_values - float(v)))])

    # ---------------------------------------------------------------------- val
    def val(self, data=None, batch_size=64):
        if data is not None:
            _, _, _, Xval, yb_val, yc_val, names, _ = load_dataset(
                data, imgsz=tuple(self.cfg.get("imgsz", (160, 120))), val_split=1.0
            )
        elif self._val_cache is not None:
            Xval, yb_val, yc_val = self._val_cache
        else:
            raise ValueError("No validation data available -- call train() first, or pass data=<yaml>.")

        if len(Xval) == 0:
            self.metrics = {"class_loss": None, "class_accuracy": None, "bbox_loss": None}
            return self.metrics

        # Cascade evaluation: class runs first, and its *predicted* value feeds bbox,
        # matching what actually happens at inference time.
        yc_val_true = self._idx_to_value(yc_val)
        class_loss = self.class_model.evaluate(Xval, yc_val_true, batch_size=batch_size, verbose=0)

        class_preds = self.class_model.predict(Xval, batch_size=batch_size, verbose=0).reshape(-1)
        pred_labels = np.array([self._value_to_idx(p) for p in class_preds])
        class_acc = float((pred_labels == yc_val).mean())

        bbox_loss = self.bbox_model.evaluate([Xval, class_preds], yb_val[:, 0], batch_size=batch_size, verbose=0)

        self.metrics = {
            "class_loss": float(class_loss),
            "class_accuracy": class_acc,
            "bbox_loss": float(bbox_loss),
        }
        return self.metrics

    # ----------------------------------------------------------------- predict
    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)

    def predict(self, source, imgsz=None):
        imgsz = tuple(imgsz or self.cfg.get("imgsz", (160, 120)))
        images_gray, originals = self._load_sources(source)

        out = []
        for gray, orig in zip(images_gray, originals):
            oh, ow = gray.shape[:2]
            resized = cv2.resize(gray, imgsz)
            inp = (resized.astype(np.float32) - 128.0)
            img_in = inp[None, ..., None]

            # 1) class first, on the full frame.
            class_val = float(self.class_model.predict(img_in, verbose=0).reshape(-1)[0])
            class_idx = self._value_to_idx(class_val)

            # 2) bbox second, conditioned on the predicted class.
            cls_in = np.array([[class_val]], dtype=np.float32)
            pred_xmin = float(self.bbox_model.predict([img_in, cls_in], verbose=0).reshape(-1)[0])

            stats = self._bbox_stats or {
                "avg_center_y": imgsz[1] / 2,
                "avg_width": imgsz[0] / 3,
                "avg_height": imgsz[1] / 2,
            }
            xmin = int(np.clip(pred_xmin, 0, imgsz[0] - 1))
            xmax = int(np.clip(pred_xmin + stats["avg_width"], 0, imgsz[0] - 1))
            ymin = int(np.clip(stats["avg_center_y"] - stats["avg_height"] / 2, 0, imgsz[1] - 1))
            ymax = int(np.clip(stats["avg_center_y"] + stats["avg_height"] / 2, 0, imgsz[1] - 1))

            sx, sy = ow / imgsz[0], oh / imgsz[1]
            box_orig = np.array([[xmin * sx, ymin * sy, xmax * sx, ymax * sy]])

            out.append(Results(
                orig_img=orig, boxes=box_orig, cls=np.array([class_idx]),
                names=self.names, conf=np.array([abs(class_val)]),
            ))
        return out

    @staticmethod
    def _load_sources(source):
        sources = source if isinstance(source, (list, tuple)) else [source]
        images, originals = [], []
        for s in sources:
            if isinstance(s, np.ndarray):
                img = s
            elif isinstance(s, str) and s.startswith(("http://", "https://")):
                with urllib.request.urlopen(s) as resp:
                    arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(s)
            if img is None:
                raise ValueError(f"Could not load image from source: {s!r}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            images.append(gray)
            originals.append(img)
        return images, originals

    # ----------------------------------------------------------------- export
    def export(self, format="onnx", **kwargs):
        exporters = {
            "tflite": self._export_tflite,
            "onnx": self._export_onnx,
            "saved_model": self._export_saved_model,
        }
        if format not in exporters:
            raise ValueError(f"Unsupported export format '{format}'. Choose from {list(exporters)}.")
        try:
            return exporters[format](**kwargs)
        except Exception as e:
            print(f"kestrel export ({format}) failed: {e}")
            return False

    def _export_tflite(self, **kwargs):
        import tensorflow as tf
        paths = []
        for name, model in (("class_model", self.class_model), ("bbox_model", self.bbox_model)):
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_bytes = converter.convert()
            path = f"{name}.tflite"
            with open(path, "wb") as f:
                f.write(tflite_bytes)
            paths.append(path)
        return paths

    def _export_onnx(self, **kwargs):
        try:
            import tf2onnx
        except ImportError as e:
            raise ImportError("Exporting to ONNX requires tf2onnx: pip install tf2onnx") from e
        paths = []
        for name, model in (("class_model", self.class_model), ("bbox_model", self.bbox_model)):
            path = f"{name}.onnx"
            tf2onnx.convert.from_keras(model, output_path=path)
            paths.append(path)
        return paths

    def _export_saved_model(self, **kwargs):
        paths = []
        for name, model in (("class_model", self.class_model), ("bbox_model", self.bbox_model)):
            path = f"{name}_saved_model"
            model.export(path)
            paths.append(path)
        return paths
