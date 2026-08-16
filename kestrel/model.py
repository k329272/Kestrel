"""High-level Kestrel model wrapper."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from tqdm.auto import tqdm

from .data import load_dataset
from .nn import build_model
from .results import Results

DEFAULT_NAMES = {0: "Rock", 1: "Paper", 2: "Scissors"}


def _normalize_names(names):
    if names is None:
        return None
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


class Kestrel:
    """A small PyTorch detector with a YOLO-style API."""

    def __init__(self, model="kestrel26n.yaml", task="detect"):
        self.task = task
        self.names = dict(DEFAULT_NAMES)
        self.cfg = {}
        self.model = None
        self.train_args = {}
        self.metrics = None
        self._val_cache = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = str(model)
        if model.endswith((".yaml", ".yml")):
            self._new(model)
        elif model.endswith(".pt"):
            self._load(model)
        else:
            raise ValueError(
                f"Unsupported model source '{model}'. Use a .yaml path to build a new model "
                "or a .pt path to load a saved checkpoint."
            )

    # ------------------------------------------------------------------ build
    def _new(self, cfg_path):
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            default_path = Path(__file__).parent / "cfg" / "default.yaml"
            with open(default_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        self.cfg = cfg
        names = _normalize_names(cfg.get("names"))
        if names:
            self.names = names
            self.cfg["names"] = self.names
            self.cfg["nc"] = len(self.names)
        self.model = build_model(self.cfg).to(self.device)

    def _ensure_arch(self, imgsz, names):
        cur_imgsz = tuple(self.cfg.get("imgsz", (160, 120)))
        if tuple(imgsz) != cur_imgsz or (names and names != self.names):
            self.cfg["imgsz"] = list(imgsz)
            if names:
                self.names = names
                self.cfg["names"] = self.names
                self.cfg["nc"] = len(self.names)
            self.model = build_model(self.cfg).to(self.device)

    # -------------------------------------------------------------- load/save
    def _load(self, pt_path):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Model weights not found: {pt_path}")
        ckpt = torch.load(pt_path, map_location="cpu")
        self.cfg = ckpt.get("cfg", {})
        self.names = {int(k): v for k, v in ckpt.get("names", DEFAULT_NAMES).items()}
        self.model = build_model(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.train_args = ckpt.get("train_args", {})
        self.metrics = ckpt.get("metrics")
        self._val_cache = None

    def save(self, path="kestrel_model.pt"):
        payload = {
            "model_state": self.model.state_dict(),
            "cfg": self.cfg,
            "names": self.names,
            "train_args": self.train_args,
            "metrics": self.metrics,
        }
        torch.save(payload, path)
        return path

    # -------------------------------------------------------------------- train
    def train(self, data, epochs=20, batch_size=32, imgsz=None, val_split=0.2, lr=1e-3, bbox_loss_weight=1.0):
        (
            Xtr,
            yb_tr,
            yc_tr,
            Xval,
            yb_val,
            yc_val,
            names,
            detected_imgsz,
        ) = load_dataset(data, imgsz=imgsz, val_split=val_split)

        self._ensure_arch(detected_imgsz, names)

        Xtr_t = torch.from_numpy(Xtr).permute(0, 3, 1, 2).contiguous().float()
        yb_tr_t = torch.from_numpy(yb_tr).float()
        yc_tr_t = torch.from_numpy(yc_tr).long()
        train_loader = DataLoader(
            TensorDataset(Xtr_t, yc_tr_t, yb_tr_t),
            batch_size=batch_size,
            shuffle=True,
        )

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        class_loss_fn = nn.CrossEntropyLoss()
        bbox_loss_fn = nn.SmoothL1Loss()

        history = {"class_loss": [], "bbox_loss": [], "total_loss": []}
        epoch_bar = tqdm(range(epochs), desc="train", unit="epoch")
        for _epoch in epoch_bar:
            class_loss_sum = 0.0
            bbox_loss_sum = 0.0
            total_sum = 0.0
            count = 0
            batch_bar = tqdm(train_loader, desc="batch", unit="batch", leave=False)
            for xb, y_class, y_bbox in batch_bar:
                xb = xb.to(self.device)
                y_class = y_class.to(self.device)
                y_bbox = y_bbox.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                class_logits, bbox_pred = self.model(xb)
                class_loss = class_loss_fn(class_logits, y_class)
                bbox_loss = bbox_loss_fn(bbox_pred, y_bbox)
                loss = class_loss + bbox_loss_weight * bbox_loss
                loss.backward()
                optimizer.step()

                batch = xb.size(0)
                class_loss_sum += float(class_loss.item()) * batch
                bbox_loss_sum += float(bbox_loss.item()) * batch
                total_sum += float(loss.item()) * batch
                count += batch

                batch_bar.set_postfix(
                    class_loss=float(class_loss.item()),
                    bbox_loss=float(bbox_loss.item()),
                    total=float(loss.item()),
                )

            history["class_loss"].append(class_loss_sum / max(1, count))
            history["bbox_loss"].append(bbox_loss_sum / max(1, count))
            history["total_loss"].append(total_sum / max(1, count))
            epoch_bar.set_postfix(
                class_loss=history["class_loss"][-1],
                bbox_loss=history["bbox_loss"][-1],
                total_loss=history["total_loss"][-1],
            )

        self.metrics = {
            "class_loss": history["class_loss"][-1] if history["class_loss"] else None,
            "bbox_loss": history["bbox_loss"][-1] if history["bbox_loss"] else None,
            "total_loss": history["total_loss"][-1] if history["total_loss"] else None,
        }
        self.train_args = {
            "data": data,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": list(detected_imgsz),
            "lr": lr,
            "bbox_loss_weight": bbox_loss_weight,
        }
        self._val_cache = (Xval, yb_val, yc_val)
        return self.metrics

    # ---------------------------------------------------------------------- val
    def val(self, data=None, batch_size=32):
        if data is not None:
            _, _, _, Xval, yb_val, yc_val, names, _ = load_dataset(
                data, imgsz=tuple(self.cfg.get("imgsz", (160, 120))), val_split=1.0
            )
            if names:
                self.names = names
        elif self._val_cache is not None:
            Xval, yb_val, yc_val = self._val_cache
        else:
            raise ValueError("No validation data available. Call train() first or pass data=<yaml>.")

        if len(Xval) == 0:
            self.metrics = {"class_loss": None, "class_accuracy": None, "bbox_loss": None}
            return self.metrics

        Xval_t = torch.from_numpy(Xval).permute(0, 3, 1, 2).contiguous().float().to(self.device)
        yb_val_t = torch.from_numpy(yb_val).float().to(self.device)
        yc_val_t = torch.from_numpy(yc_val).long().to(self.device)

        self.model.eval()
        class_loss_fn = nn.CrossEntropyLoss()
        bbox_loss_fn = nn.SmoothL1Loss()
        with torch.no_grad():
            val_loader = DataLoader(
                TensorDataset(Xval_t, yc_val_t, yb_val_t),
                batch_size=batch_size,
                shuffle=False,
            )
            class_loss_sum = 0.0
            bbox_loss_sum = 0.0
            correct = 0
            total = 0
            val_bar = tqdm(val_loader, desc="val", unit="batch")
            for xb, y_class, y_bbox in val_bar:
                class_logits, bbox_pred = self.model(xb)
                class_loss = class_loss_fn(class_logits, y_class)
                bbox_loss = bbox_loss_fn(bbox_pred, y_bbox)
                pred_labels = class_logits.argmax(dim=1)

                batch = xb.size(0)
                class_loss_sum += float(class_loss.item()) * batch
                bbox_loss_sum += float(bbox_loss.item()) * batch
                correct += int((pred_labels == y_class).sum().item())
                total += batch

                val_bar.set_postfix(
                    class_loss=float(class_loss.item()),
                    bbox_loss=float(bbox_loss.item()),
                )

            class_loss = class_loss_sum / max(1, total)
            bbox_loss = bbox_loss_sum / max(1, total)
            class_acc = correct / max(1, total)

        self.metrics = {
            "class_loss": class_loss,
            "class_accuracy": class_acc,
            "bbox_loss": bbox_loss,
        }
        return self.metrics

    # ----------------------------------------------------------------- predict
    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)

    def predict(self, source, imgsz=None):
        imgsz = tuple(imgsz or self.cfg.get("imgsz", (160, 120)))
        images_gray, originals = self._load_sources(source)

        out = []
        self.model.eval()
        with torch.no_grad():
            predict_bar = tqdm(zip(images_gray, originals), total=len(originals), desc="predict", unit="image")
            for gray, orig in predict_bar:
                oh, ow = gray.shape[:2]
                resized = cv2.resize(gray, imgsz).astype(np.float32) / 255.0
                img_in = torch.from_numpy(resized)[None, None, ...].to(self.device)

                class_logits, bbox_pred = self.model(img_in)
                probs = torch.softmax(class_logits, dim=1)[0]
                class_idx = int(probs.argmax().item())
                conf = float(probs[class_idx].item())

                cx, cy, w, h = bbox_pred[0].clamp(0.0, 1.0).cpu().numpy()
                x1 = int(np.clip((cx - w / 2) * ow, 0, ow - 1))
                y1 = int(np.clip((cy - h / 2) * oh, 0, oh - 1))
                x2 = int(np.clip((cx + w / 2) * ow, 0, ow - 1))
                y2 = int(np.clip((cy + h / 2) * oh, 0, oh - 1))

                out.append(
                    Results(
                        orig_img=orig,
                        boxes=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                        cls=np.array([class_idx], dtype=np.int64),
                        names=self.names,
                        conf=np.array([conf], dtype=np.float32),
                    )
                )
                predict_bar.set_postfix(class_id=class_idx, conf=f"{conf:.2f}")
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
                img = cv2.imread(str(s))
            if img is None:
                raise ValueError(f"Could not load image from source: {s!r}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            images.append(gray)
            originals.append(img)
        return images, originals

    # ----------------------------------------------------------------- export
    def export(self, format="torchscript", **kwargs):
        if format == "checkpoint":
            return self.save(kwargs.get("path", "kestrel_model.pt"))
        if format != "torchscript":
            raise ValueError("Unsupported export format. Use 'torchscript' or 'checkpoint'.")

        example_size = tuple(self.cfg.get("imgsz", (160, 120)))
        example = torch.zeros(1, 1, example_size[1], example_size[0], device=self.device)
        traced = torch.jit.trace(self.model.eval(), example)
        path = kwargs.get("path", "kestrel_model.torchscript.pt")
        traced.save(path)
        return path
