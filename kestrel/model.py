"""High-level Kestrel model wrapper."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

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


def _available_ram_bytes() -> int | None:
    """Best-effort cross-platform estimate of available system RAM."""
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            return None
        return None

    if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names and "SC_PAGE_SIZE" in os.sysconf_names:
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, TypeError):
            return None
    return None


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
        self.preprocess = {
            "adaptive_equalization": False,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": (8, 8),
        }
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

    @staticmethod
    def _resolve_loader_args(batch_size: int, pin_memory: bool | None, num_workers: int | None, device: torch.device):
        if pin_memory is None:
            pin_memory = device.type == "cuda"
        if num_workers is None:
            # Default to single-process loading to avoid multiprocessing shutdown
            num_workers = 0
        return {
            "batch_size": batch_size,
            "pin_memory": pin_memory,
            "num_workers": num_workers,
            "persistent_workers": bool(num_workers),
        }

    @staticmethod
    def _auto_batch_size(Xtr, batch_size: int | str | None, device: torch.device) -> int:
        if batch_size not in (None, "auto"):
            return max(1, int(batch_size))
        if len(Xtr) == 0:
            return 1

        sample_bytes = int(np.asarray(Xtr[0]).nbytes)
        available = _available_ram_bytes()
        if available is None:
            return min(32, len(Xtr))

        reserve = 512 * 1024 * 1024 if device.type == "cuda" else 256 * 1024 * 1024
        usable = max(available - reserve, sample_bytes)
        target = int(usable * (0.02 if device.type == "cuda" else 0.04))
        estimated = max(1, target // max(1, sample_bytes))
        return max(1, min(len(Xtr), estimated, 256))

    @staticmethod
    def _mean_loss(sum_value: float, count: int) -> float | None:
        return None if count <= 0 else sum_value / count

    @staticmethod
    def _metric_bundle(class_loss: float | None, bbox_loss: float | None, class_acc: float | None = None):
        metrics: dict[str, Any] = {
            "class_loss": class_loss,
            "bbox_loss": bbox_loss,
        }
        if class_acc is not None:
            metrics["class_accuracy"] = class_acc
        total_loss = None
        if class_loss is not None and bbox_loss is not None:
            total_loss = class_loss + bbox_loss
        metrics["total_loss"] = total_loss
        return metrics

    def _evaluate_tensors(self, Xval, yb_val, yc_val, batch_size: int = 32):
        if len(Xval) == 0:
            return self._metric_bundle(None, None, None)

        Xval_t = torch.from_numpy(Xval).permute(0, 3, 1, 2).contiguous().float()
        yb_val_t = torch.from_numpy(yb_val).float()
        yc_val_t = torch.from_numpy(yc_val).float()
        val_loader = DataLoader(
            TensorDataset(Xval_t, yc_val_t, yb_val_t),
            shuffle=False,
            **self._resolve_loader_args(batch_size, pin_memory=None, num_workers=None, device=self.device),
        )

        self.model.eval()
        class_loss_fn = nn.BCEWithLogitsLoss()
        bbox_loss_fn = nn.SmoothL1Loss()
        class_loss_sum = 0.0
        bbox_loss_sum = 0.0
        correct = 0
        total = 0
        use_amp = self.device.type == "cuda"
        with torch.no_grad():
            for xb, y_class, y_bbox in val_loader:
                xb = xb.to(self.device, non_blocking=True)
                y_class = y_class.to(self.device, non_blocking=True)
                y_bbox = y_bbox.to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    class_logits, bbox_pred = self.model(xb)
                    class_loss = class_loss_fn(class_logits, y_class)
                    bbox_loss = self._class_specific_bbox_loss(bbox_pred, y_bbox, y_class, bbox_loss_fn)
                pred_labels = (torch.sigmoid(class_logits) >= 0.5).to(y_class.dtype)
                pred_labels = pred_labels if pred_labels.ndim == y_class.ndim else pred_labels.unsqueeze(0)
                batch = xb.size(0)
                class_loss_sum += float(class_loss.item()) * batch
                bbox_loss_sum += float(bbox_loss.item()) * batch
                correct += int(((pred_labels == y_class).all(dim=1)).sum().item())
                total += batch

        class_loss = self._mean_loss(class_loss_sum, total)
        bbox_loss = self._mean_loss(bbox_loss_sum, total)
        class_acc = None if total <= 0 else correct / total
        return self._metric_bundle(class_loss, bbox_loss, class_acc)

    def _preprocess_image(self, gray):
        from .data import apply_adaptive_equalization

        return apply_adaptive_equalization(
            gray,
            enabled=self.preprocess.get("adaptive_equalization", False),
            clip_limit=self.preprocess.get("clahe_clip_limit", 2.0),
            tile_grid_size=self.preprocess.get("clahe_tile_grid_size", (8, 8)),
        )

    @staticmethod
    def _class_specific_bbox_loss(bbox_pred, y_bbox, y_class, loss_fn, bbox_scale=None):
        """Compute bbox loss only for positive classes, sharing the same target box."""
        if bbox_pred.ndim != 3:
            raise ValueError(f"Expected bbox predictions with shape [B, nc, 4], got {tuple(bbox_pred.shape)}")
        target = y_bbox[:, None, :].expand_as(bbox_pred)
        mask = y_class > 0.5
        if mask.ndim != 2:
            mask = mask.reshape(mask.shape[0], -1)
        if not torch.any(mask):
            return bbox_pred.sum() * 0.0
        pred = bbox_pred[mask]
        tgt = target[mask]
        if bbox_scale is not None:
            pred = pred * bbox_scale
            tgt = tgt * bbox_scale
        return loss_fn(pred, tgt)

    # -------------------------------------------------------------- load/save
    def _load(self, pt_path):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Model weights not found: {pt_path}")
        ckpt = torch.load(pt_path, map_location="cpu")
        self.cfg = ckpt.get("cfg", {})
        self.names = {int(k): v for k, v in ckpt.get("names", DEFAULT_NAMES).items()}
        self.preprocess.update(ckpt.get("preprocess", {}))
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
            "preprocess": {
                "adaptive_equalization": self.preprocess["adaptive_equalization"],
                "clahe_clip_limit": self.preprocess["clahe_clip_limit"],
                "clahe_tile_grid_size": list(self.preprocess["clahe_tile_grid_size"]),
            },
        }
        torch.save(payload, path)
        return path

    # -------------------------------------------------------------------- train
    def train(
        self,
        data,
        epochs=20,
        batch_size="auto",
        imgsz=None,
        val_split=0.2,
        lr=1e-3,
        bbox_loss_weight=1.0,
        patience=None,
        min_delta=0.0,
        scheduler_factor=0.5,
        scheduler_patience=3,
        adaptive_equalization=None,
        clahe_clip_limit=None,
        clahe_tile_grid_size=None,
        num_workers=None,
        pin_memory=None,
        amp=None,
        save_best=False,
        best_path="kestrel_best.pt",
    ):
        if adaptive_equalization is not None:
            self.preprocess["adaptive_equalization"] = bool(adaptive_equalization)
        if clahe_clip_limit is not None:
            self.preprocess["clahe_clip_limit"] = float(clahe_clip_limit)
        if clahe_tile_grid_size is not None:
            self.preprocess["clahe_tile_grid_size"] = tuple(clahe_tile_grid_size)

        (
            Xtr,
            yb_tr,
            yc_tr,
            Xval,
            yb_val,
            yc_val,
            names,
            detected_imgsz,
        ) = load_dataset(
            data,
            imgsz=imgsz,
            val_split=val_split,
            adaptive_equalization=self.preprocess["adaptive_equalization"],
            clahe_clip_limit=self.preprocess["clahe_clip_limit"],
            clahe_tile_grid_size=self.preprocess["clahe_tile_grid_size"],
        )

        self._ensure_arch(detected_imgsz, names)
        bbox_scale = torch.tensor(
            [float(detected_imgsz[0]), float(detected_imgsz[1]), float(detected_imgsz[0]), float(detected_imgsz[1])],
            device=self.device,
        )

        batch_size = self._auto_batch_size(Xtr, batch_size, self.device)

        Xtr_t = torch.from_numpy(Xtr).permute(0, 3, 1, 2).contiguous().float()
        yb_tr_t = torch.from_numpy(yb_tr).float()
        yc_tr_t = torch.from_numpy(yc_tr).float()
        train_loader = DataLoader(
            TensorDataset(Xtr_t, yc_tr_t, yb_tr_t),
            shuffle=True,
            **self._resolve_loader_args(batch_size, pin_memory=pin_memory, num_workers=num_workers, device=self.device),
        )

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )
        class_loss_fn = nn.BCEWithLogitsLoss()
        bbox_loss_fn = nn.SmoothL1Loss()
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda" if amp is None else bool(amp) and self.device.type == "cuda"))
        use_amp = self.device.type == "cuda" if amp is None else bool(amp) and self.device.type == "cuda"

        history = {"class_loss": [], "bbox_loss": [], "total_loss": []}
        best_total = None
        epochs_without_improve = 0
        last_val_metrics = None
        epoch_bar = tqdm(range(epochs), desc="train", unit="epoch")
        for _epoch in epoch_bar:
            class_loss_sum = 0.0
            bbox_loss_sum = 0.0
            total_sum = 0.0
            count = 0
            batch_bar = tqdm(train_loader, desc="batch", unit="batch", leave=False)
            for xb, y_class, y_bbox in batch_bar:
                xb = xb.to(self.device, non_blocking=True)
                y_class = y_class.to(self.device, non_blocking=True)
                y_bbox = y_bbox.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    class_logits, bbox_pred = self.model(xb)
                    class_loss = class_loss_fn(class_logits, y_class)
                    bbox_loss = self._class_specific_bbox_loss(
                        bbox_pred, y_bbox, y_class, bbox_loss_fn, bbox_scale=bbox_scale
                    )
                    loss = class_loss + bbox_loss_weight * bbox_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
            scheduler.step(history["total_loss"][-1])
            epoch_bar.set_postfix(
                class_loss=history["class_loss"][-1],
                bbox_loss=history["bbox_loss"][-1],
                total_loss=history["total_loss"][-1],
            )

            val_metrics = self._evaluate_tensors(Xval, yb_val, yc_val, batch_size=batch_size)
            last_val_metrics = val_metrics
            current_total = val_metrics.get("total_loss")
            if current_total is not None:
                improved = best_total is None or current_total < (best_total - min_delta)
                if improved:
                    best_total = current_total
                    epochs_without_improve = 0
                    if save_best:
                        self.save(best_path)
                else:
                    epochs_without_improve += 1
                    if patience is not None and epochs_without_improve >= patience:
                        break

        train_metrics = {
            "class_loss": self._mean_loss(sum(history["class_loss"]), len(history["class_loss"])) if history["class_loss"] else None,
            "bbox_loss": self._mean_loss(sum(history["bbox_loss"]), len(history["bbox_loss"])) if history["bbox_loss"] else None,
            "total_loss": self._mean_loss(sum(history["total_loss"]), len(history["total_loss"])) if history["total_loss"] else None,
        }
        self.metrics = last_val_metrics or train_metrics
        self.train_args = {
            "data": data,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": list(detected_imgsz),
            "lr": lr,
            "bbox_loss_weight": bbox_loss_weight,
            "patience": patience,
            "min_delta": min_delta,
            "scheduler_factor": scheduler_factor,
            "scheduler_patience": scheduler_patience,
            "adaptive_equalization": self.preprocess["adaptive_equalization"],
            "clahe_clip_limit": self.preprocess["clahe_clip_limit"],
            "clahe_tile_grid_size": list(self.preprocess["clahe_tile_grid_size"]),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "amp": amp,
            "save_best": save_best,
            "best_path": best_path,
        }
        self._val_cache = (Xval, yb_val, yc_val)
        return self.metrics

    # ---------------------------------------------------------------------- val
    def val(self, data=None, batch_size=32):
        if data is not None:
            imgsz = tuple(self.cfg.get("imgsz", (160, 120)))
            _, _, _, Xval, yb_val, yc_val, names, _ = load_dataset(
                data,
                imgsz=imgsz,
                val_split=1.0,
                adaptive_equalization=self.preprocess.get("adaptive_equalization", False),
                clahe_clip_limit=self.preprocess.get("clahe_clip_limit", 2.0),
                clahe_tile_grid_size=self.preprocess.get("clahe_tile_grid_size", (8, 8)),
            )
            if names:
                self.names = names
            bbox_scale = torch.tensor([float(imgsz[0]), float(imgsz[1]), float(imgsz[0]), float(imgsz[1])], device=self.device)
        elif self._val_cache is not None:
            Xval, yb_val, yc_val = self._val_cache
            imgsz = tuple(self.cfg.get("imgsz", (160, 120)))
            bbox_scale = torch.tensor([float(imgsz[0]), float(imgsz[1]), float(imgsz[0]), float(imgsz[1])], device=self.device)
        else:
            raise ValueError("No validation data available. Call train() first or pass data=<yaml>.")

        self.metrics = self._evaluate_tensors(Xval, yb_val, yc_val, batch_size=batch_size)
        return self.metrics

    # ----------------------------------------------------------------- predict
    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)

    def predict(self, source, imgsz=None, conf=0.25):
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
                probs = torch.softmax(class_logits, dim=1)[0].detach().cpu().numpy()
                class_ids = np.flatnonzero(probs >= conf)
                if class_ids.size == 0:
                    class_ids = np.array([int(np.argmax(probs))], dtype=np.int64)

                class_boxes = bbox_pred[0, class_ids].clamp(0.0, 1.0).cpu().numpy()
                boxes = []
                for cx, cy, w, h in class_boxes:
                    x1 = int(np.clip((cx - w / 2) * ow, 0, ow - 1))
                    y1 = int(np.clip((cy - h / 2) * oh, 0, oh - 1))
                    x2 = int(np.clip((cx + w / 2) * ow, 0, ow - 1))
                    y2 = int(np.clip((cy + h / 2) * oh, 0, oh - 1))
                    boxes.append([x1, y1, x2, y2])
                boxes = np.asarray(boxes, dtype=np.float32)
                confs = probs[class_ids].astype(np.float32)

                out.append(
                    Results(
                        orig_img=orig,
                        boxes=boxes,
                        cls=class_ids.astype(np.int64),
                        names=self.names,
                        conf=confs,
                        probs=probs.astype(np.float32),
                    )
                )
                predict_bar.set_postfix(class_ids=",".join(map(str, class_ids.tolist())), conf=f"{float(confs.max()):.2f}")
        return out

    def _load_sources(self, source):
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
            gray = self._preprocess_image(gray)
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
