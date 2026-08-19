"""Dataset loading for Kestrel."""

from __future__ import annotations

import glob
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")


def apply_adaptive_equalization(image, enabled=False, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Optionally enhance grayscale contrast with CLAHE."""
    if not enabled:
        return image
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    return clahe.apply(image)


def _read_yaml(data: str | Path) -> dict:
    with open(data, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_names(names):
    if names is None:
        return None
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


def _resolve_split_images(base: Path, split_spec):
    if split_spec is None:
        return []
    if isinstance(split_spec, list):
        paths = []
        for item in split_spec:
            paths.extend(_resolve_split_images(base, item))
        return sorted(set(paths))

    p = Path(split_spec)
    p = p if p.is_absolute() else (base / p)

    if p.is_dir():
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(str(p / "**" / ext), recursive=True))
        return sorted(set(files))

    if p.is_file() and p.suffix.lower() == ".txt":
        with open(p, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        paths = []
        for line in lines:
            lp = Path(line)
            paths.append(str(lp) if lp.is_absolute() else str((base / line).resolve()))
        return sorted(set(paths))

    raise FileNotFoundError(f"Could not resolve dataset split '{split_spec}' under {base}")


def _label_path_for(image_path: str | Path) -> Path:
    p = Path(image_path)
    parts = list(p.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return p.with_suffix(".txt")


def _read_yolo_labels(label_path: str | Path):
    if not Path(label_path).exists():
        return []
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:5])
            boxes.append((cls_id, xc, yc, w, h))
    return boxes


def discover_classes(image_paths):
    max_id = -1
    for img in image_paths:
        for cls_id, *_ in _read_yolo_labels(_label_path_for(img)):
            max_id = max(max_id, cls_id)
    return None if max_id < 0 else {i: str(i) for i in range(max_id + 1)}


def probe_image_size(image_paths, sample=32):
    sizes = Counter()
    for image_path in image_paths[:sample]:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        sizes[(w, h)] += 1
    return sizes.most_common(1)[0][0] if sizes else None


def _preprocess_split(image_paths, imgsz, adaptive_equalization=False, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
    target_width, target_height = imgsz
    images, bbox_labels, class_labels = [], [], []

    for image_path in image_paths:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = apply_adaptive_equalization(
            img,
            enabled=adaptive_equalization,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )
        boxes = _read_yolo_labels(_label_path_for(image_path))
        if not boxes:
            continue

        # Keep at most one annotation per class by selecting the largest box for each class.
        boxes_by_class = {}
        for cls_id, xc, yc, w, h in boxes:
            area = w * h
            current = boxes_by_class.get(cls_id)
            if current is None or area > current[-1]:
                boxes_by_class[cls_id] = (xc, yc, w, h, area)

        cls_ids = sorted(boxes_by_class.keys())
        _, xc, yc, w, h, _ = max(boxes_by_class.values(), key=lambda b: b[-1])
        resized = cv2.resize(img, (target_width, target_height)).astype(np.float32) / 255.0

        images.append(resized)
        bbox_labels.append([xc, yc, w, h])
        class_labels.append(cls_ids)

    X = (
        np.expand_dims(np.array(images, dtype=np.float32), axis=-1)
        if images
        else np.empty((0, target_height, target_width, 1), dtype=np.float32)
    )
    y_bbox = np.array(bbox_labels, dtype=np.float32) if bbox_labels else np.empty((0, 4), dtype=np.float32)
    y_class = class_labels
    return X, y_bbox, y_class


def load_dataset(
    data,
    imgsz=None,
    val_split=0.2,
    seed=0,
    adaptive_equalization=False,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
):
    cfg = _read_yaml(data)
    base = Path(cfg.get("path", "."))
    if not base.is_absolute():
        base = (Path(data).parent / base).resolve()

    train_images = _resolve_split_images(base, cfg.get("train"))
    val_spec = cfg.get("val")
    val_images = _resolve_split_images(base, val_spec) if val_spec else []

    if not train_images:
        raise FileNotFoundError(f"No training images found for dataset '{data}' (checked {base}).")

    names = _normalize_names(cfg.get("names"))
    if names is None:
        names = discover_classes(train_images + val_images) or {}

    if imgsz is None:
        probed = probe_image_size(train_images or val_images)
        imgsz = probed or (160, 120)
    imgsz = tuple(imgsz)

    Xtr, yb_tr, yc_tr = _preprocess_split(
        train_images,
        imgsz,
        adaptive_equalization=adaptive_equalization,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )

    if val_images:
        Xval, yb_val, yc_val = _preprocess_split(
            val_images,
            imgsz,
            adaptive_equalization=adaptive_equalization,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
        )
    else:
        n = len(Xtr)
        n_val = int(n * val_split)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        yc_tr_list = list(yc_tr)
        Xval, yb_val, yc_val = Xtr[val_idx], yb_tr[val_idx], [yc_tr_list[i] for i in val_idx]
        Xtr, yb_tr, yc_tr = Xtr[train_idx], yb_tr[train_idx], [yc_tr_list[i] for i in train_idx]

    nc = len(names) if names else 0

    def _to_multihot(label_lists):
        y = np.zeros((len(label_lists), nc), dtype=np.float32)
        for row, cls_ids in enumerate(label_lists):
            for cls_id in cls_ids:
                if 0 <= int(cls_id) < nc:
                    y[row, int(cls_id)] = 1.0
        return y

    yc_tr = _to_multihot(yc_tr)
    yc_val = _to_multihot(yc_val)

    return Xtr, yb_tr, yc_tr, Xval, yb_val, yc_val, names, imgsz
