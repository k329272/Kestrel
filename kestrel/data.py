"""Dataset loading for Kestrel

Accepts a standard ultralytics-style data yaml:

    path: ../datasets/rps       # optional root, relative to the yaml's folder
    train: images/train          # directory, or a .txt file listing image paths
    val: images/val               # optional
    nc: 3                          # optional
    names: [Rock, Paper, Scissors] # optional

Labels follow the standard YOLO convention: for an image at
`.../images/<split>/foo.jpg`, its label file is expected at
`.../labels/<split>/foo.txt`, one line per object:

    class_id x_center y_center width height      # all normalized 0-1

Like ultralytics, Kestrel does NOT require `nc`/`names`/image size to be
declared up front -- it scans the dataset to figure them out, the same way
`ultralytics.data.utils.check_det_dataset` infers class counts and
`autobatch`/loader logic infers image geometry.
"""

import glob
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")


def _read_yaml(data):
    with open(data) as f:
        return yaml.safe_load(f) or {}


def _normalize_names(names):
    """YOLO yaml allows `names` as a list OR a {id: name} dict -- support both."""
    if names is None:
        return None
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


def _resolve_split_images(base, split_spec):
    """Resolve a YOLO-style split entry into a sorted list of image paths.

    `split_spec` may be a directory, a .txt file listing image paths (one per
    line, relative to `base` or absolute), or a list of paths -- all valid
    forms in ultralytics data yamls.
    """
    if split_spec is None: return []
    if isinstance(split_spec, list):
        out = []
        for p in split_spec: 
            out.extend(_resolve_split_images(base, p))
        return sorted(set(out))

    p = Path(split_spec)
    p = p if p.is_absolute() else (base / p)

    if p.is_dir():
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(str(p / "**" / ext), recursive=True))
        return sorted(set(files))

    if p.is_file() and p.suffix.lower() == ".txt":
        with open(p) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        out = []
        for ln in lines:
            lp = Path(ln)
            out.append(str(lp) if lp.is_absolute() else str((base / ln).resolve()))
        return sorted(set(out))

    raise FileNotFoundError(f"Could not resolve dataset split '{split_spec}' under {base}")


def _label_path_for(image_path):
    """Standard YOLO convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt"""
    p = Path(image_path)
    parts = list(p.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        return Path(*parts).with_suffix(".txt")
    return p.with_suffix(".txt")  # fallback: label sits next to the image


def _read_yolo_labels(label_path):
    """Returns [(class_id, x_center, y_center, width, height), ...], all normalized 0-1."""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:5])
            boxes.append((cls_id, xc, yc, w, h))
    return boxes


def discover_classes(image_paths):
    """Scan every label file to infer nc/names, the way YOLO datasets self-describe
    their class list when a yaml doesn't provide `names` explicitly."""
    max_id = -1
    for img in image_paths:
        for cls_id, *_ in _read_yolo_labels(_label_path_for(img)):
            max_id = max(max_id, cls_id)
    return None if max_id < 0 else {i: str(i) for i in range(max_id + 1)}


def probe_image_size(image_paths, sample=32):
    """Auto-detect a representative (width, height) for the dataset, YOLO-style,
    by sampling real images instead of trusting a hardcoded constant."""
    sizes = Counter()
    for image_path in image_paths[:sample]:
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        sizes[(w, h)] += 1
    return sizes.most_common(1)[0][0] if sizes else None


def _preprocess_split(image_paths, imgsz):
    target_width, target_height = imgsz
    images, bbox_labels, class_labels = [], [], []

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        boxes = _read_yolo_labels(_label_path_for(image_path))
        if not boxes:
            continue

        # Single-object assumption (as in the source pipeline): keep the largest box.
        cls_id, xc, yc, w, h = max(boxes, key=lambda b: b[3] * b[4])

        img_resized = cv2.resize(img, (target_width, target_height))
        img_resized = (img_resized - 128).astype(np.float32)

        # Normalized YOLO box -> pixel coords in the resized frame.
        sxmin = int(np.clip((xc - w / 2) * target_width, 0, target_width - 1))
        sxmax = int(np.clip((xc + w / 2) * target_width, 0, target_width - 1))
        scy = int(np.clip(yc * target_height, 0, target_height - 1))
        symax = int(np.clip((yc + h / 2) * target_height, 0, target_height - 1))

        images.append(img_resized)
        bbox_labels.append([sxmin, scy, sxmax, symax])  # [xmin, center_y, xmax, ymax]
        class_labels.append(cls_id)

    X = (np.expand_dims(np.array(images), axis=-1) if images
         else np.empty((0, target_height, target_width, 1), dtype=np.float32))
    y_bbox = np.array(bbox_labels, dtype=np.float32) if bbox_labels else np.empty((0, 4), dtype=np.float32)
    y_class = np.array(class_labels, dtype=np.int64) if class_labels else np.empty((0,), dtype=np.int64)
    return X, y_bbox, y_class


def load_dataset(data, imgsz=None, val_split=0.2, seed=0):
    """Parse a YOLO-format dataset described by `data` (a yaml path).

    Auto-detects whatever the yaml doesn't explicitly declare: image size
    (`imgsz=None`) and class list (`names` missing from the yaml) -- mirroring
    how ultralytics datasets self-describe rather than requiring everything
    spelled out up front.

    Returns: (X_train, y_bbox_train, y_class_train,
              X_val, y_bbox_val, y_class_val,
              names, imgsz)
    """
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

    Xtr, yb_tr, yc_tr = _preprocess_split(train_images, imgsz)

    if val_images:
        Xval, yb_val, yc_val = _preprocess_split(val_images, imgsz)
    else:
        n = len(Xtr)
        n_val = int(n * val_split)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        Xval, yb_val, yc_val = Xtr[val_idx], yb_tr[val_idx], yc_tr[val_idx]
        Xtr, yb_tr, yc_tr = Xtr[train_idx], yb_tr[train_idx], yc_tr[train_idx]

    return Xtr, yb_tr, yc_tr, Xval, yb_val, yc_val, names, imgsz