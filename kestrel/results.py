"""Prediction result container for Kestrel outputs.

The package already expects a `Results` object with Ultralytics-like behavior:
`model.predict(...)` returns a list of results, and each result can be plotted
or saved. This module provides that missing container.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Results:
    """Lightweight prediction container."""

    orig_img: np.ndarray
    boxes: np.ndarray
    cls: np.ndarray
    names: dict
    conf: np.ndarray | None = None
    probs: np.ndarray | None = None

    def __post_init__(self):
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.cls = np.asarray(self.cls).reshape(-1)
        self.conf = None if self.conf is None else np.asarray(self.conf).reshape(-1)
        self.probs = None if self.probs is None else np.asarray(self.probs, dtype=np.float32)

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        return Results(
            orig_img=self.orig_img,
            boxes=self.boxes[idx],
            cls=self.cls[idx],
            names=self.names,
            conf=None if self.conf is None else self.conf[idx],
            probs=self.probs,
        )

    def _label_for(self, i: int) -> str:
        cls_idx = int(self.cls[i])
        name = self.names.get(cls_idx, str(cls_idx))
        return name if self.conf is None else f"{name} {float(self.conf[i]):.2f}"

    def plot(self, image=None, color=(0, 255, 0), thickness=2):
        """Return an annotated BGR image."""
        img = np.array(image if image is not None else self.orig_img).copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = max(0, x2)
            y2 = max(0, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            if label := self._label_for(i):
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                text_x = x1
                text_y = max(0, y1 - 6)
                bg_y1 = max(0, text_y - th - baseline)
                bg_y2 = text_y + baseline
                cv2.rectangle(img, (text_x, bg_y1), (text_x + tw + 4, bg_y2), color, -1)
                cv2.putText(
                    img,
                    label,
                    (text_x + 2, text_y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
        return img

    def save(self, path="results.jpg"):
        """Save the plotted image and return the output path."""
        path = Path(path)
        annotated = self.plot()
        cv2.imwrite(str(path), annotated)
        return str(path)

    def to_dict(self):
        return {
            "boxes": self.boxes.tolist(),
            "cls": self.cls.tolist(),
            "conf": None if self.conf is None else self.conf.tolist(),
            "probs": None if self.probs is None else self.probs.tolist(),
            "names": dict(self.names),
        }
