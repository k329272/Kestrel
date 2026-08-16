"""PyTorch network used by Kestrel."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

DEFAULT_MODEL_CFG = {
    "channels": [16, 32, 64],
    "hidden": 64,
}


def _as_channels(value: Sequence[int] | int | None) -> list[int]:
    if value is None:
        return list(DEFAULT_MODEL_CFG["channels"])
    if isinstance(value, int):
        return [value]
    return [int(v) for v in value]


class KestrelNet(nn.Module):
    """A small shared-backbone classifier + box regressor."""

    def __init__(self, nc: int = 3, channels: Sequence[int] | int | None = None, hidden: int = 64):
        super().__init__()
        channels = _as_channels(channels)
        if not channels:
            channels = list(DEFAULT_MODEL_CFG["channels"])

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.backbone = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
        )
        self.class_head = nn.Linear(hidden, nc)
        self.box_head = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        class_logits = self.class_head(features)
        box = torch.sigmoid(self.box_head(features))
        return class_logits, box


def build_model(cfg: dict) -> KestrelNet:
    model_cfg = cfg.get("model", {})
    nc = int(cfg.get("nc", len(cfg.get("names", {})) or 3))
    channels = model_cfg.get("channels", DEFAULT_MODEL_CFG["channels"])
    hidden = int(model_cfg.get("hidden", DEFAULT_MODEL_CFG["hidden"]))
    return KestrelNet(nc=nc, channels=channels, hidden=hidden)
