"""PyTorch network used by Kestrel."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

DEFAULT_MODEL_CFG = {
    "channels": [24, 48, 96],
    "hidden": 128,
    "dropout": 0.1,
    "se_ratio": 0.25,
}


def _as_channels(value: Sequence[int] | int | None) -> list[int]:
    if value is None:
        return list(DEFAULT_MODEL_CFG["channels"])
    if isinstance(value, int):
        return [value]
    return [int(v) for v in value]


class ConvBlock(nn.Sequential):
    """Conv-BN-SiLU block used throughout the backbone."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """Small residual block to keep the backbone expressive without changing outputs."""

    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        squeezed = max(8, int(channels * se_ratio))
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeezed, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeezed, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        residual = residual * self.se(residual)
        return self.act(x + residual)


class KestrelNet(nn.Module):
    """A small shared-backbone classifier + class-specific box regressor."""

    def __init__(
        self,
        nc: int = 3,
        channels: Sequence[int] | int | None = None,
        hidden: int = 128,
        dropout: float = 0.1,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        channels = _as_channels(channels)
        if not channels:
            channels = list(DEFAULT_MODEL_CFG["channels"])

        layers: list[nn.Module] = [ConvBlock(1, channels[0], stride=2)]
        in_channels = channels[0]
        for out_channels in channels[1:]:
            layers.append(ConvBlock(in_channels, out_channels, stride=2))
            layers.append(ResidualBlock(out_channels, se_ratio=se_ratio))
            in_channels = out_channels
        layers.append(ResidualBlock(in_channels, se_ratio=se_ratio))

        self.backbone = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.class_head = nn.Linear(hidden, nc)
        self.box_head = nn.Linear(hidden, nc * 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        class_logits = self.class_head(features)
        box = torch.sigmoid(self.box_head(features)).view(-1, class_logits.shape[-1], 4)
        return class_logits, box


def build_model(cfg: dict) -> KestrelNet:
    model_cfg = cfg.get("model", {})
    nc = int(cfg.get("nc", len(cfg.get("names", {})) or 3))
    channels = model_cfg.get("channels", DEFAULT_MODEL_CFG["channels"])
    hidden = int(model_cfg.get("hidden", DEFAULT_MODEL_CFG["hidden"]))
    dropout = float(model_cfg.get("dropout", DEFAULT_MODEL_CFG["dropout"]))
    se_ratio = float(model_cfg.get("se_ratio", DEFAULT_MODEL_CFG["se_ratio"]))
    return KestrelNet(nc=nc, channels=channels, hidden=hidden, dropout=dropout, se_ratio=se_ratio)
