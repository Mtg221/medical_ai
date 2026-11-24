"""From-scratch PyTorch dermatology architectures (ResNetV1, SE-ResNetV2, ConvNeXt-Derm) & metadata encoders."""

from __future__ import annotations

from typing import Any

import numpy as np

from medresearch_platform.vision.catalog import DermCatalogRecord

SKIN_MODALITIES: tuple[str, ...] = ("clinical", "dermoscopic", "tbp_tile")


def _resolve_device(device: str | None = "auto") -> Any:
    import torch

    if device is None or str(device).strip().lower() in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _compute_binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    mask = (y_true == 0) | (y_true == 1)
    yt = y_true[mask]
    ys = y_score[mask]
    pos = int(np.sum(yt == 1))
    neg = int(np.sum(yt == 0))
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(-ys, kind="mergesort")
    yt_sorted = yt[order]
    tp = np.cumsum(yt_sorted == 1)
    fp = np.cumsum(yt_sorted == 0)
    tpr = tp / pos
    fpr = fp / neg
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz_fn(tpr, fpr))


def _build_resnet_module(
    include_metadata: bool,
    n_dx: int,
    n_mod: int,
    *,
    architecture_variant: str = "resnet_v1",
) -> Any:
    import torch
    import torch.nn as nn

    class SEBlock(nn.Module):
        def __init__(self, channels: int, reduction: int = 8) -> None:
            super().__init__()
            mid = max(8, channels // reduction)
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(channels, mid),
                nn.ReLU(inplace=True),
                nn.Linear(mid, channels),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
            return x * w

    class ResBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = False) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.se: nn.Module = SEBlock(out_ch) if use_se else nn.Identity()
            self.shortcut: nn.Module
            if stride != 1 or in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.se(self.bn2(self.conv2(out)))
            out = self.relu(out + self.shortcut(x))
            return out

    class ConvNeXtBlock(nn.Module):
        """From-scratch ConvNeXt block: 7x7 depthwise conv + GroupNorm(1) + 4x inverted GELU MLP + LayerScale."""

        def __init__(self, dim: int, layer_scale_init: float = 1e-2) -> None:
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
            self.norm = nn.GroupNorm(1, dim)
            self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
            self.act = nn.GELU()
            self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)
            self.gamma = nn.Parameter(layer_scale_init * torch.ones((1, dim, 1, 1)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.dwconv(x)
            h = self.norm(h)
            h = self.pwconv2(self.act(self.pwconv1(h)))
            return x + self.gamma * h

    class DermResNet4Stage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if architecture_variant == "convnext_derm":
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 48, kernel_size=4, stride=4),
                    nn.GroupNorm(1, 48),
                )
                self.stage1: nn.Module = nn.Sequential(ConvNeXtBlock(48), ConvNeXtBlock(48))
                self.stage2: nn.Module = nn.Sequential(
                    nn.GroupNorm(1, 48),
                    nn.Conv2d(48, 96, kernel_size=2, stride=2),
                    ConvNeXtBlock(96),
                    ConvNeXtBlock(96),
                )
                self.stage3: nn.Module = nn.Sequential(
                    nn.GroupNorm(1, 96),
                    nn.Conv2d(96, 192, kernel_size=2, stride=2),
                    ConvNeXtBlock(192),
                    ConvNeXtBlock(192),
                    ConvNeXtBlock(192),
                )
                self.stage4: nn.Module = nn.Sequential(
                    nn.GroupNorm(1, 192),
                    nn.Conv2d(192, 256, kernel_size=2, stride=2),
                    ConvNeXtBlock(256),
                    ConvNeXtBlock(256),
                    nn.GroupNorm(1, 256),
                )
                drop_p = 0.30
            elif architecture_variant == "resnet_v2_se":
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                )
                self.stage1 = nn.Sequential(ResBlock(32, 32, stride=1, use_se=True), ResBlock(32, 32, stride=1, use_se=True))
                self.stage2 = nn.Sequential(ResBlock(32, 64, stride=2, use_se=True), ResBlock(64, 64, stride=1, use_se=True))
                self.stage3 = nn.Sequential(ResBlock(64, 128, stride=2, use_se=True), ResBlock(128, 128, stride=1, use_se=True))
                self.stage4 = nn.Sequential(ResBlock(128, 256, stride=2, use_se=True), ResBlock(256, 256, stride=1, use_se=True))
                drop_p = 0.30
            else:
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                )
                self.stage1 = ResBlock(32, 32, stride=1)
                self.stage2 = ResBlock(32, 64, stride=2)
                self.stage3 = ResBlock(64, 128, stride=2)
                self.stage4 = ResBlock(128, 256, stride=2)
                drop_p = 0.20
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(drop_p)
            feat_dim = 256 + (8 if include_metadata else 0)
            self.fc_mal = nn.Linear(feat_dim, 1)
            self.fc_dx = nn.Linear(feat_dim, n_dx)
            self.fc_mod = nn.Linear(feat_dim, n_mod)

        def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h = self.stem(x)
            h = self.stage1(h)
            h = self.stage2(h)
            h = self.stage3(h)
            h = self.stage4(h)
            emb = self.pool(h).flatten(1)
            if include_metadata and meta is not None:
                emb = torch.cat([emb, meta], dim=1)
            emb = self.dropout(emb)
            return self.fc_mal(emb).squeeze(1), self.fc_dx(emb), self.fc_mod(emb)

    return DermResNet4Stage()


def encode_metadata_vector(rec: DermCatalogRecord, *, metadata_mode: str = "full") -> np.ndarray:
    """8-dim metadata vector: modality (3) + body_site group (5), respecting `metadata_mode`."""
    vec = np.zeros(8, dtype=np.float32)
    if metadata_mode == "pixels_only":
        return vec
    if rec.modality in SKIN_MODALITIES:
        vec[SKIN_MODALITIES.index(rec.modality)] = 1.0
    if metadata_mode == "modality_only":
        return vec
    bs = str(rec.body_site or "").lower()
    if any(k in bs for k in ("head", "neck", "face", "scalp")):
        vec[3] = 1.0
    elif any(k in bs for k in ("torso", "trunk", "back", "chest", "abdomen")):
        vec[4] = 1.0
    elif any(k in bs for k in ("upper", "arm", "forearm", "hand")):
        vec[5] = 1.0
    elif any(k in bs for k in ("lower", "leg", "thigh", "foot")):
        vec[6] = 1.0
    else:
        vec[7] = 1.0
    return vec
