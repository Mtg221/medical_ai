"""Step 4: Real 4-Stage PyTorch Residual CNN (32->64->128->256) trained from random init on CPU with masked malignancy loss and real Grad-CAM."""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.dx_mapping import (
    CORE_DX_CLASSES,
    MASKED_DX_INDEX,
    map_record_to_core_dx_index,
)

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

    class DermResNet4Stage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if architecture_variant == "resnet_v2_se":
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                )
                self.stage1: nn.Module = nn.Sequential(ResBlock(32, 32, stride=1, use_se=True), ResBlock(32, 32, stride=1, use_se=True))
                self.stage2: nn.Module = nn.Sequential(ResBlock(32, 64, stride=2, use_se=True), ResBlock(64, 64, stride=1, use_se=True))
                self.stage3: nn.Module = nn.Sequential(ResBlock(64, 128, stride=2, use_se=True), ResBlock(128, 128, stride=1, use_se=True))
                self.stage4: nn.Module = nn.Sequential(ResBlock(128, 256, stride=2, use_se=True), ResBlock(256, 256, stride=1, use_se=True))
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


class PyTorchDermResNet:
    """From-scratch 4-stage PyTorch Residual CNN (32->64->128->256) for skin lesion analysis."""

    def __init__(
        self,
        *,
        image_size: int = 96,
        include_metadata: bool = True,
        metadata_mode: str | None = None,
        body_site_dropout: float = 0.0,
        seed: int = 1,
        n_dx: int | None = None,
        device: str | None = "auto",
        architecture_variant: str = "resnet_v1",
    ) -> None:
        import torch

        if metadata_mode is None:
            resolved_mode = "full" if include_metadata else "pixels_only"
        else:
            resolved_mode = str(metadata_mode)
            if resolved_mode not in ("pixels_only", "modality_only", "full"):
                raise ValueError(f"Unsupported metadata_mode: {resolved_mode}")

        self.image_size = image_size
        self.metadata_mode = resolved_mode
        self.include_metadata = resolved_mode != "pixels_only"
        self.body_site_dropout = float(body_site_dropout)
        self.seed = seed
        self.n_dx = int(n_dx) if n_dx is not None else len(CORE_DX_CLASSES)
        self.architecture_variant = str(architecture_variant)
        self.architecture_name = (
            "PyTorchDermSEResNetV2-v2.0.0" if self.architecture_variant == "resnet_v2_se" else "PyTorchDermResNet4Stage-v1.0.0"
        )
        self.temperature_mal = 1.0
        self.temperature_dx = 1.0
        self.abstain_min_confidence = 0.55
        self.dx_head_supported = False
        self.train_history: dict[str, Any] = {}
        self._device = _resolve_device(device)

        torch.manual_seed(seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.net = _build_resnet_module(
            self.include_metadata,
            self.n_dx,
            len(SKIN_MODALITIES),
            architecture_variant=self.architecture_variant,
        ).to(self._device)
        self.parameter_count = sum(p.numel() for p in self.net.parameters())
        self.initial_weights_sha256 = self._weights_sha256()

    def _weights_sha256(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.net.state_dict().keys()):
            h.update(k.encode("utf-8"))
            h.update(self.net.state_dict()[k].detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def _arr_to_tensor(self, arr: np.ndarray) -> Any:
        import torch

        a = np.asarray(arr)
        if a.dtype == np.uint8:
            t = torch.from_numpy(a.transpose(2, 0, 1)).float().div_(255.0)
        else:
            t = torch.from_numpy(np.asarray(a, dtype=np.float32).transpose(2, 0, 1))
        if t.shape[1] != self.image_size or t.shape[2] != self.image_size:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return t

    def _arr_to_uint8_tensor(self, arr: np.ndarray) -> Any:
        """Compact [3, H, W] uint8 tensor (4x lower RAM at 224x224)."""
        import torch

        a = np.asarray(arr)
        if a.dtype == np.uint8 and a.shape[0] == self.image_size and a.shape[1] == self.image_size:
            return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        t = self._arr_to_tensor(a)
        return torch.clamp(t * 255.0, 0.0, 255.0).to(torch.uint8)

    def fit(
        self,
        train_samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
        val_samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
        *,
        epochs: int = 12,
        batch_size: int = 48,
        lr: float = 1e-3,
        patience: int = 4,
        max_samples_per_epoch: int = 12000,
        use_amp: bool = True,
        clinical_focus: bool = False,
        clinical_oversample: float = 4.0,
        focal_gamma: float = 0.0,
        mixup_alpha: float = 0.0,
    ) -> dict[str, Any]:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

        t0 = time.perf_counter()
        rng = random.Random(self.seed)
        dev = self._device
        is_cuda = dev.type == "cuda"
        amp_enabled = bool(use_amp and is_cuda)
        if is_cuda:
            amp_scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", None)
            scaler: Any = (
                amp_scaler_cls("cuda", enabled=amp_enabled)
                if amp_scaler_cls is not None
                else torch.cuda.amp.GradScaler(enabled=amp_enabled)
            )
        else:
            scaler = None

        def _autocast_ctx() -> Any:
            if is_cuda and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast("cuda", enabled=amp_enabled)
            return torch.cuda.amp.autocast(enabled=amp_enabled)

        def _pack(samples: list[tuple[np.ndarray, int, DermCatalogRecord]]) -> tuple[torch.Tensor, ...]:
            xs = torch.stack([self._arr_to_uint8_tensor(a) for a, _, _ in samples], dim=0)
            y_mal = torch.tensor([lbl for _, lbl, _ in samples], dtype=torch.long)
            y_dx = torch.tensor([map_record_to_core_dx_index(r) for _, _, r in samples], dtype=torch.long)
            y_mod = torch.tensor([SKIN_MODALITIES.index(r.modality) if r.modality in SKIN_MODALITIES else 0 for _, _, r in samples], dtype=torch.long)
            meta = torch.stack(
                [torch.from_numpy(encode_metadata_vector(r, metadata_mode=self.metadata_mode)) for _, _, r in samples],
                dim=0,
            )
            return xs, y_mal, y_dx, y_mod, meta

        x_tr, y_mal_tr, y_dx_tr, y_mod_tr, m_tr = _pack(train_samples)
        x_va, y_mal_va, y_dx_va, y_mod_va, m_va = _pack(val_samples)

        # Class-balanced sampler weights on malignancy (masked=-1 gets neutral weight)
        n_pos = int((y_mal_tr == 1).sum().item())
        n_neg = int((y_mal_tr == 0).sum().item())
        w_pos = len(y_mal_tr) / max(1, 2 * n_pos)
        w_neg = len(y_mal_tr) / max(1, 2 * n_neg)
        sample_weights = torch.where(
            y_mal_tr == 1,
            torch.tensor(w_pos, dtype=torch.float32),
            torch.where(y_mal_tr == 0, torch.tensor(w_neg, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)),
        )
        clinical_mod_idx = SKIN_MODALITIES.index("clinical")
        if clinical_focus and clinical_oversample > 1.0:
            sample_weights = torch.where(
                y_mod_tr == clinical_mod_idx,
                sample_weights * float(clinical_oversample),
                sample_weights,
            )
        sampler = WeightedRandomSampler(
            sample_weights.tolist(),
            num_samples=min(len(y_mal_tr), max_samples_per_epoch),
            replacement=True,
        )
        loader = DataLoader(
            TensorDataset(x_tr, y_mal_tr, y_dx_tr, y_mod_tr, m_tr),
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=is_cuda,
        )

        bce = nn.BCEWithLogitsLoss(reduction="none")
        ce = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

        def _eval_val_logits() -> tuple[torch.Tensor, torch.Tensor]:
            mal_parts: list[torch.Tensor] = []
            dx_parts: list[torch.Tensor] = []
            val_bs = max(64, batch_size)
            self.net.eval()
            with torch.no_grad():
                for i in range(0, len(x_va), val_bs):
                    xv = x_va[i : i + val_bs].to(dev, non_blocking=is_cuda).float().div_(255.0)
                    mv = m_va[i : i + val_bs].to(dev, non_blocking=is_cuda) if self.include_metadata else None
                    with _autocast_ctx():
                        lm, ld, _ = self.net(xv, mv)
                    mal_parts.append(lm.float().cpu())
                    dx_parts.append(ld.float().cpu())
            return torch.cat(mal_parts, dim=0), torch.cat(dx_parts, dim=0)

        best_val_auroc = -1.0
        best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        stale = 0
        history: list[dict[str, float]] = []

        for ep in range(1, epochs + 1):
            ep_t0 = time.perf_counter()
            self.net.train()
            ep_loss = 0.0
            n_batches = 0
            for xb, ymb, ydb, ymodb, mb in loader:
                xb = xb.to(dev, non_blocking=is_cuda).float().div_(255.0)
                ymb = ymb.to(dev, non_blocking=is_cuda)
                ydb = ydb.to(dev, non_blocking=is_cuda)
                ymodb = ymodb.to(dev, non_blocking=is_cuda)
                mb = mb.to(dev, non_blocking=is_cuda)

                # Augmentations: flip H/V, 90-deg rot, brightness/contrast/channel jitter + cutout
                if rng.random() < 0.5:
                    xb = torch.flip(xb, dims=[3])
                if rng.random() < 0.35:
                    xb = torch.flip(xb, dims=[2])
                if rng.random() < 0.5:
                    k_rot = rng.choice([1, 2, 3])
                    xb = torch.rot90(xb, k=k_rot, dims=[2, 3])
                gain = rng.uniform(0.88, 1.12)
                bias = rng.uniform(-0.06, 0.06)
                xb = torch.clamp(xb * gain + bias, 0.0, 1.0)
                if clinical_focus:
                    # Smartphone-realistic augmentations: lesion scale-crop, RGB white-balance cast, uneven lighting
                    if rng.random() < 0.55 and xb.shape[2] >= 64:
                        h_im, w_im = xb.shape[2], xb.shape[3]
                        scale = rng.uniform(0.72, 0.96)
                        ch = max(32, int(round(h_im * scale)))
                        cw = max(32, int(round(w_im * scale)))
                        y0 = rng.randint(0, h_im - ch)
                        x0 = rng.randint(0, w_im - cw)
                        xb = F.interpolate(
                            xb[:, :, y0 : y0 + ch, x0 : x0 + cw],
                            size=(h_im, w_im),
                            mode="bilinear",
                            align_corners=False,
                        )
                    if rng.random() < 0.60:
                        rgb_gain = torch.tensor(
                            [rng.uniform(0.88, 1.12), rng.uniform(0.88, 1.12), rng.uniform(0.88, 1.12)],
                            device=dev,
                            dtype=xb.dtype,
                        ).view(1, 3, 1, 1)
                        xb = torch.clamp(xb * rgb_gain, 0.0, 1.0)
                    if rng.random() < 0.30:
                        grad_1d = torch.linspace(rng.uniform(-0.08, 0.0), rng.uniform(0.0, 0.08), xb.shape[3], device=dev, dtype=xb.dtype)
                        xb = torch.clamp(xb + grad_1d.view(1, 1, 1, -1), 0.0, 1.0)
                if rng.random() < 0.25 and xb.shape[2] >= 64:
                    h_im, w_im = xb.shape[2], xb.shape[3]
                    ch = rng.randint(max(8, h_im // 10), max(12, h_im // 5))
                    cw = rng.randint(max(8, w_im // 10), max(12, w_im // 5))
                    y0 = rng.randint(0, h_im - ch)
                    x0 = rng.randint(0, w_im - cw)
                    xb[:, :, y0 : y0 + ch, x0 : x0 + cw] = 0.0

                if self.metadata_mode == "full" and self.body_site_dropout > 0.0:
                    mb = mb.clone()
                    for row_i in range(mb.shape[0]):
                        if rng.random() < self.body_site_dropout:
                            mb[row_i, 3:7] = 0.0
                            mb[row_i, 7] = 1.0

                # Optional Mixup on malignancy-labeled batches (only mixes within same modality to preserve clinical metadata)
                use_mixup = bool(mixup_alpha > 0.0 and rng.random() < 0.40 and xb.shape[0] >= 4)
                ymb_soft = ymb.float()
                if use_mixup:
                    lam = float(np.random.default_rng(self.seed + ep * 10000 + n_batches).beta(mixup_alpha, mixup_alpha))
                    lam = max(lam, 1.0 - lam)
                    perm_idx = torch.randperm(xb.shape[0], device=dev)
                    same_mod = (ymodb == ymodb[perm_idx]) & ((ymb == 0) | (ymb == 1)) & ((ymb[perm_idx] == 0) | (ymb[perm_idx] == 1))
                    if same_mod.any():
                        xb = torch.where(same_mod.view(-1, 1, 1, 1), lam * xb + (1.0 - lam) * xb[perm_idx], xb)
                        ymb_soft = torch.where(same_mod, lam * ymb.float() + (1.0 - lam) * ymb[perm_idx].float(), ymb_soft)

                opt.zero_grad()
                with _autocast_ctx():
                    l_mal, l_dx, l_mod = self.net(xb, mb if self.include_metadata else None)
                    mask = (ymb == 0) | (ymb == 1)
                    if mask.any():
                        raw_bce = bce(l_mal[mask], ymb_soft[mask])
                        if focal_gamma > 0.0:
                            pt = torch.exp(-raw_bce).clamp(1e-6, 1.0)
                            raw_bce = ((1.0 - pt) ** float(focal_gamma)) * raw_bce
                        if clinical_focus:
                            w_m = torch.where(
                                ymodb[mask] == clinical_mod_idx,
                                torch.tensor(1.5, device=dev, dtype=raw_bce.dtype),
                                torch.tensor(1.0, device=dev, dtype=raw_bce.dtype),
                            )
                            loss_mal = (raw_bce * w_m).sum() / w_m.sum().clamp(min=1.0)
                        else:
                            loss_mal = raw_bce.mean()
                    else:
                        loss_mal = torch.tensor(0.0, device=dev)
                    dx_mask = ydb != MASKED_DX_INDEX
                    loss_dx = ce(l_dx[dx_mask], ydb[dx_mask]) if dx_mask.any() else torch.tensor(0.0, device=dev)
                    loss_mod = ce(l_mod, ymodb)
                    loss = loss_mal + 0.35 * loss_dx + 0.15 * loss_mod

                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                ep_loss += float(loss.item())
                n_batches += 1
            scheduler.step()

            vl_mal, _ = _eval_val_logits()
            vp_mal = torch.sigmoid(vl_mal).numpy()
            val_auroc = _compute_binary_auroc(y_mal_va.numpy(), vp_mal)
            val_clin_mask = (y_mod_va.numpy() == clinical_mod_idx) & ((y_mal_va.numpy() == 0) | (y_mal_va.numpy() == 1))
            val_clin_auroc = (
                _compute_binary_auroc(y_mal_va.numpy()[val_clin_mask], vp_mal[val_clin_mask])
                if int(np.sum(val_clin_mask)) >= 10
                else val_auroc
            )
            selection_metric = (0.65 * val_clin_auroc + 0.35 * val_auroc) if clinical_focus else val_auroc

            ep_sec = round(time.perf_counter() - ep_t0, 2)
            mean_loss = round(ep_loss / max(1, n_batches), 4)
            print(
                f"[seed={self.seed} dev={dev} img={self.image_size} meta={self.include_metadata} clin_focus={clinical_focus}] "
                f"epoch {ep}/{epochs}: loss={mean_loss:.4f} val_auroc={val_auroc:.4f} val_clin_auroc={val_clin_auroc:.4f} ({ep_sec}s)",
                flush=True,
            )
            history.append({
                "epoch": float(ep),
                "loss": mean_loss,
                "val_auroc": round(val_auroc, 4),
                "val_clin_auroc": round(val_clin_auroc, 4),
                "sec": ep_sec,
            })

            if selection_metric > best_val_auroc + 1e-4:
                best_val_auroc = selection_metric
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        self.net.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        self.net.eval()

        # Fit scalar temperature on val malignancy logits (masked)
        vl_mal, vl_dx = _eval_val_logits()
        mask = (y_mal_va == 0) | (y_mal_va == 1)
        if mask.any():
            best_t = 1.0
            best_nll = float("inf")
            for t in np.linspace(0.5, 2.5, 21):
                nll = float(bce(vl_mal[mask] / float(t), y_mal_va[mask].float()).mean().item())
                if nll < best_nll:
                    best_nll = nll
                    best_t = float(t)
            self.temperature_mal = round(best_t, 3)

        dx_va_mask = y_dx_va != MASKED_DX_INDEX
        if dx_va_mask.any():
            best_tdx = 1.0
            best_nll_dx = float("inf")
            for t in np.linspace(0.5, 2.5, 21):
                nll_dx = float(ce(vl_dx[dx_va_mask] / float(t), y_dx_va[dx_va_mask]).item())
                if nll_dx < best_nll_dx:
                    best_nll_dx = nll_dx
                    best_tdx = float(t)
            self.temperature_dx = round(best_tdx, 3)

        dx_tr_valid = y_dx_tr[y_dx_tr != MASKED_DX_INDEX].numpy()
        cls_counts = (
            np.bincount(dx_tr_valid, minlength=len(CORE_DX_CLASSES))
            if len(dx_tr_valid)
            else np.zeros(len(CORE_DX_CLASSES), dtype=np.int64)
        )
        self.dx_head_supported = bool(int(np.sum(cls_counts >= 30)) >= 2)

        train_info = {
            "seed": self.seed,
            "include_metadata": self.include_metadata,
            "parameter_count": self.parameter_count,
            "epochs_run": len(history),
            "best_val_auroc": round(best_val_auroc, 4),
            "dx_labelled_train_count": int(len(dx_tr_valid)),
            "dx_head_supported": self.dx_head_supported,
            "temperature_mal": self.temperature_mal,
            "temperature_dx": self.temperature_dx,
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "history": history,
            "initial_weights_sha256": self.initial_weights_sha256,
            "pretrained_weights_used": False,
        }
        self.train_history = train_info
        return train_info

    def predict_batch(
        self,
        samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
        *,
        batch_size: int = 64,
        tta: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        self.net.eval()
        dev = self._device
        p_mal_list, p_dx_list, p_mod_list = [], [], []
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                chunk = samples[i : i + batch_size]
                xb = torch.stack([self._arr_to_tensor(a) for a, _, _ in chunk], dim=0).to(dev)
                mb = torch.stack(
                    [torch.from_numpy(encode_metadata_vector(r, metadata_mode=self.metadata_mode)) for _, _, r in chunk],
                    dim=0,
                ).to(dev)
                meta_in = mb if self.include_metadata else None
                if tta:
                    views = [
                        xb,
                        torch.flip(xb, dims=[3]),
                        torch.flip(xb, dims=[2]),
                        torch.rot90(xb, k=1, dims=[2, 3]),
                    ]
                    l_mal_sum, l_dx_sum, l_mod_sum = self.net(views[0], meta_in)
                    for xv in views[1:]:
                        lm, ld, lmo = self.net(xv, meta_in)
                        l_mal_sum = l_mal_sum + lm
                        l_dx_sum = l_dx_sum + ld
                        l_mod_sum = l_mod_sum + lmo
                    l_mal = l_mal_sum / float(len(views))
                    l_dx = l_dx_sum / float(len(views))
                    l_mod = l_mod_sum / float(len(views))
                else:
                    l_mal, l_dx, l_mod = self.net(xb, meta_in)
                p_mal_list.append(torch.sigmoid(l_mal / max(0.1, self.temperature_mal)).cpu().numpy())
                p_dx_list.append(torch.softmax(l_dx / max(0.1, self.temperature_dx), dim=1).cpu().numpy())
                p_mod_list.append(torch.softmax(l_mod, dim=1).cpu().numpy())
        return (
            np.concatenate(p_mal_list, axis=0),
            np.concatenate(p_dx_list, axis=0),
            np.concatenate(p_mod_list, axis=0),
        )

    def predict_image(
        self,
        image: Image.Image,
        *,
        modality_hint: str | None = None,
        body_site: str | None = None,
        tta: bool = False,
    ) -> Any:
        from medresearch_platform.vision.cnn_multitask import (
            SAFETY_NOTICE_EN,
            SAFETY_NOTICE_FR,
            SkinPredictionResult,
        )
        from medresearch_platform.vision.quality import assess_image_quality

        quality = assess_image_quality(image, check_skin_area=True)
        rgb = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float64) / 255.0
        rec = DermCatalogRecord(
            image_id="live_upload",
            dataset="live",
            modality=modality_hint or "clinical",
            thumb_path=None,
            raw_path=None,
            skin_tone="UNKNOWN",
            diagnosis_raw=None,
            malignancy_group="unknown",
            body_site=body_site,
            license_name="RESEARCH",
            quality_ok=quality.quality_ok,
            confirmation=None,
            group_key="live_upload",
            split_assignment="live",
        )
        p_mal_arr, p_dx_arr, p_mod_arr = self.predict_batch([(arr, -1, rec)], batch_size=1, tta=tta)
        p_mal = float(p_mal_arr[0])
        dx_probs = p_dx_arr[0]
        mod_probs = p_mod_arr[0]
        conf = max(p_mal, 1.0 - p_mal)
        abstained = (not quality.quality_ok) or (conf < self.abstain_min_confidence)
        if abstained:
            risk_band = "abstained"
        elif p_mal >= 0.65:
            risk_band = "high"
        elif p_mal >= 0.35:
            risk_band = "medium"
        else:
            risk_band = "low"

        top3_idx = np.argsort(-dx_probs)[:3]
        top3 = [
            {"rank": r + 1, "diagnosis": CORE_DX_CLASSES[int(idx)], "probability": round(float(dx_probs[int(idx)]), 4)}
            for r, idx in enumerate(top3_idx)
        ]
        cam = self.compute_gradcam(arr, rec)
        ab_en = None
        ab_fr = None
        if not quality.quality_ok:
            ab_en = "; ".join(quality.reasons_en) or "Image quality insufficient — please take a new photo or see a dermatologist."
            ab_fr = "; ".join(quality.reasons_fr) or "Qualité insuffisante — veuillez reprendre une photo ou consulter un dermatologue."
        elif abstained:
            ab_en = f"Low calibrated confidence ({conf:.2f} < {self.abstain_min_confidence:.2f}) — please take a new photo or see a dermatologist."
            ab_fr = f"Confiance calibrée insuffisante ({conf:.2f} < {self.abstain_min_confidence:.2f}) — veuillez reprendre une photo ou consulter un dermatologue."

        return SkinPredictionResult(
            malignancy_probability=round(p_mal, 4),
            risk_band=risk_band,
            top3_diagnoses=top3 if (not abstained and self.dx_head_supported) else [],
            predicted_modality=SKIN_MODALITIES[int(np.argmax(mod_probs))],
            modality_probabilities={m: round(float(mod_probs[i]), 4) for i, m in enumerate(SKIN_MODALITIES)},
            energy_score=round(float(-np.log(max(conf, 1e-6))), 4),
            confidence=round(conf, 4),
            abstained=abstained,
            abstain_reason_en=ab_en,
            abstain_reason_fr=ab_fr,
            needs_review=True,
            quality=quality,
            heatmap_grid=cam.tolist(),
            safety_notice_en=SAFETY_NOTICE_EN,
            safety_notice_fr=SAFETY_NOTICE_FR,
            model_name=self.architecture_name,
            pretrained_weights_used=False,
        )

    def compute_gradcam_heatmap(self, image: Image.Image) -> tuple[np.ndarray, float]:
        rgb = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float64) / 255.0
        cam = self.compute_gradcam(arr)
        return cam, float(np.mean(cam))

    def render_gradcam_overlay_png(self, image: Image.Image, cam_grid: np.ndarray) -> bytes:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float64) / 255.0
        return render_gradcam_overlay_png_from_cam(arr, cam_grid)

    def compute_gradcam(self, arr: np.ndarray, rec: DermCatalogRecord | None = None) -> np.ndarray:
        """Real forward/backward hook Grad-CAM on `stage4` upsampled to `(image_size, image_size)`."""
        import torch

        self.net.eval()
        acts: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []

        def fwd_hook(_m: Any, _inp: Any, out: torch.Tensor) -> None:
            acts.append(out)

        def bwd_hook(_m: Any, _gin: Any, gout: tuple[torch.Tensor, ...]) -> None:
            grads.append(gout[0])

        h_f = self.net.stage4.register_forward_hook(fwd_hook)
        h_b = self.net.stage4.register_full_backward_hook(bwd_hook)
        dev = self._device
        try:
            x = self._arr_to_tensor(arr).unsqueeze(0).to(dev).requires_grad_(True)
            m = (
                torch.from_numpy(encode_metadata_vector(rec, metadata_mode=self.metadata_mode)).unsqueeze(0).to(dev)
                if (self.include_metadata and rec is not None)
                else (torch.zeros((1, 8), dtype=torch.float32, device=dev) if self.include_metadata else None)
            )
            self.net.zero_grad()
            l_mal, _, _ = self.net(x, m)
            l_mal.sum().backward()

            a = acts[0].detach()[0]  # [C, h, w]
            g = grads[0].detach()[0]  # [C, h, w]
            weights = g.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * a).sum(dim=0))
            if float(cam.max()) > 0:
                cam = cam / float(cam.max())
            cam_up = torch.nn.functional.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            return np.asarray(cam_up.cpu().numpy(), dtype=np.float64)
        finally:
            h_f.remove()
            h_b.remove()

    def save(self, out_dir: str | Path, *, corpus_hash: str = "") -> dict[str, Any]:
        import torch

        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        weights_path = p / "weights_torch.pt"
        cpu_state = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        torch.save(cpu_state, weights_path)
        import subprocess

        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            git_commit = "unknown"

        trained_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        manifest = {
            "architecture_name": self.architecture_name,
            "architecture_variant": self.architecture_variant,
            "backend": f"pytorch_{self._device.type}",
            "git_commit": git_commit,
            "image_size": self.image_size,
            "stage_channels": [32, 64, 128, 256],
            "include_metadata": self.include_metadata,
            "metadata_mode": self.metadata_mode,
            "body_site_dropout": self.body_site_dropout,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
            "temperature_mal": self.temperature_mal,
            "temperature_dx": self.temperature_dx,
            "abstain_min_confidence": self.abstain_min_confidence,
            "dx_head_supported": self.dx_head_supported,
            "train_history": self.train_history,
            "initial_weights_sha256": self.initial_weights_sha256,
            "trained_weights_sha256": trained_sha,
            "corpus_hash": corpus_hash,
            "pretrained_weights_used": False,
            "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (p / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, model_dir: str | Path, *, device: str | None = "cpu") -> PyTorchDermResNet:
        import torch

        p = Path(model_dir)
        manifest_path = p / "manifest.json"
        weights_path = p / "weights_torch.pt"
        if not manifest_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"PyTorch skin model files missing in {p}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = str(manifest.get("trained_weights_sha256") or "")
        actual_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if not expected_sha or expected_sha != actual_sha:
            raise ValueError(f"PyTorch skin model SHA-256 mismatch in {p}")
        state_dict = torch.load(weights_path, map_location="cpu")
        n_dx = int(state_dict["fc_dx.weight"].shape[0]) if "fc_dx.weight" in state_dict else len(CORE_DX_CLASSES)
        obj = cls(
            image_size=int(manifest.get("image_size", 96)),
            include_metadata=bool(manifest.get("include_metadata", True)),
            metadata_mode=manifest.get("metadata_mode"),
            body_site_dropout=float(manifest.get("body_site_dropout", 0.0)),
            seed=int(manifest.get("seed", 1)),
            n_dx=n_dx,
            device=device,
            architecture_variant=str(manifest.get("architecture_variant", "resnet_v1")),
        )
        obj.net.load_state_dict({k: v.to(obj._device) for k, v in state_dict.items()})
        obj.net.eval()
        obj.temperature_mal = float(manifest.get("temperature_mal", 1.0))
        obj.temperature_dx = float(manifest.get("temperature_dx", 1.0))
        obj.abstain_min_confidence = float(manifest.get("abstain_min_confidence", 0.55))
        obj.dx_head_supported = bool(manifest.get("dx_head_supported", False))
        obj.train_history = dict(manifest.get("train_history") or {})
        return obj


def verify_gradcam_adebayo_and_lesion_center(
    model: PyTorchDermResNet,
    samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
    *,
    max_images: int = 220,
) -> dict[str, float]:
    """
    1. Adebayo et al. (2018) model-parameter randomisation sanity test: compare Grad-CAM of trained model vs random-init model.
    2. On dermoscopic images, compute fraction of Grad-CAM mass inside the central lesion region (r <= 0.35*H).
    """
    rand_model = PyTorchDermResNet(image_size=model.image_size, include_metadata=model.include_metadata, seed=model.seed + 999)
    l1_shifts: list[float] = []
    center_shares: list[float] = []

    h = model.image_size
    yy, xx = np.ogrid[:h, :h]
    cy, cx = h / 2.0, h / 2.0
    center_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (0.35 * h) ** 2

    for arr, _, rec in samples[:max_images]:
        cam_tr = model.compute_gradcam(arr, rec)
        cam_rd = rand_model.compute_gradcam(arr, rec)
        l1_shifts.append(float(np.mean(np.abs(cam_tr - cam_rd))))
        if rec.modality == "dermoscopic":
            tot = float(np.sum(cam_tr))
            if tot > 1e-8:
                center_shares.append(float(np.sum(cam_tr[center_mask]) / tot))

    return {
        "adebayo_mean_l1_shift": round(float(np.mean(l1_shifts)) if l1_shifts else 0.0, 4),
        "dermoscopic_center_heat_share": round(float(np.mean(center_shares)) if center_shares else 0.0, 4),
        "n_dermoscopic_checked": float(len(center_shares)),
    }


def render_gradcam_overlay_png_from_cam(arr: np.ndarray, cam_grid: np.ndarray) -> bytes:
    import io

    rgb_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    base_im = Image.fromarray(rgb_u8, mode="RGB")
    cam = np.clip(cam_grid, 0.0, 1.0)
    heat_rgb = np.zeros((cam.shape[0], cam.shape[1], 3), dtype=np.uint8)
    heat_rgb[..., 0] = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    heat_rgb[..., 1] = np.clip((1.0 - np.abs(cam - 0.5) * 2.0) * 180.0, 0, 255).astype(np.uint8)
    heat_rgb[..., 2] = np.clip((1.0 - cam) * 200.0, 0, 255).astype(np.uint8)
    heat_im = Image.fromarray(heat_rgb, mode="RGB").resize(base_im.size, Image.Resampling.BILINEAR)
    blended = Image.blend(base_im, heat_im, alpha=0.45)
    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    return buf.getvalue()
