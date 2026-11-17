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
from medresearch_platform.vision.classical_skin import CORE_DX_CLASSES

CORE_DX_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CORE_DX_CLASSES)}
SKIN_MODALITIES: tuple[str, ...] = ("clinical", "dermoscopic", "tbp_tile")


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


def _build_resnet_module(include_metadata: bool, n_dx: int, n_mod: int) -> Any:
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
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
            out = self.bn2(self.conv2(out))
            out = self.relu(out + self.shortcut(x))
            return out

    class DermResNet4Stage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.stage1 = ResBlock(32, 32, stride=1)
            self.stage2 = ResBlock(32, 64, stride=2)
            self.stage3 = ResBlock(64, 128, stride=2)
            self.stage4 = ResBlock(128, 256, stride=2)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(0.2)
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


def encode_metadata_vector(rec: DermCatalogRecord) -> np.ndarray:
    """8-dim metadata vector: modality (3) + body_site group (5)."""
    vec = np.zeros(8, dtype=np.float32)
    if rec.modality in SKIN_MODALITIES:
        vec[SKIN_MODALITIES.index(rec.modality)] = 1.0
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
        seed: int = 1,
    ) -> None:
        import torch

        self.image_size = image_size
        self.include_metadata = include_metadata
        self.seed = seed
        self.architecture_name = "PyTorchDermResNet4Stage-v1.0.0"
        self.temperature_mal = 1.0
        self.temperature_dx = 1.0
        self.abstain_min_confidence = 0.55

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.net = _build_resnet_module(include_metadata, len(CORE_DX_CLASSES), len(SKIN_MODALITIES))
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

        t = torch.from_numpy(np.asarray(arr, dtype=np.float32).transpose(2, 0, 1))
        if t.shape[1] != self.image_size or t.shape[2] != self.image_size:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return t

    def fit(
        self,
        train_samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
        val_samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
        *,
        epochs: int = 12,
        batch_size: int = 48,
        lr: float = 1e-3,
        patience: int = 4,
    ) -> dict[str, Any]:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

        t0 = time.perf_counter()
        rng = random.Random(self.seed)

        def _pack(samples: list[tuple[np.ndarray, int, DermCatalogRecord]]) -> tuple[torch.Tensor, ...]:
            xs = torch.stack([self._arr_to_tensor(a) for a, _, _ in samples], dim=0)
            y_mal = torch.tensor([lbl for _, lbl, _ in samples], dtype=torch.long)
            y_dx = torch.tensor([CORE_DX_TO_IDX.get(r.diagnosis_harmonised, CORE_DX_TO_IDX["other"]) for _, _, r in samples], dtype=torch.long)
            y_mod = torch.tensor([SKIN_MODALITIES.index(r.modality) if r.modality in SKIN_MODALITIES else 0 for _, _, r in samples], dtype=torch.long)
            meta = torch.stack([torch.from_numpy(encode_metadata_vector(r)) for _, _, r in samples], dim=0)
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
        sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=min(len(y_mal_tr), 12000), replacement=True)
        loader = DataLoader(TensorDataset(x_tr, y_mal_tr, y_dx_tr, y_mod_tr, m_tr), batch_size=batch_size, sampler=sampler)

        bce = nn.BCEWithLogitsLoss(reduction="none")
        ce = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

        best_val_auroc = -1.0
        best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        stale = 0
        history: list[dict[str, float]] = []

        for ep in range(1, epochs + 1):
            ep_t0 = time.perf_counter()
            self.net.train()
            ep_loss = 0.0
            n_batches = 0
            for xb, ymb, ydb, ymodb, mb in loader:
                # Augmentations: flip H/V, 90-deg rot, subtle brightness/contrast jitter
                if rng.random() < 0.5:
                    xb = torch.flip(xb, dims=[3])
                if rng.random() < 0.3:
                    xb = torch.flip(xb, dims=[2])
                if rng.random() < 0.5:
                    k_rot = rng.choice([1, 2, 3])
                    xb = torch.rot90(xb, k=k_rot, dims=[2, 3])
                gain = rng.uniform(0.9, 1.1)
                bias = rng.uniform(-0.05, 0.05)
                xb = torch.clamp(xb * gain + bias, 0.0, 1.0)

                opt.zero_grad()
                l_mal, l_dx, l_mod = self.net(xb, mb if self.include_metadata else None)
                mask = (ymb == 0) | (ymb == 1)
                if mask.any():
                    loss_mal = bce(l_mal[mask], ymb[mask].float()).mean()
                else:
                    loss_mal = torch.tensor(0.0)
                loss_dx = ce(l_dx, ydb)
                loss_mod = ce(l_mod, ymodb)
                loss = loss_mal + 0.35 * loss_dx + 0.15 * loss_mod
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                n_batches += 1
            scheduler.step()

            self.net.eval()
            with torch.no_grad():
                vl_mal, _, _ = self.net(x_va, m_va if self.include_metadata else None)
                vp_mal = torch.sigmoid(vl_mal).cpu().numpy()
                val_auroc = _compute_binary_auroc(y_mal_va.numpy(), vp_mal)

            ep_sec = round(time.perf_counter() - ep_t0, 2)
            mean_loss = round(ep_loss / max(1, n_batches), 4)
            print(f"[seed={self.seed} meta={self.include_metadata}] epoch {ep}/{epochs}: loss={mean_loss:.4f} val_auroc={val_auroc:.4f} ({ep_sec}s)", flush=True)
            history.append({"epoch": float(ep), "loss": mean_loss, "val_auroc": round(val_auroc, 4), "sec": ep_sec})

            if val_auroc > best_val_auroc + 1e-4:
                best_val_auroc = val_auroc
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        self.net.load_state_dict(best_state)
        self.net.eval()

        # Fit scalar temperature on val malignancy logits (masked)
        with torch.no_grad():
            vl_mal, vl_dx, _ = self.net(x_va, m_va if self.include_metadata else None)
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

            best_tdx = 1.0
            best_nll_dx = float("inf")
            for t in np.linspace(0.5, 2.5, 21):
                nll_dx = float(ce(vl_dx / float(t), y_dx_va).item())
                if nll_dx < best_nll_dx:
                    best_nll_dx = nll_dx
                    best_tdx = float(t)
            self.temperature_dx = round(best_tdx, 3)

        return {
            "seed": self.seed,
            "include_metadata": self.include_metadata,
            "parameter_count": self.parameter_count,
            "epochs_run": len(history),
            "best_val_auroc": round(best_val_auroc, 4),
            "temperature_mal": self.temperature_mal,
            "temperature_dx": self.temperature_dx,
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "history": history,
            "initial_weights_sha256": self.initial_weights_sha256,
            "pretrained_weights_used": False,
        }

    def predict_batch(
        self,
        samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
        *,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        self.net.eval()
        p_mal_list, p_dx_list, p_mod_list = [], [], []
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                chunk = samples[i : i + batch_size]
                xb = torch.stack([self._arr_to_tensor(a) for a, _, _ in chunk], dim=0)
                mb = torch.stack([torch.from_numpy(encode_metadata_vector(r)) for _, _, r in chunk], dim=0)
                l_mal, l_dx, l_mod = self.net(xb, mb if self.include_metadata else None)
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
        p_mal_arr, p_dx_arr, p_mod_arr = self.predict_batch([(arr, -1, rec)], batch_size=1)
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
            top3_diagnoses=top3 if not abstained else [],
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
        try:
            x = self._arr_to_tensor(arr).unsqueeze(0).requires_grad_(True)
            m = (
                torch.from_numpy(encode_metadata_vector(rec)).unsqueeze(0)
                if (self.include_metadata and rec is not None)
                else (torch.zeros((1, 8), dtype=torch.float32) if self.include_metadata else None)
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
        torch.save(self.net.state_dict(), weights_path)
        trained_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        manifest = {
            "architecture_name": self.architecture_name,
            "backend": "pytorch_cpu",
            "image_size": self.image_size,
            "stage_channels": [32, 64, 128, 256],
            "include_metadata": self.include_metadata,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
            "temperature_mal": self.temperature_mal,
            "temperature_dx": self.temperature_dx,
            "abstain_min_confidence": self.abstain_min_confidence,
            "initial_weights_sha256": self.initial_weights_sha256,
            "trained_weights_sha256": trained_sha,
            "corpus_hash": corpus_hash,
            "pretrained_weights_used": False,
            "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (p / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, model_dir: str | Path) -> PyTorchDermResNet:
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
        obj = cls(
            image_size=int(manifest.get("image_size", 96)),
            include_metadata=bool(manifest.get("include_metadata", True)),
            seed=int(manifest.get("seed", 1)),
        )
        obj.net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        obj.net.eval()
        obj.temperature_mal = float(manifest.get("temperature_mal", 1.0))
        obj.temperature_dx = float(manifest.get("temperature_dx", 1.0))
        obj.abstain_min_confidence = float(manifest.get("abstain_min_confidence", 0.55))
        return obj


def verify_gradcam_adebayo_and_lesion_center(
    model: PyTorchDermResNet,
    samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
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

    for arr, _, rec in samples[:24]:
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
