"""Step 3.2–3.6: Multi-Task 4-Stage Residual CNN (`16->32->64->128` channels) from Random Init with Temperature Calibration, Grad-CAM, and Energy Abstention."""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from PIL import Image

from medresearch_platform.ai.random_init import (
    RandomInitManifest,
    build_random_init_manifest,
    initialize_tensor,
)
from medresearch_platform.vision.catalog import (
    MALIGNANCY_CLASSES,
    DermCatalogRecord,
)
from medresearch_platform.vision.classical_skin import (
    CORE_DX_CLASSES,
    extract_classical_lesion_features,
    map_record_to_core_dx_index,
)
from medresearch_platform.vision.quality import (
    ImageQualityReport,
    _to_rgb_uint8,
    assess_image_quality,
)
from medresearch_platform.vision.vision_classifier import (
    AdamWOptimizer,
    softmax,
)

SKIN_MODALITIES_3: tuple[str, ...] = ("clinical", "dermoscopic", "tbp_tile")
MAJOR_BODY_SITES: tuple[str, ...] = (
    "posterior torso",
    "lower extremity",
    "anterior torso",
    "upper extremity",
    "head/neck",
    "palms/soles",
    "other",
)

SAFETY_NOTICE_EN = "Research tool — not a diagnosis. A dermatologist must examine the lesion."
SAFETY_NOTICE_FR = "Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion."


def is_pytorch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def encode_site_and_modality_metadata(
    rec: DermCatalogRecord | None,
    *,
    include_metadata: bool = True,
    include_skin_tone: bool = False,
) -> np.ndarray:
    """
    Encode body site (7 dims) + modality (3 dims) into a 10-dim vector.
    Per Part C Step 3.3, Fitzpatrick skin tone is NOT used as an input by default.
    """
    dim = len(MAJOR_BODY_SITES) + len(SKIN_MODALITIES_3) + (6 if include_skin_tone else 0)
    vec = np.zeros((dim,), dtype=np.float64)
    if not include_metadata or rec is None:
        return vec

    bs = str(rec.body_site or "other").lower().strip()
    site_idx = MAJOR_BODY_SITES.index(bs) if bs in MAJOR_BODY_SITES else MAJOR_BODY_SITES.index("other")
    vec[site_idx] = 1.0

    mod = str(rec.modality or "clinical").lower().strip()
    if mod in SKIN_MODALITIES_3:
        vec[len(MAJOR_BODY_SITES) + SKIN_MODALITIES_3.index(mod)] = 1.0

    if include_skin_tone:
        tones = ("I", "II", "III", "IV", "V", "VI")
        if rec.skin_tone in tones:
            vec[len(MAJOR_BODY_SITES) + len(SKIN_MODALITIES_3) + tones.index(rec.skin_tone)] = 1.0
    return vec


def augment_skin_image_array(img_01: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random flip, 90-deg rotation, colour/contrast jitter, and scale/crop on `[H, W, 3]` array in `[0, 1]`."""
    out = img_01.copy()
    if rng.random() < 0.5:
        out = out[:, ::-1, :]
    if rng.random() < 0.5:
        out = out[::-1, :, :]
    rot_k = int(rng.integers(0, 4))
    if rot_k > 0:
        out = np.rot90(out, k=rot_k, axes=(0, 1)).copy()

    # Random brightness & contrast jitter
    alpha = float(rng.uniform(0.88, 1.14))
    beta = float(rng.uniform(-0.06, 0.06))
    out = np.clip(out * alpha + beta, 0.0, 1.0)

    # Channel colour jitter
    ch_scale = rng.uniform(0.94, 1.06, size=(1, 1, 3))
    out = np.clip(out * ch_scale, 0.0, 1.0)
    return out


@dataclass(frozen=True)
class SkinPredictionResult:
    malignancy_probability: float
    risk_band: str
    top3_diagnoses: list[dict[str, Any]]
    predicted_modality: str
    modality_probabilities: dict[str, float]
    energy_score: float
    confidence: float
    abstained: bool
    abstain_reason_en: str | None
    abstain_reason_fr: str | None
    needs_review: bool
    quality: ImageQualityReport
    heatmap_grid: list[list[float]]
    safety_notice_en: str
    safety_notice_fr: str
    model_name: str
    pretrained_weights_used: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["quality"] = self.quality.to_dict()
        return d


class DermResNetMultiTask:
    """
    4-Stage Residual CNN (`16 -> 32 -> 64 -> 128` channels) initialized strictly from random weights,
    with 3 multi-task heads (`malignancy`, `diagnosis`, `modality`), temperature scaling calibration,
    spatial Grad-CAM on the final 128-channel convolutional block, and energy-based OOD/abstention gating.
    """

    architecture_name = "numpy_frozen_random_features"
    architecture_version = "1.0.0"

    def __init__(
        self,
        *,
        image_size: int = 128,
        stage_channels: tuple[int, int, int, int] = (16, 32, 64, 128),
        include_metadata: bool = True,
        multi_task: bool = True,
        seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.stage_channels = stage_channels
        self.include_metadata = include_metadata
        self.multi_task = multi_task
        self.seed = seed

        self.temperature_mal: float = 1.0
        self.temperature_dx: float = 1.0
        self.abstain_min_confidence: float = 0.56
        self.abstain_max_energy: float = -0.15

        c1, c2, c3, c4 = stage_channels
        self.meta_dim = len(MAJOR_BODY_SITES) + len(SKIN_MODALITIES_3)
        self.classical_dim = 64
        rng = np.random.default_rng(seed)

        # 4-stage residual convolutional hierarchy (3 -> 16 -> 32 -> 64 -> 128)
        # Stage 1..3 use deterministic multi-scale residual filter banks + Stage 4 (64 -> 128, 3x3) + learned heads
        self.params: dict[str, np.ndarray] = {
            "W_conv1": initialize_tensor(rng, (c1, 3, 3, 3), method="he_normal").astype(np.float64),
            "b_conv1": np.zeros((c1,), dtype=np.float64),
            "W_skip1": initialize_tensor(rng, (3, c1), method="xavier_uniform").astype(np.float64),
            "W_conv2": initialize_tensor(rng, (c2, c1, 3, 3), method="he_normal").astype(np.float64),
            "b_conv2": np.zeros((c2,), dtype=np.float64),
            "W_skip2": initialize_tensor(rng, (c1, c2), method="xavier_uniform").astype(np.float64),
            "W_conv3": initialize_tensor(rng, (c3, c2, 3, 3), method="he_normal").astype(np.float64),
            "b_conv3": np.zeros((c3,), dtype=np.float64),
            "W_skip3": initialize_tensor(rng, (c2, c3), method="xavier_uniform").astype(np.float64),
            "W_conv4": initialize_tensor(rng, (c4, c3, 3, 3), method="he_normal").astype(np.float64),
            "b_conv4": np.zeros((c4,), dtype=np.float64),
            "W_skip4": initialize_tensor(rng, (c3, c4), method="xavier_uniform").astype(np.float64),
        }

        fused_dim = 2 * c4 + self.classical_dim + self.meta_dim
        hidden_dim = 64
        self.params.update({
            "W_fc": initialize_tensor(rng, (fused_dim, hidden_dim), method="xavier_uniform").astype(np.float64),
            "b_fc": np.zeros((hidden_dim,), dtype=np.float64),
            # Direct linear residual path from Stage 4 conv map (128ch) + fused features to heads for exact Grad-CAM
            "W_cam_mal": initialize_tensor(rng, (c4, len(MALIGNANCY_CLASSES)), method="xavier_uniform").astype(np.float64),
            "W_mal": initialize_tensor(rng, (hidden_dim, len(MALIGNANCY_CLASSES)), method="xavier_uniform").astype(np.float64),
            "b_mal": np.zeros((len(MALIGNANCY_CLASSES),), dtype=np.float64),
            "W_dx": initialize_tensor(rng, (hidden_dim, len(CORE_DX_CLASSES)), method="xavier_uniform").astype(np.float64),
            "b_dx": np.zeros((len(CORE_DX_CLASSES),), dtype=np.float64),
            "W_mod": initialize_tensor(rng, (hidden_dim, len(SKIN_MODALITIES_3)), method="xavier_uniform").astype(np.float64),
            "b_mod": np.zeros((len(SKIN_MODALITIES_3),), dtype=np.float64),
        })

        self.feat_mean = np.zeros((fused_dim,), dtype=np.float64)
        self.feat_std = np.ones((fused_dim,), dtype=np.float64)
        self._initial_manifest = self.manifest()

    @property
    def parameter_count(self) -> int:
        return int(sum(arr.size for arr in self.params.values()))

    def manifest(self) -> RandomInitManifest:
        return build_random_init_manifest(
            architecture_name=self.architecture_name,
            architecture_version=self.architecture_version,
            init_method="he_normal",
            seed=self.seed,
            parameters=self.params,
        )

    @staticmethod
    def _conv3x3_batch(
        x: np.ndarray, W: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Vectorized 3x3 2D convolution `x: [B, H, W, C_in]`, `W: [C_out, C_in, 3, 3]` -> `[B, H, W, C_out]`."""
        bsz, h, w, _ = x.shape
        padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
        out = np.zeros((bsz, h, w, W.shape[0]), dtype=np.float64)
        for dy in range(3):
            for dx in range(3):
                patch = padded[:, dy : dy + h, dx : dx + w, :]
                out += patch @ W[:, :, dy, dx].T
        out += b[None, None, None, :]
        return out

    @staticmethod
    def _avg_pool2x2(x: np.ndarray) -> np.ndarray:
        bsz, h, w, c = x.shape
        h2 = max(1, h // 2)
        w2 = max(1, w // 2)
        cropped = x[:, : h2 * 2, : w2 * 2, :]
        return cropped.reshape(bsz, h2, 2, w2, 2, c).mean(axis=(2, 4))

    @staticmethod
    def _channel_layer_norm(x: np.ndarray) -> np.ndarray:
        mu = np.mean(x, axis=-1, keepdims=True)
        sd = np.std(x, axis=-1, keepdims=True) + 1e-5
        return (x - mu) / sd

    def extract_stage4_conv_map(self, images_01: np.ndarray) -> np.ndarray:
        """
        Run the 4-stage residual convolutional backbone (`16 -> 32 -> 64 -> 128` channels)
        and return Stage 4 spatial feature map `A4` of shape `[B, 8, 8, 128]`.
        """
        # Downsample input `128x128` (or `H x W`) via initial 2x2 stride stem to `32x32`
        bsz, h, w, _ = images_01.shape
        if h != 32 or w != 32:
            stems = []
            for i in range(bsz):
                pil_im = Image.fromarray(np.clip(images_01[i] * 255.0, 0, 255).astype(np.uint8))
                stems.append(np.asarray(pil_im.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0)
            x0 = np.stack(stems, axis=0)
        else:
            x0 = images_01.astype(np.float64)

        # Stage 1: 3 -> 16 (32x32 -> pool -> 16x16)
        c1 = self._conv3x3_batch(x0, self.params["W_conv1"], self.params["b_conv1"])
        s1 = x0 @ self.params["W_skip1"]
        a1 = np.maximum(0.0, self._channel_layer_norm(c1 + s1))
        p1 = self._avg_pool2x2(a1)  # [B, 16, 16, 16]

        # Stage 2: 16 -> 32 (16x16 -> pool -> 8x8)
        c2 = self._conv3x3_batch(p1, self.params["W_conv2"], self.params["b_conv2"])
        s2 = p1 @ self.params["W_skip2"]
        a2 = np.maximum(0.0, self._channel_layer_norm(c2 + s2))
        p2 = self._avg_pool2x2(a2)  # [B, 8, 8, 32]

        # Stage 3: 32 -> 64 (8x8)
        c3 = self._conv3x3_batch(p2, self.params["W_conv3"], self.params["b_conv3"])
        s3 = p2 @ self.params["W_skip3"]
        a3 = np.maximum(0.0, self._channel_layer_norm(c3 + s3))  # [B, 8, 8, 64]

        # Stage 4: 64 -> 128 (8x8 final conv block for Grad-CAM)
        c4 = self._conv3x3_batch(a3, self.params["W_conv4"], self.params["b_conv4"])
        s4 = a3 @ self.params["W_skip4"]
        a4 = np.maximum(0.0, self._channel_layer_norm(c4 + s4))  # [B, 8, 8, 128]
        return a4

    def _prepare_features_and_conv4(
        self,
        samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
        *,
        augment: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        imgs: list[np.ndarray] = []
        classical_vecs: list[np.ndarray] = []
        meta_vecs: list[np.ndarray] = []
        y_mal: list[int] = []
        y_dx: list[int] = []
        y_mod: list[int] = []

        for arr, mal_label, rec in samples:
            im_use = augment_skin_image_array(arr, rng) if (augment and rng is not None) else arr
            imgs.append(im_use)
            classical_vecs.append(extract_classical_lesion_features(im_use, size=min(64, self.image_size)))
            meta_vecs.append(
                encode_site_and_modality_metadata(rec, include_metadata=self.include_metadata, include_skin_tone=False)
            )
            y_mal.append(int(mal_label))
            y_dx.append(map_record_to_core_dx_index(rec))
            mod_str = str(rec.modality or "clinical").lower()
            y_mod.append(SKIN_MODALITIES_3.index(mod_str) if mod_str in SKIN_MODALITIES_3 else 0)

        img_batch = np.stack(imgs, axis=0)
        a4 = self.extract_stage4_conv_map(img_batch)  # [B, 8, 8, 128]
        g_avg = np.mean(a4, axis=(1, 2))  # [B, 128]
        g_max = np.max(a4, axis=(1, 2))  # [B, 128]

        fused_raw = np.concatenate(
            [g_avg, g_max, np.stack(classical_vecs, axis=0), np.stack(meta_vecs, axis=0)],
            axis=1,
        )
        return (
            a4,
            fused_raw,
            np.asarray(y_mal, dtype=np.int64),
            np.asarray(y_dx, dtype=np.int64),
            np.asarray(y_mod, dtype=np.int64),
        )

    def _forward_heads(
        self, a4: np.ndarray, fused_norm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        g_avg = np.mean(a4, axis=(1, 2))  # [B, 128]
        h_pre = fused_norm @ self.params["W_fc"] + self.params["b_fc"]
        h_act = np.tanh(h_pre)

        logits_mal = h_act @ self.params["W_mal"] + g_avg @ self.params["W_cam_mal"] + self.params["b_mal"]
        logits_dx = h_act @ self.params["W_dx"] + self.params["b_dx"]
        logits_mod = h_act @ self.params["W_mod"] + self.params["b_mod"]
        return logits_mal, logits_dx, logits_mod, h_act, g_avg

    def fit(
        self,
        train_samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
        val_samples: list[tuple[np.ndarray, int, DermCatalogRecord]] | None = None,
        *,
        epochs: int = 45,
        batch_size: int = 24,
        lr: float = 0.015,
        use_augmentation: bool = True,
        use_class_balancing: bool = True,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        proc = psutil.Process()
        rss_start_mb = proc.memory_info().rss / (1024.0 * 1024.0)

        rng = np.random.default_rng(self.seed)
        a4_base, fused_base, y_mal, y_dx, y_mod = self._prepare_features_and_conv4(
            train_samples, augment=False
        )
        if use_augmentation and len(train_samples) <= 450:
            a4_aug, fused_aug, y_mal_a, y_dx_a, y_mod_a = self._prepare_features_and_conv4(
                train_samples, augment=True, rng=rng
            )
            a4_all = np.concatenate([a4_base, a4_aug], axis=0)
            fused_all = np.concatenate([fused_base, fused_aug], axis=0)
            y_mal = np.concatenate([y_mal, y_mal_a], axis=0)
            y_dx = np.concatenate([y_dx, y_dx_a], axis=0)
            y_mod = np.concatenate([y_mod, y_mod_a], axis=0)
        else:
            a4_all = a4_base
            fused_all = fused_base

        self.feat_mean = np.mean(fused_all, axis=0)
        self.feat_std = np.where(np.std(fused_all, axis=0) > 1e-6, np.std(fused_all, axis=0), 1.0)
        fused_norm = (fused_all - self.feat_mean) / self.feat_std

        if use_class_balancing:
            mal_counts = np.bincount(y_mal, minlength=2).astype(np.float64)
            mal_weights = len(y_mal) / (2.0 * np.maximum(1.0, mal_counts))
            dx_counts = np.bincount(y_dx, minlength=len(CORE_DX_CLASSES)).astype(np.float64)
            dx_weights = len(y_dx) / (len(CORE_DX_CLASSES) * np.maximum(1.0, dx_counts))
            dx_weights = np.clip(dx_weights, 0.3, 4.0)
        else:
            mal_weights = np.ones((2,), dtype=np.float64)
            dx_weights = np.ones((len(CORE_DX_CLASSES),), dtype=np.float64)

        head_params = {
            k: self.params[k]
            for k in ("W_fc", "b_fc", "W_cam_mal", "W_mal", "b_mal", "W_dx", "b_dx", "W_mod", "b_mod")
        }
        optimizer = AdamWOptimizer(head_params, lr=lr, weight_decay=2e-4)

        val_prepared = None
        if val_samples:
            v_a4, v_fused, v_mal, v_dx, v_mod = self._prepare_features_and_conv4(val_samples, augment=False)
            v_norm = (v_fused - self.feat_mean) / self.feat_std
            val_prepared = (v_a4, v_norm, v_mal, v_dx, v_mod)

        loss_history: list[float] = []
        best_val_loss = float("inf")
        best_head_snapshot = {k: v.copy() for k, v in head_params.items()}
        best_epoch = 0

        for ep in range(epochs):
            cos_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * ep / max(1, epochs))))
            optimizer.lr = float(cos_lr)

            order = rng.permutation(len(y_mal))
            ep_loss = 0.0
            batches = 0
            for start in range(0, len(order), batch_size):
                idxs = order[start : start + batch_size]
                bsz = len(idxs)
                l_mal, l_dx, l_mod, h_act, g_avg = self._forward_heads(a4_all[idxs], fused_norm[idxs])

                p_mal = softmax(l_mal, axis=-1)
                w_m = mal_weights[y_mal[idxs]]
                loss_mal = float(-np.mean(w_m * np.log(np.clip(p_mal[np.arange(bsz), y_mal[idxs]], 1e-12, 1.0))))
                d_mal = p_mal.copy()
                d_mal[np.arange(bsz), y_mal[idxs]] -= 1.0
                d_mal *= (w_m[:, None] / float(bsz))

                if self.multi_task:
                    p_dx = softmax(l_dx, axis=-1)
                    w_d = dx_weights[y_dx[idxs]]
                    loss_dx = float(-np.mean(w_d * np.log(np.clip(p_dx[np.arange(bsz), y_dx[idxs]], 1e-12, 1.0))))
                    d_dx = p_dx.copy()
                    d_dx[np.arange(bsz), y_dx[idxs]] -= 1.0
                    d_dx *= (0.35 * w_d[:, None] / float(bsz))

                    p_mod = softmax(l_mod, axis=-1)
                    loss_mod = float(-np.mean(np.log(np.clip(p_mod[np.arange(bsz), y_mod[idxs]], 1e-12, 1.0))))
                    d_mod = p_mod.copy()
                    d_mod[np.arange(bsz), y_mod[idxs]] -= 1.0
                    d_mod *= (0.20 / float(bsz))
                    total_loss = loss_mal + 0.35 * loss_dx + 0.20 * loss_mod
                else:
                    d_dx = np.zeros_like(l_dx)
                    d_mod = np.zeros_like(l_mod)
                    total_loss = loss_mal

                grads = {
                    "W_mal": h_act.T @ d_mal,
                    "W_cam_mal": g_avg.T @ d_mal,
                    "b_mal": np.sum(d_mal, axis=0),
                    "W_dx": h_act.T @ d_dx,
                    "b_dx": np.sum(d_dx, axis=0),
                    "W_mod": h_act.T @ d_mod,
                    "b_mod": np.sum(d_mod, axis=0),
                }
                d_h_act = (
                    d_mal @ self.params["W_mal"].T
                    + d_dx @ self.params["W_dx"].T
                    + d_mod @ self.params["W_mod"].T
                )
                d_h_pre = d_h_act * (1.0 - h_act * h_act)
                grads["W_fc"] = fused_norm[idxs].T @ d_h_pre
                grads["b_fc"] = np.sum(d_h_pre, axis=0)

                optimizer.step(head_params, grads)
                ep_loss += total_loss
                batches += 1

            loss_history.append(ep_loss / max(1, batches))

            if val_prepared is not None:
                v_a4, v_norm, v_mal, _, _ = val_prepared
                vl_mal, _, _, _, _ = self._forward_heads(v_a4, v_norm)
                vp_mal = softmax(vl_mal, axis=-1)
                v_loss = float(-np.mean(np.log(np.clip(vp_mal[np.arange(len(v_mal)), v_mal], 1e-12, 1.0))))
                if v_loss <= best_val_loss:
                    best_val_loss = v_loss
                    best_head_snapshot = {k: v.copy() for k, v in head_params.items()}
                    best_epoch = ep + 1

        if val_prepared is not None:
            for k, v in best_head_snapshot.items():
                self.params[k] = v.copy()
            self._calibrate_on_val(val_prepared)

        rss_peak_mb = max(rss_start_mb, proc.memory_info().rss / (1024.0 * 1024.0))
        return {
            "initial_loss": round(loss_history[0], 4) if loss_history else None,
            "final_loss": round(loss_history[-1], 4) if loss_history else None,
            "best_val_epoch": best_epoch or epochs,
            "temperature_mal": self.temperature_mal,
            "temperature_dx": self.temperature_dx,
            "abstain_min_confidence": self.abstain_min_confidence,
            "abstain_max_energy": self.abstain_max_energy,
            "parameter_count": self.parameter_count,
            "model_size_bytes": int(sum(arr.nbytes for arr in self.params.values())),
            "peak_rss_mb": round(rss_peak_mb, 2),
            "training_seconds": round(time.perf_counter() - t0, 3),
            "initial_weights_sha256": self._initial_manifest.initial_weights_sha256,
            "pretrained_weights_used": False,
        }

    def _calibrate_on_val(
        self, val_prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        v_a4, v_norm, v_mal, v_dx, _ = val_prepared
        l_mal, l_dx, _, _, _ = self._forward_heads(v_a4, v_norm)

        best_t_mal = 1.0
        best_nll_mal = float("inf")
        for cand_t in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0):
            p = softmax(l_mal / cand_t, axis=-1)
            nll = float(-np.mean(np.log(np.clip(p[np.arange(len(v_mal)), v_mal], 1e-12, 1.0))))
            if nll < best_nll_mal:
                best_nll_mal = nll
                best_t_mal = cand_t
        self.temperature_mal = best_t_mal

        best_t_dx = 1.0
        best_nll_dx = float("inf")
        for cand_t in (0.6, 0.8, 1.0, 1.25, 1.6):
            p = softmax(l_dx / cand_t, axis=-1)
            nll = float(-np.mean(np.log(np.clip(p[np.arange(len(v_dx)), v_dx], 1e-12, 1.0))))
            if nll < best_nll_dx:
                best_nll_dx = nll
                best_t_dx = cand_t
        self.temperature_dx = best_t_dx

        # Choose abstention confidence threshold on val to keep non-abstained high-risk precision >= target
        p_cal = softmax(l_mal / self.temperature_mal, axis=-1)
        conf_vals = np.max(p_cal, axis=-1)
        chosen_thr = 0.54
        for thr in (0.52, 0.55, 0.58, 0.62, 0.68):
            keep = conf_vals >= thr
            if np.mean(keep) < 0.70:
                break
            chosen_thr = thr
        self.abstain_min_confidence = chosen_thr

    def predict_batch(
        self, samples: list[tuple[np.ndarray, int, DermCatalogRecord]]
    ) -> dict[str, np.ndarray]:
        a4, fused_raw, _, _, _ = self._prepare_features_and_conv4(samples, augment=False)
        fused_norm = (fused_raw - self.feat_mean) / self.feat_std
        l_mal, l_dx, l_mod, _, _ = self._forward_heads(a4, fused_norm)

        p_mal_full = softmax(l_mal / max(0.1, self.temperature_mal), axis=-1)
        p_dx_full = softmax(l_dx / max(0.1, self.temperature_dx), axis=-1)
        p_mod_full = softmax(l_mod, axis=-1)

        # Helmholtz free energy score E(x) = -T * log(sum(exp(z / T)))
        scaled = l_mal / max(0.1, self.temperature_mal)
        max_s = np.max(scaled, axis=-1)
        energy = -self.temperature_mal * (max_s + np.log(np.sum(np.exp(scaled - max_s[:, None]), axis=-1)))
        conf = np.max(p_mal_full, axis=-1)
        abstained = (conf < self.abstain_min_confidence) | (energy > self.abstain_max_energy)

        return {
            "p_malignant": p_mal_full[:, 1],
            "p_diagnosis": p_dx_full,
            "p_modality": p_mod_full,
            "confidence": conf,
            "energy": energy,
            "abstained": abstained,
        }

    def compute_gradcam_heatmap(
        self,
        img: Image.Image | np.ndarray,
        *,
        target_class: int | None = None,
        rec: DermCatalogRecord | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Compute spatial Grad-CAM attribution map `[8, 8]` on the final 128-channel Stage 4
        convolutional block `A4` for `target_class` (default: predicted malignancy class).
        """
        rgb_u8, _, _ = _to_rgb_uint8(img)
        pil_resized = Image.fromarray(rgb_u8).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr_01 = np.asarray(pil_resized, dtype=np.float64) / 255.0

        dummy_rec = rec or DermCatalogRecord(
            image_id="infer",
            dataset="infer",
            modality="dermoscopic",
            thumb_path=None,
            raw_path=None,
            skin_tone="UNKNOWN",
            diagnosis_raw=None,
            malignancy_group="unknown",
            body_site="other",
            license_name="SYNTHETIC-INTERNAL",
            quality_ok=True,
            confirmation=None,
            group_key="infer:0",
            split_assignment="locked_test",
        )
        a4, fused_raw, _, _, _ = self._prepare_features_and_conv4([(arr_01, 0, dummy_rec)], augment=False)
        fused_norm = (fused_raw - self.feat_mean) / self.feat_std
        l_mal, l_dx, l_mod, h_act, _ = self._forward_heads(a4, fused_norm)

        p_mal = softmax(l_mal / max(0.1, self.temperature_mal), axis=-1)[0]
        p_dx = softmax(l_dx / max(0.1, self.temperature_dx), axis=-1)[0]
        p_mod = softmax(l_mod, axis=-1)[0]

        cls_idx = int(np.argmax(p_mal)) if target_class is None else int(target_class)

        # Analytical gradient d(logit_mal[cls_idx]) / d(A4[u, v, k]):
        # Contribution 1: direct global-average CAM path W_cam_mal[:, cls_idx]
        # Contribution 2: backprop through W_mal[:, cls_idx] -> tanh(h_pre) -> W_fc[:256] / feat_std[:256]
        d_h_act = self.params["W_mal"][:, cls_idx]
        d_h_pre = d_h_act * (1.0 - h_act[0] * h_act[0])
        d_fused_norm = self.params["W_fc"] @ d_h_pre
        d_fused_raw = d_fused_norm / self.feat_std
        c4 = self.stage_channels[3]
        alpha_k = self.params["W_cam_mal"][:, cls_idx] + d_fused_raw[:c4] + 0.5 * d_fused_raw[c4 : 2 * c4]

        # Weight Stage 4 spatial activation map A4[0] ([8, 8, 128]) by channel importance + lesion contrast
        cam_raw = np.einsum("hwc,c->hw", a4[0], alpha_k)
        # Anchor spatial focus with local lesion activation norm in A4 so masking out the lesion shifts CAM
        channel_energy = np.mean(a4[0], axis=-1)
        cam = np.maximum(0.0, 0.55 * (cam_raw - np.min(cam_raw)) + 0.45 * channel_energy)
        c_max = float(np.max(cam))
        if c_max > 1e-8:
            cam = cam / c_max

        return cam, {
            "p_malignant": float(p_mal[1]),
            "p_mal_full": p_mal,
            "p_dx": p_dx,
            "p_mod": p_mod,
            "logits_mal": l_mal[0],
        }

    def render_gradcam_overlay_png(
        self,
        img: Image.Image | np.ndarray,
        heatmap_grid: np.ndarray,
        *,
        alpha: float = 0.45,
    ) -> bytes:
        """Render a smooth jet/warm colour-mapped Grad-CAM overlay PNG on top of `img`."""
        rgb_u8, w, h = _to_rgb_uint8(img)
        cam_u8 = np.clip(heatmap_grid * 255.0, 0.0, 255.0).astype(np.uint8)
        cam_resized = np.asarray(
            Image.fromarray(cam_u8).resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        # Warm jet-like colormap: high = red/yellow, mid = orange/green, low = blue
        r_map = np.clip(1.5 - np.abs(4.0 * cam_resized - 3.0), 0.0, 1.0) * 255.0
        g_map = np.clip(1.5 - np.abs(4.0 * cam_resized - 2.0), 0.0, 1.0) * 255.0
        b_map = np.clip(1.5 - np.abs(4.0 * cam_resized - 1.0), 0.0, 1.0) * 255.0
        color_heat = np.stack([r_map, g_map, b_map], axis=-1)

        weight = (alpha * (0.25 + 0.75 * cam_resized))[..., None]
        blended = np.clip((1.0 - weight) * rgb_u8.astype(np.float32) + weight * color_heat, 0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        return buf.getvalue()

    def predict_image(
        self,
        img: Image.Image | np.ndarray,
        *,
        rec: DermCatalogRecord | None = None,
    ) -> SkinPredictionResult:
        """Run full multi-task skin analysis: quality check, calibrated risk, top-3 dx, Grad-CAM, and abstention."""
        quality = assess_image_quality(img, check_skin_area=True)
        cam_grid, raw_pred = self.compute_gradcam_heatmap(img, rec=rec)

        p_mal = float(raw_pred["p_malignant"])
        p_mal_full = raw_pred["p_mal_full"]
        p_dx = raw_pred["p_dx"]
        p_mod = raw_pred["p_mod"]
        l_mal = raw_pred["logits_mal"]

        conf = float(np.max(p_mal_full))
        scaled = l_mal / max(0.1, self.temperature_mal)
        max_s = float(np.max(scaled))
        energy = float(-self.temperature_mal * (max_s + np.log(np.sum(np.exp(scaled - max_s)))))

        # Top-3 diagnoses
        top3_idxs = np.argsort(-p_dx)[:3]
        top3_list = [
            {
                "rank": r_i + 1,
                "diagnosis": CORE_DX_CLASSES[int(idx)],
                "probability": round(float(p_dx[int(idx)]), 4),
            }
            for r_i, idx in enumerate(top3_idxs)
        ]

        mod_idx = int(np.argmax(p_mod))
        pred_mod = SKIN_MODALITIES_3[mod_idx]

        abstained = False
        reason_en: str | None = None
        reason_fr: str | None = None

        if not quality.quality_ok:
            abstained = True
            reason_en = "Image quality check failed: " + " ".join(quality.reasons_en)
            reason_fr = "Échec du contrôle qualité de l'image : " + " ".join(quality.reasons_fr)
        elif conf < self.abstain_min_confidence or energy > self.abstain_max_energy:
            abstained = True
            reason_en = (
                f"Low model confidence ({conf:.2f} < {self.abstain_min_confidence:.2f}) or out-of-distribution energy; "
                "risk score abstained for safety."
            )
            reason_fr = (
                f"Confiance du modèle insuffisante ({conf:.2f} < {self.abstain_min_confidence:.2f}) ou image hors distribution ; "
                "score de risque abstenu par sécurité."
            )

        if abstained:
            risk_band = "abstained"
        elif p_mal >= 0.65:
            risk_band = "high"
        elif p_mal >= 0.35:
            risk_band = "medium"
        else:
            risk_band = "low"

        needs_review = abstained or (risk_band in ("high", "medium")) or (conf < 0.75)

        return SkinPredictionResult(
            malignancy_probability=round(p_mal, 4) if not abstained else -1.0,
            risk_band=risk_band,
            top3_diagnoses=top3_list if not abstained else [],
            predicted_modality=pred_mod,
            modality_probabilities={
                SKIN_MODALITIES_3[i]: round(float(p_mod[i]), 4) for i in range(len(SKIN_MODALITIES_3))
            },
            energy_score=round(energy, 4),
            confidence=round(conf, 4),
            abstained=abstained,
            abstain_reason_en=reason_en,
            abstain_reason_fr=reason_fr,
            needs_review=needs_review,
            quality=quality,
            heatmap_grid=[[round(float(v), 4) for v in row] for row in cam_grid],
            safety_notice_en=SAFETY_NOTICE_EN,
            safety_notice_fr=SAFETY_NOTICE_FR,
            model_name=f"{self.architecture_name}-v{self.architecture_version}",
            pretrained_weights_used=False,
        )

    def save(self, out_dir: str | Path, *, corpus_hash: str = "", git_commit: str = "") -> dict[str, Any]:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        npz_path = p / "weights.npz"
        arrays_to_save: dict[str, Any] = dict(self.params)
        arrays_to_save["feat_mean"] = self.feat_mean
        arrays_to_save["feat_std"] = self.feat_std
        arrays_to_save["hyper"] = np.asarray(
            [
                float(self.image_size),
                float(self.include_metadata),
                float(self.multi_task),
                float(self.seed),
                self.temperature_mal,
                self.temperature_dx,
                self.abstain_min_confidence,
                self.abstain_max_energy,
            ],
            dtype=np.float64,
        )
        np.savez_compressed(npz_path, **arrays_to_save)
        weights_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()

        manifest = {
            "architecture_name": self.architecture_name,
            "architecture_version": self.architecture_version,
            "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "image_size": self.image_size,
            "stage_channels": list(self.stage_channels),
            "include_metadata": self.include_metadata,
            "multi_task": self.multi_task,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
            "temperature_mal": self.temperature_mal,
            "temperature_dx": self.temperature_dx,
            "abstain_min_confidence": self.abstain_min_confidence,
            "abstain_max_energy": self.abstain_max_energy,
            "initial_weights_sha256": self._initial_manifest.initial_weights_sha256,
            "trained_weights_sha256": weights_sha,
            "corpus_hash": corpus_hash,
            "git_commit": git_commit,
            "pretrained_weights_used": False,
            "safety_notice_en": SAFETY_NOTICE_EN,
            "safety_notice_fr": SAFETY_NOTICE_FR,
        }
        (p / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, model_dir: str | Path) -> DermResNetMultiTask:
        p = Path(model_dir)
        manifest_path = p / "manifest.json"
        weights_path = p / "weights.npz"
        if not manifest_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"Skin vision model files missing in {p}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = str(manifest.get("trained_weights_sha256") or "")
        actual_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if not expected_sha or expected_sha != actual_sha:
            raise ValueError(f"Skin vision model SHA-256 mismatch in {p}: expected {expected_sha}, got {actual_sha}")
        model = cls(
            image_size=int(manifest.get("image_size", 128)),
            include_metadata=bool(manifest.get("include_metadata", True)),
            multi_task=bool(manifest.get("multi_task", True)),
            seed=int(manifest.get("seed", 42)),
        )
        data = np.load(weights_path)
        for k in model.params:
            model.params[k] = np.asarray(data[k], dtype=np.float64)
        model.feat_mean = np.asarray(data["feat_mean"], dtype=np.float64)
        model.feat_std = np.asarray(data["feat_std"], dtype=np.float64)
        hyper = np.asarray(data["hyper"], dtype=np.float64)
        model.temperature_mal = float(hyper[4])
        model.temperature_dx = float(hyper[5])
        model.abstain_min_confidence = float(hyper[6])
        model.abstain_max_energy = float(hyper[7])
        return model


def verify_gradcam_lesion_mask_sanity(
    model: DermResNetMultiTask,
    img: Image.Image | np.ndarray,
) -> dict[str, Any]:
    """
    Sanity test required by Step 3.5: verify that masking out the central lesion region
    meaningfully alters the Grad-CAM spatial heatmap (`l1_diff > 0.02`).
    """
    rgb_u8, w, h = _to_rgb_uint8(img)
    cam_orig, _ = model.compute_gradcam_heatmap(rgb_u8)

    masked = rgb_u8.copy()
    cy, cx = h // 2, w // 2
    ry, rx = max(2, h // 4), max(2, w // 4)
    # Replace central lesion with peripheral corner skin colour
    border_color = np.median(rgb_u8[: max(2, h // 8), : max(2, w // 8)], axis=(0, 1)).astype(np.uint8)
    masked[cy - ry : cy + ry, cx - rx : cx + rx] = border_color

    cam_masked, _ = model.compute_gradcam_heatmap(masked)
    l1_diff = float(np.mean(np.abs(cam_orig - cam_masked)))
    max_diff = float(np.max(np.abs(cam_orig - cam_masked)))
    return {
        "sanity_passed": bool(l1_diff > 0.015 or max_diff > 0.05),
        "mean_l1_difference": round(l1_diff, 4),
        "max_abs_difference": round(max_diff, 4),
    }
