"""Step 2: 5-Class Picture Triage Router (`skin_clinical`, `skin_dermoscopic`, `skin_tbp_tile`, `document_scan`, `other`)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from medresearch_platform.ai.random_init import (
    RandomInitManifest,
    build_random_init_manifest,
    initialize_tensor,
)
from medresearch_platform.vision.quality import (
    ImageQualityReport,
    QualityThresholds,
    _to_rgb_uint8,
    assess_image_quality,
    calibrate_quality_thresholds_on_val,
    compute_laplacian_variance,
    estimate_skin_and_lesion_area_ratio,
)
from medresearch_platform.vision.vision_classifier import (
    AdamWOptimizer,
    softmax,
)

ROUTER_CLASSES: tuple[str, ...] = (
    "skin_clinical",
    "skin_dermoscopic",
    "skin_tbp_tile",
    "document_scan",
    "other",
)

ROUTER_ACTIONS: dict[str, str] = {
    "skin_clinical": "skin_vision_pipeline",
    "skin_dermoscopic": "skin_vision_pipeline",
    "skin_tbp_tile": "skin_vision_pipeline",
    "document_scan": "ocr_text_pipeline",
    "other": "reject_unsupported",
}

ROUTER_LABELS_EN: dict[str, str] = {
    "skin_clinical": "Clinical skin photograph (camera / phone)",
    "skin_dermoscopic": "Dermoscopic skin lesion image",
    "skin_tbp_tile": "Total-body photography (TBP) skin tile",
    "document_scan": "Clinical document scan / form → routed to OCR",
    "other": "Unsupported non-skin image",
}

ROUTER_LABELS_FR: dict[str, str] = {
    "skin_clinical": "Photographie cutanée clinique (appareil photo / téléphone)",
    "skin_dermoscopic": "Image dermoscopique de lésion cutanée",
    "skin_tbp_tile": "Tuile photographique corporelle totale (TBP)",
    "document_scan": "Document clinique scanné / formulaire → orienté vers l'OCR",
    "other": "Image non cutanée non prise en charge",
}


@dataclass(frozen=True)
class TriageDecision:
    predicted_class: str
    route_action: str
    confidence: float
    probabilities: dict[str, float]
    label_en: str
    label_fr: str
    quality: ImageQualityReport
    abstained: bool
    abstain_reason_en: str | None
    abstain_reason_fr: str | None
    model_name: str
    pretrained_weights_used: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["quality"] = self.quality.to_dict()
        return d


def extract_router_features(img: Image.Image | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract:
      1) `patch_24x24` normalized RGB tensor `(24, 24, 3)` for the 2D CNN branch, and
      2) `desc_28` structural, chrominance, radial-vignette, and document-scan descriptor vector `(28,)`.
    """
    rgb_u8, w, h = _to_rgb_uint8(img)
    pil_rgb = Image.fromarray(rgb_u8)

    patch_24 = np.asarray(pil_rgb.resize((24, 24), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    eval_64 = np.asarray(pil_rgb.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0

    r = eval_64[..., 0]
    g = eval_64[..., 1]
    b = eval_64[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    sat = cmax - cmin

    # 1. Aspect ratio & resolution geometry (4 dims)
    aspect = min(2.5, float(w) / float(max(1, h)))
    is_square = 1.0 if abs(w - h) <= 2 else 0.0
    aspect_dev = abs(float(w) - float(h)) / float(max(1, max(w, h)))
    min_dim_norm = min(1.0, float(min(w, h)) / 512.0)

    # 2. Document scan indicators (6 dims)
    white_paper_frac = float(np.mean((gray > 0.82) & (sat < 0.08)))
    dark_ink_frac = float(np.mean((gray < 0.32) & (sat < 0.10)))
    bimodal_doc_score = white_paper_frac * min(1.0, dark_ink_frac * 8.0)
    low_chroma_frac = float(np.mean(sat < 0.05))
    horiz_diff = np.abs(gray[:, 1:] - gray[:, :-1])
    row_edge_profile = np.mean(horiz_diff, axis=1)
    horiz_line_periodicity = float(np.std(row_edge_profile)) * 5.0
    lap_var = min(1.0, compute_laplacian_variance(gray * 255.0) / 800.0)

    # 3. Cutaneous & chrominance moments (8 dims)
    skin_ratio = estimate_skin_and_lesion_area_ratio((eval_64 * 255.0).astype(np.uint8))
    mean_r, mean_g, mean_b = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    std_r, std_g, std_b = float(np.std(r)), float(np.std(g)), float(np.std(b))
    rg_diff = float(np.mean(r - g))
    gb_diff = float(np.mean(g - b))

    # 4. Radial center-surround & corner vignetting geometry (10 dims)
    yy, xx = np.ogrid[:64, :64]
    dist = np.sqrt(((yy - 31.5) / 31.5) ** 2 + ((xx - 31.5) / 31.5) ** 2)
    center_mask = dist <= 0.36
    mid_mask = (dist > 0.36) & (dist <= 0.72)
    corner_mask = dist > 0.85

    center_lum = float(np.mean(gray[center_mask]))
    mid_lum = float(np.mean(gray[mid_mask]))
    corner_lum = float(np.mean(gray[corner_mask]))

    corner_vignette = max(0.0, mid_lum - corner_lum)
    dark_corner_frac = float(np.mean(gray[corner_mask] < 0.15))
    center_lesion_dip = mid_lum - center_lum

    center_std = float(np.std(gray[center_mask]))
    mid_std = float(np.std(gray[mid_mask]))
    corner_std = float(np.std(gray[corner_mask]))

    eval_u8 = (eval_64 * 255.0).astype(np.uint8)
    corner_skin = estimate_skin_and_lesion_area_ratio(eval_u8[corner_mask].reshape(-1, 1, 3))
    center_skin = estimate_skin_and_lesion_area_ratio(eval_u8[center_mask].reshape(-1, 1, 3))

    # High-frequency spatial gradient ratio (distinguishes fine TBP tiles vs smooth dermoscopic vs clinical)
    grad_mag = float(np.mean(np.abs(gray[1:, :] - gray[:-1, :])) + np.mean(np.abs(gray[:, 1:] - gray[:, :-1])))

    desc = np.array(
        [
            aspect,
            is_square,
            aspect_dev,
            min_dim_norm,
            white_paper_frac,
            dark_ink_frac,
            bimodal_doc_score,
            low_chroma_frac,
            horiz_line_periodicity,
            lap_var,
            skin_ratio,
            mean_r,
            mean_g,
            mean_b,
            std_r,
            std_g,
            std_b,
            rg_diff,
            gb_diff,
            center_lum,
            mid_lum,
            corner_lum,
            corner_vignette,
            dark_corner_frac,
            center_lesion_dip,
            center_std,
            mid_std,
            corner_std + 0.0 * (corner_skin + center_skin + grad_mag),
        ],
        dtype=np.float64,
    )
    # Replace last 3 slots with corner_skin, center_skin, grad_mag explicitly
    desc[25] = corner_skin
    desc[26] = center_skin
    desc[27] = grad_mag
    return patch_24, desc


class PictureTriageRouter:
    """
    From-scratch 2D ConvNet + 28-dim structural/chrominance descriptor router for 5-class picture triage:
    `skin_clinical`, `skin_dermoscopic`, `skin_tbp_tile`, `document_scan`, `other`.
    """

    architecture_name = "PictureTriageRouterCNN"
    architecture_version = "1.0.0"

    def __init__(
        self,
        conv_channels: int = 8,
        hidden_dim: int = 32,
        desc_dim: int = 28,
        num_classes: int = len(ROUTER_CLASSES),
        seed: int = 42,
    ) -> None:
        self.conv_channels = conv_channels
        self.hidden_dim = hidden_dim
        self.desc_dim = desc_dim
        self.num_classes = num_classes
        self.seed = seed
        self.temperature: float = 1.0

        rng = np.random.default_rng(seed)
        fused_dim = 2 * conv_channels + desc_dim
        self.params: dict[str, np.ndarray] = {
            "W_conv": initialize_tensor(rng, (conv_channels, 3, 3, 3), method="he_normal").astype(np.float64),
            "b_conv": np.zeros((conv_channels,), dtype=np.float64),
            "W_fc1": initialize_tensor(rng, (fused_dim, hidden_dim), method="xavier_uniform").astype(np.float64),
            "b_fc1": np.zeros((hidden_dim,), dtype=np.float64),
            "W_out": initialize_tensor(rng, (hidden_dim, num_classes), method="xavier_uniform").astype(np.float64),
            "b_out": np.zeros((num_classes,), dtype=np.float64),
        }
        self.desc_mean = np.zeros((desc_dim,), dtype=np.float64)
        self.desc_std = np.ones((desc_dim,), dtype=np.float64)
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

    def _conv3x3_forward(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bsz, h, w, _ = patches.shape
        padded = np.pad(patches, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
        out = np.zeros((bsz, h, w, self.conv_channels), dtype=np.float64)
        W = self.params["W_conv"]
        b = self.params["b_conv"]
        for dy in range(3):
            for dx in range(3):
                patch = padded[:, dy : dy + h, dx : dx + w, :]
                out += np.einsum("bhwi,ci->bhwc", patch, W[:, dy, dx, :])
        out += b[None, None, None, :]
        return out, padded

    def _conv3x3_backward(self, d_conv: np.ndarray, padded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, h, w, _ = d_conv.shape
        dW = np.zeros_like(self.params["W_conv"])
        db = np.sum(d_conv, axis=(0, 1, 2))
        for dy in range(3):
            for dx in range(3):
                patch = padded[:, dy : dy + h, dx : dx + w, :]
                dW[:, dy, dx, :] = np.einsum("bhwc,bhwi->ci", d_conv, patch)
        return dW, db

    def forward(self, patches: np.ndarray, descs_norm: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        bsz = patches.shape[0]
        conv_pre, padded = self._conv3x3_forward(patches)
        conv_act = np.tanh(conv_pre)

        h2 = conv_act.shape[1] // 2
        w2 = conv_act.shape[2] // 2
        cropped = conv_act[:, : h2 * 2, : w2 * 2, :]
        pooled = cropped.reshape(bsz, h2, 2, w2, 2, self.conv_channels).mean(axis=(2, 4))

        g_avg = pooled.mean(axis=(1, 2))
        flat_sp = pooled.reshape(bsz, h2 * w2, self.conv_channels)
        max_idx = np.argmax(flat_sp, axis=1)
        g_max = np.take_along_axis(flat_sp, max_idx[:, None, :], axis=1)[:, 0, :]

        fused = np.concatenate([g_avg, g_max, descs_norm], axis=1)
        fc1_pre = fused @ self.params["W_fc1"] + self.params["b_fc1"]
        fc1_act = np.tanh(fc1_pre)
        logits = fc1_act @ self.params["W_out"] + self.params["b_out"]

        ctx = {
            "padded": padded,
            "conv_shape": conv_act.shape,
            "conv_act": conv_act,
            "h2": h2,
            "w2": w2,
            "max_idx": max_idx,
            "fused": fused,
            "fc1_act": fc1_act,
        }
        return logits, ctx

    def backward(self, d_logits: np.ndarray, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
        fc1_act: np.ndarray = ctx["fc1_act"]
        fused: np.ndarray = ctx["fused"]
        conv_act: np.ndarray = ctx["conv_act"]
        bsz, h, w, c = ctx["conv_shape"]
        h2, w2 = ctx["h2"], ctx["w2"]

        dW_out = fc1_act.T @ d_logits
        db_out = np.sum(d_logits, axis=0)
        d_fc1_act = d_logits @ self.params["W_out"].T
        d_fc1_pre = d_fc1_act * (1.0 - fc1_act * fc1_act)

        dW_fc1 = fused.T @ d_fc1_pre
        db_fc1 = np.sum(d_fc1_pre, axis=0)
        d_fused = d_fc1_pre @ self.params["W_fc1"].T

        d_g_avg = d_fused[:, :c]
        d_g_max = d_fused[:, c : 2 * c]

        d_pooled = np.broadcast_to(d_g_avg[:, None, None, :] / float(h2 * w2), (bsz, h2, w2, c)).copy()
        d_flat_max = np.zeros((bsz, h2 * w2, c), dtype=np.float64)
        np.put_along_axis(d_flat_max, ctx["max_idx"][:, None, :], d_g_max[:, None, :], axis=1)
        d_pooled += d_flat_max.reshape(bsz, h2, w2, c)

        d_cropped = np.repeat(np.repeat(d_pooled / 4.0, 2, axis=1), 2, axis=2)
        d_conv_act = np.zeros((bsz, h, w, c), dtype=np.float64)
        d_conv_act[:, : h2 * 2, : w2 * 2, :] = d_cropped
        d_conv_pre = d_conv_act * (1.0 - conv_act * conv_act)

        dW_conv, db_conv = self._conv3x3_backward(d_conv_pre, ctx["padded"])
        return {
            "W_conv": dW_conv,
            "b_conv": db_conv,
            "W_fc1": dW_fc1,
            "b_fc1": db_fc1,
            "W_out": dW_out,
            "b_out": db_out,
        }

    def fit(
        self,
        train_samples: list[tuple[np.ndarray, np.ndarray, int]],
        val_samples: list[tuple[np.ndarray, np.ndarray, int]] | None = None,
        *,
        epochs: int = 35,
        batch_size: int = 24,
        lr: float = 0.015,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        patches = np.stack([s[0] for s in train_samples], axis=0)
        descs = np.stack([s[1] for s in train_samples], axis=0)
        labels = np.asarray([s[2] for s in train_samples], dtype=np.int64)

        self.desc_mean = np.mean(descs, axis=0)
        self.desc_std = np.where(np.std(descs, axis=0) > 1e-6, np.std(descs, axis=0), 1.0)
        descs_norm = (descs - self.desc_mean) / self.desc_std

        # Inverse-frequency class weights for balanced multi-class learning
        class_counts = np.bincount(labels, minlength=self.num_classes).astype(np.float64)
        class_weights = len(labels) / (self.num_classes * np.maximum(1.0, class_counts))

        optimizer = AdamWOptimizer(self.model_params(), lr=lr, weight_decay=1e-4)
        rng = np.random.default_rng(self.seed)
        loss_history: list[float] = []

        for ep in range(epochs):
            # Cosine LR decay
            cos_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * ep / max(1, epochs))))
            optimizer.lr = float(cos_lr)

            order = rng.permutation(len(train_samples))
            ep_loss = 0.0
            batches = 0
            for start in range(0, len(order), batch_size):
                idxs = order[start : start + batch_size]
                logits, ctx = self.forward(patches[idxs], descs_norm[idxs])
                probs = softmax(logits, axis=-1)
                bsz = len(idxs)
                y_b = labels[idxs]
                w_b = class_weights[y_b]
                chosen = np.clip(probs[np.arange(bsz), y_b], 1e-12, 1.0)
                loss = float(-np.mean(w_b * np.log(chosen)))

                d_logits = probs.copy()
                d_logits[np.arange(bsz), y_b] -= 1.0
                d_logits *= (w_b[:, None] / float(max(1, bsz)))

                grads = self.backward(d_logits, ctx)
                optimizer.step(self.params, grads)
                ep_loss += loss
                batches += 1
            loss_history.append(ep_loss / max(1, batches))

        # Calibrate temperature on val if provided
        if val_samples:
            v_patches = np.stack([s[0] for s in val_samples], axis=0)
            v_descs = (np.stack([s[1] for s in val_samples], axis=0) - self.desc_mean) / self.desc_std
            v_labels = np.asarray([s[2] for s in val_samples], dtype=np.int64)
            v_logits, _ = self.forward(v_patches, v_descs)
            best_t = 1.0
            best_nll = float("inf")
            for cand_t in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                p = softmax(v_logits / cand_t, axis=-1)
                nll = float(-np.mean(np.log(np.clip(p[np.arange(len(v_labels)), v_labels], 1e-12, 1.0))))
                if nll < best_nll:
                    best_nll = nll
                    best_t = cand_t
            self.temperature = best_t

        return {
            "initial_loss": round(loss_history[0], 4) if loss_history else None,
            "final_loss": round(loss_history[-1], 4) if loss_history else None,
            "epochs": epochs,
            "temperature": self.temperature,
            "parameter_count": self.parameter_count,
            "initial_weights_sha256": self._initial_manifest.initial_weights_sha256,
            "pretrained_weights_used": False,
            "training_seconds": round(time.perf_counter() - t0, 3),
        }

    def model_params(self) -> dict[str, np.ndarray]:
        return self.params

    def predict_proba_from_features(self, patch_24: np.ndarray, desc_28: np.ndarray) -> np.ndarray:
        desc_norm = (desc_28 - self.desc_mean) / self.desc_std
        logits, _ = self.forward(patch_24[None, ...], desc_norm[None, ...])
        probs = softmax(logits / max(0.1, self.temperature), axis=-1)[0].copy()

        # Structural safety guard: a high-confidence bimodal low-chroma document page with text scanlines
        # is never sent to the skin model.
        white_paper_frac = desc_28[4]
        dark_ink_frac = desc_28[5]
        low_chroma_frac = desc_28[7]
        horiz_periodicity = desc_28[8]
        skin_ratio = desc_28[10]
        if (
            white_paper_frac > 0.50
            and dark_ink_frac > 0.015
            and low_chroma_frac > 0.72
            and horiz_periodicity > 0.03
            and skin_ratio < 0.15
        ):
            doc_idx = ROUTER_CLASSES.index("document_scan")
            probs[:] = 0.01 / (len(ROUTER_CLASSES) - 1)
            probs[doc_idx] = 0.99

        return probs

    def triage_image(
        self,
        img: Image.Image | np.ndarray,
        *,
        quality_thresholds: QualityThresholds | None = None,
    ) -> TriageDecision:
        """Run picture triage and bilingual image-quality assessment on an uploaded image."""
        patch_24, desc_28 = extract_router_features(img)
        probs = self.predict_proba_from_features(patch_24, desc_28)
        pred_idx = int(np.argmax(probs))
        pred_cls = ROUTER_CLASSES[pred_idx]
        conf = float(probs[pred_idx])
        action = ROUTER_ACTIONS[pred_cls]

        is_skin_class = pred_cls in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile")
        quality = assess_image_quality(
            img,
            thresholds=quality_thresholds,
            check_skin_area=is_skin_class,
        )

        abstained = False
        reason_en: str | None = None
        reason_fr: str | None = None

        if pred_cls == "other":
            abstained = True
            reason_en = "Not a supported skin photograph or clinical document scan; skin risk estimation abstained."
            reason_fr = "Image non reconnue comme photographie cutanée ou document clinique ; estimation du risque cutané abstenue."
        elif pred_cls == "document_scan":
            abstained = False
            reason_en = "Detected scanned clinical document page; routed to OCR and text extraction pipeline."
            reason_fr = "Page de document clinique scanné détectée ; orientée vers le pipeline OCR et d'extraction textuelle."
        elif not quality.quality_ok:
            abstained = True
            reason_en = "Image quality check failed: " + " ".join(quality.reasons_en)
            reason_fr = "Échec du contrôle qualité de l'image : " + " ".join(quality.reasons_fr)

        return TriageDecision(
            predicted_class=pred_cls,
            route_action=action,
            confidence=round(conf, 4),
            probabilities={cls_name: round(float(probs[i]), 4) for i, cls_name in enumerate(ROUTER_CLASSES)},
            label_en=ROUTER_LABELS_EN[pred_cls],
            label_fr=ROUTER_LABELS_FR[pred_cls],
            quality=quality,
            abstained=abstained,
            abstain_reason_en=reason_en,
            abstain_reason_fr=reason_fr,
            model_name=f"{self.architecture_name}-v{self.architecture_version}",
            pretrained_weights_used=False,
        )

    def evaluate(self, samples: list[tuple[np.ndarray, np.ndarray, int]]) -> dict[str, Any]:
        y_true = np.asarray([s[2] for s in samples], dtype=np.int64)
        preds: list[int] = []
        for patch_24, desc_28, _ in samples:
            p = self.predict_proba_from_features(patch_24, desc_28)
            preds.append(int(np.argmax(p)))
        y_pred = np.asarray(preds, dtype=np.int64)

        acc = float(np.mean(y_pred == y_true))
        conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, p_idx in zip(y_true, y_pred):
            conf_mat[int(t), int(p_idx)] += 1

        per_class: dict[str, dict[str, Any]] = {}
        f1s: list[float] = []
        for c_idx, cls_name in enumerate(ROUTER_CLASSES):
            tp = int(conf_mat[c_idx, c_idx])
            fp = int(np.sum(conf_mat[:, c_idx]) - tp)
            fn = int(np.sum(conf_mat[c_idx, :]) - tp)
            support = int(np.sum(conf_mat[c_idx, :]))
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = (2 * prec * rec) / max(1e-9, prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
            per_class[cls_name] = {
                "support": support,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }

        doc_idx = ROUTER_CLASSES.index("document_scan")
        skin_indices = [0, 1, 2]
        docs_sent_to_skin = int(sum(conf_mat[doc_idx, s_i] for s_i in skin_indices))

        return {
            "total_samples": len(samples),
            "accuracy": round(acc, 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "document_photos_sent_to_skin_model": docs_sent_to_skin,
            "per_class": per_class,
            "confusion_matrix": {
                "labels": list(ROUTER_CLASSES),
                "matrix": conf_mat.tolist(),
            },
            "pretrained_weights_used": False,
        }

    def save(self, out_dir: str | Path) -> dict[str, Any]:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        npz_path = p / "router_weights.npz"
        arrays_to_save: dict[str, Any] = dict(self.params)
        arrays_to_save["desc_mean"] = self.desc_mean
        arrays_to_save["desc_std"] = self.desc_std
        arrays_to_save["temperature"] = np.asarray([self.temperature], dtype=np.float64)
        np.savez_compressed(npz_path, **arrays_to_save)
        weights_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        manifest = {
            "architecture_name": self.architecture_name,
            "architecture_version": self.architecture_version,
            "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "classes": list(ROUTER_CLASSES),
            "parameter_count": self.parameter_count,
            "seed": self.seed,
            "temperature": self.temperature,
            "initial_weights_sha256": self._initial_manifest.initial_weights_sha256,
            "trained_weights_sha256": weights_sha,
            "pretrained_weights_used": False,
        }
        (p / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, model_dir: str | Path) -> PictureTriageRouter:
        p = Path(model_dir)
        manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        router = cls(seed=int(manifest.get("seed", 42)))
        data = np.load(p / "router_weights.npz")
        for k in router.params:
            router.params[k] = np.asarray(data[k], dtype=np.float64)
        router.desc_mean = np.asarray(data["desc_mean"], dtype=np.float64)
        router.desc_std = np.asarray(data["desc_std"], dtype=np.float64)
        router.temperature = float(data["temperature"][0])
        return router


def generate_document_scan_image(idx: int, *, seed: int = 42) -> Image.Image:
    """Render a realistic synthetic clinical document page / form scan with slight noise and lines."""
    rng = np.random.default_rng(seed + idx * 31)
    w, h = 240, 320
    paper_tone = int(rng.integers(235, 252))
    img = Image.new("RGB", (w, h), color=(paper_tone, paper_tone, max(225, paper_tone - 3)))
    draw = ImageDraw.Draw(img)

    # Header bar & clinical lines
    draw.rectangle([16, 16, w - 16, 36], outline=(40, 40, 45), width=1)
    draw.text((22, 20), f"CLINICAL REPORT #{idx:04d} - HOPITAL / LAB", fill=(20, 20, 25))

    y = 48
    lines = [
        "Patient: ANON-SUBJECT   Date: 2026-11-16",
        "Service: Dermatologie / Medecine Interne",
        "----------------------------------------",
        "Motif: Surveillance lesion pigmentee.",
        "Examen: Macule asymetrique 6mm dos.",
        "Biologie: Hb 13.4 g/dL | CRP 4.2 mg/L",
        "Traitement: Excision chirurgicale prevue.",
        "Conclusion: Controle histopathologique.",
    ]
    for line_idx in range(14):
        txt = lines[line_idx % len(lines)]
        if rng.random() > 0.12:
            draw.text((18, y), txt[: int(rng.integers(22, len(txt) + 1))], fill=(25, 25, 30))
        y += int(rng.integers(14, 19))
        if y > h - 24:
            break

    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0.0, float(rng.uniform(1.5, 5.0)), size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    if rng.random() < 0.35:
        out = out.filter(ImageFilter.GaussianBlur(radius=0.5))
    return out


def generate_non_skin_other_image(idx: int, *, seed: int = 42) -> Image.Image:
    """Generate a non-skin / non-document image (`other`: sky/foliage, geometric objects, blue/green patterns, radiographs)."""
    rng = np.random.default_rng(seed + idx * 53)
    mode = idx % 4
    w, h = 160, 160
    yy, xx = np.ogrid[:h, :w]

    if mode == 0:
        # Cool blue/cyan/green natural landscape / sky-foliage gradient
        r = np.clip(30 + 0.15 * yy + rng.normal(0, 8, (h, w)), 0, 110)
        g = np.clip(120 + 0.4 * yy + rng.normal(0, 12, (h, w)), 60, 240)
        b = np.clip(210 - 0.5 * yy + rng.normal(0, 12, (h, w)), 80, 255)
        arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
        return Image.fromarray(arr)
    if mode == 1:
        # High-saturation RGB geometric shapes on dark/cool background
        img = Image.new("RGB", (w, h), color=(18, 35, 85))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 130, 90], fill=(20, 190, 220), outline=(240, 240, 250), width=3)
        draw.ellipse([50, 60, 145, 145], fill=(40, 210, 95))
        return img
    if mode == 2:
        # Grayscale chest radiograph / metallic grid phantom (dark background, bright ribs/grid)
        base = 25.0 + 95.0 * np.exp(-(((xx - 80.0) / 48.0) ** 2 + ((yy - 80.0) / 55.0) ** 2))
        ribs = 35.0 * (np.sin(yy / 5.5) > 0.4) * (np.abs(xx - 80) < 55)
        gray = np.clip(base + ribs + rng.normal(0, 5, (h, w)), 5, 220).astype(np.uint8)
        return Image.fromarray(np.stack([gray, gray, gray], axis=-1))

    # Checkerboard / vivid purple-teal texture
    checker = ((xx // 16 + yy // 16) % 2).astype(np.float32)
    r = (40 + 60 * checker + rng.normal(0, 6, (h, w))).clip(0, 255)
    g = (170 * (1.0 - checker) + 30 + rng.normal(0, 6, (h, w))).clip(0, 255)
    b = (210 * checker + 90 * (1.0 - checker) + rng.normal(0, 6, (h, w))).clip(0, 255)
    return Image.fromarray(np.stack([r, g, b], axis=-1).astype(np.uint8))


def generate_synthetic_skin_modality_image(modality: str, idx: int, *, seed: int = 42) -> Image.Image:
    """Fallback synthetic skin image generator when offline in CI."""
    rng = np.random.default_rng(seed + idx * 71)
    tone_shift = (idx % 6) * 22
    base_r = max(75, 225 - tone_shift)
    base_g = max(52, 182 - int(tone_shift * 0.9))
    base_b = max(38, 155 - int(tone_shift * 0.85))

    if modality == "skin_clinical":
        w, h = 200, 150  # 4:3 camera aspect ratio
    elif modality == "skin_dermoscopic":
        w, h = 160, 140
    else:
        w, h = 128, 128  # Square TBP tile

    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt(((yy - cy) / (h / 2.0)) ** 2 + ((xx - cx) / (w / 2.0)) ** 2)

    r = np.full((h, w), base_r, dtype=np.float32) + rng.normal(0, 6, (h, w))
    g = np.full((h, w), base_g, dtype=np.float32) + rng.normal(0, 5, (h, w))
    b = np.full((h, w), base_b, dtype=np.float32) + rng.normal(0, 5, (h, w))

    if modality == "skin_dermoscopic":
        # Central lesion + circular dermatoscope dark corner vignette
        lesion = dist < 0.42
        r[lesion] *= 0.52
        g[lesion] *= 0.42
        b[lesion] *= 0.38
        vignette = np.clip(1.0 - np.maximum(0.0, dist - 0.82) * 2.8, 0.08, 1.0)
        r *= vignette
        g *= vignette
        b *= vignette
    elif modality == "skin_tbp_tile":
        # Uniform square skin tile with small central macule and no vignette
        lesion = dist < 0.22
        r[lesion] *= 0.68
        g[lesion] *= 0.58
        b[lesion] *= 0.52
    else:
        # Clinical photo: non-square with mild directional lighting gradient and non-skin border strip
        grad = 1.0 + 0.15 * ((xx - cx) / w)
        r *= grad
        g *= grad
        b *= grad
        lesion = dist < 0.28
        r[lesion] *= 0.58
        g[lesion] *= 0.46
        b[lesion] *= 0.42
        # Clinical clothing/background edge on top corners
        bg_edge = (yy < h * 0.18) & (xx < w * 0.35)
        r[bg_edge], g[bg_edge], b[bg_edge] = 45.0, 65.0, 95.0

    arr = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_router_dataset(
    meta_dir: str | Path = "var/cache/derm/meta",
    cache_dirs: list[str | Path] | None = None,
    *,
    per_class_target: int = 90,
    seed: int = 42,
) -> dict[str, list[tuple[np.ndarray, np.ndarray, int]]]:
    """
    Build `train`, `val`, and `locked_test` feature datasets for the 5-class picture triage router,
    combining real cached `derm-corpus` thumbnails across modalities with rendered clinical document
    scans and non-skin images.
    """
    dirs = [Path(d) for d in (cache_dirs or ["var/cache/derm/thumbs", "var/cache/derm_thumbs"])]
    existing_map: dict[str, Path] = {}
    for base in dirs:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".webp", ".jpg", ".jpeg", ".png"):
                existing_map.setdefault(str(p.relative_to(base)).replace("\\", "/"), p)

    mod_to_router_cls = {
        "clinical": "skin_clinical",
        "dermoscopic": "skin_dermoscopic",
        "tbp_tile": "skin_tbp_tile",
    }
    by_mod_records: dict[str, list[tuple[Path, str]]] = {
        "skin_clinical": [],
        "skin_dermoscopic": [],
        "skin_tbp_tile": [],
    }

    # Scan cached thumbnails directly from disk and infer modality from first page manifests
    from medresearch_platform.vision.catalog import _deterministic_split, _open_json_auto

    meta_base = Path(meta_dir)
    for mod_name, r_cls in mod_to_router_cls.items():
        first_page = meta_base / f"manifest-{mod_name}-000.json"
        if not first_page.exists():
            continue
        payload = _open_json_auto(first_page)
        items = payload.get("items", []) if isinstance(payload, dict) else payload
        for row in items:
            if len(by_mod_records[r_cls]) >= per_class_target:
                break
            t_rel = row.get("t")
            if not t_rel:
                continue
            cand = existing_map.get(str(t_rel).lstrip("/"))
            if cand is not None:
                img_id = str(row.get("i") or "")
                ds = str(row.get("d") or "isic")
                sp = _deterministic_split(f"{ds}:{img_id.split(':')[-1]}")
                by_mod_records[r_cls].append((cand, sp))

    splits: dict[str, list[tuple[np.ndarray, np.ndarray, int]]] = {
        "train": [],
        "val": [],
        "locked_test": [],
    }

    def _split_for_idx(i: int) -> str:
        mod10 = i % 10
        if mod10 < 7:
            return "train"
        if mod10 < 8 or (mod10 == 8 and i % 20 == 8):
            return "val"
        return "locked_test"

    # 1. Populate the 3 skin modality classes (real cached thumbnails + fallback if needed)
    for r_cls in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile"):
        cls_idx = ROUTER_CLASSES.index(r_cls)
        real_items = by_mod_records[r_cls]
        for path, sp in real_items:
            try:
                with Image.open(path) as im:
                    p24, d28 = extract_router_features(im.convert("RGB"))
                    splits[sp].append((p24, d28, cls_idx))
            except Exception:
                continue

        missing = max(0, per_class_target - len(real_items))
        for m_idx in range(missing):
            syn_im = generate_synthetic_skin_modality_image(r_cls, m_idx, seed=seed)
            p24, d28 = extract_router_features(syn_im)
            splits[_split_for_idx(m_idx)].append((p24, d28, cls_idx))

    # 2. Populate `document_scan` class
    doc_cls_idx = ROUTER_CLASSES.index("document_scan")
    for idx in range(per_class_target):
        doc_im = generate_document_scan_image(idx, seed=seed)
        p24, d28 = extract_router_features(doc_im)
        splits[_split_for_idx(idx)].append((p24, d28, doc_cls_idx))

    # 3. Populate `other` class
    other_cls_idx = ROUTER_CLASSES.index("other")
    for idx in range(per_class_target):
        oth_im = generate_non_skin_other_image(idx, seed=seed)
        p24, d28 = extract_router_features(oth_im)
        splits[_split_for_idx(idx)].append((p24, d28, other_cls_idx))

    return splits


def train_and_evaluate_router(
    meta_dir: str | Path = "var/cache/derm/meta",
    out_model_dir: str | Path = "models/vision_router_v1",
    out_eval_json: str | Path = "docs/results/vision/step2_router_eval.json",
    *,
    per_class_target: int = 90,
    epochs: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the 5-class PictureTriageRouter from scratch, calibrate quality thresholds on val, and evaluate."""
    t0 = time.perf_counter()
    splits = build_router_dataset(meta_dir, per_class_target=per_class_target, seed=seed)
    router = PictureTriageRouter(seed=seed)
    train_info = router.fit(splits["train"], splits["val"], epochs=epochs)
    val_eval = router.evaluate(splits["val"])
    test_eval = router.evaluate(splits["locked_test"])
    manifest = router.save(out_model_dir)
    from medresearch_platform.vision.splits import log_locked_test_access

    log_locked_test_access(
        dataset_name="derm-router-5class:locked_test",
        model_name=f"{router.architecture_name}-v{router.architecture_version}",
        reason="Step 2 5-class picture triage router locked-test evaluation",
        sample_count=int(test_eval["total_samples"]),
    )

    # Calibrate image quality thresholds on full-resolution clean vs degraded validation skin images
    val_skin_images = [
        generate_synthetic_skin_modality_image(ROUTER_CLASSES[i % 3], i, seed=seed + 100)
        for i in range(30)
    ]
    quality_cal = calibrate_quality_thresholds_on_val(val_skin_images, seed=seed)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_manifest": manifest,
        "training": train_info,
        "val_evaluation": val_eval,
        "locked_test_evaluation": test_eval,
        "quality_calibration_val": quality_cal,
        "total_elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
    eval_path = Path(out_eval_json)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
