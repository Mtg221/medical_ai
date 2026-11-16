"""Step 3.1(a): Classical Dermatology Vision Baseline (RGB/HSV/Lab colour, LBP & GLCM/Haralick texture, Otsu ABCD lesion shape + LogReg/SVM/GBDT)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.quality import _to_rgb_uint8
from medresearch_platform.vision.vision_classifier import encode_derm_metadata

CORE_DX_CLASSES: tuple[str, ...] = (
    "melanoma",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "actinic_keratosis",
    "nevus",
    "seborrheic_keratosis",
    "dermatofibroma",
    "vascular_lesion",
    "other",
)


def map_record_to_core_dx_index(rec: DermCatalogRecord) -> int:
    dx = rec.diagnosis_harmonised
    if dx in CORE_DX_CLASSES:
        return CORE_DX_CLASSES.index(dx)
    if dx == "malignant_unspecified" or rec.malignancy_group == "malignant":
        return CORE_DX_CLASSES.index("melanoma")
    if dx == "benign_unspecified" or rec.malignancy_group == "benign":
        return CORE_DX_CLASSES.index("nevus")
    return CORE_DX_CLASSES.index("other")


def _rgb_to_hsv_and_lab(rgb_01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized `[H, W, 3]` RGB in `[0, 1]` to HSV and approximate CIE-Lab."""
    r = rgb_01[..., 0]
    g = rgb_01[..., 1]
    b = rgb_01[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 1e-6)
    mask_g = (cmax == g) & (delta > 1e-6)
    mask_b = (cmax == b) & (delta > 1e-6)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0
    h = (h / 6.0) % 1.0
    s = np.where(cmax > 1e-6, delta / np.maximum(1e-6, cmax), 0.0)
    v = cmax
    hsv = np.stack([h, s, v], axis=-1)

    # Approximate sRGB -> XYZ -> CIE-Lab
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b

    def _f(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, np.cbrt(np.maximum(0.0, t)), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = _f(x / 0.95047), _f(y), _f(z / 1.08883)
    L = (116.0 * fy - 16.0) / 100.0
    a = (500.0 * (fx - fy) + 128.0) / 255.0
    b_lab = (200.0 * (fy - fz) + 128.0) / 255.0
    lab = np.clip(np.stack([L, a, b_lab], axis=-1), 0.0, 1.0)
    return hsv, lab


def _otsu_lesion_mask(gray_01: np.ndarray) -> np.ndarray:
    """Compute Otsu threshold mask for darker central cutaneous lesion (`True` = lesion)."""
    vals = np.clip((gray_01 * 255.0).astype(np.int32), 0, 255)
    hist = np.bincount(vals.ravel(), minlength=256).astype(np.float64)
    total = float(vals.size)
    if total == 0:
        return np.zeros_like(gray_01, dtype=bool)

    sum_total = float(np.dot(np.arange(256), hist))
    weight_bg = 0.0
    sum_bg = 0.0
    max_var = -1.0
    thresh = 128

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += float(t) * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t

    mask = vals <= thresh
    # Suppress outer 10% border frame so dermatoscope corners don't dominate lesion mask
    h, w = gray_01.shape
    yy, xx = np.ogrid[:h, :w]
    central_roi = (np.abs(yy - h / 2.0) <= h * 0.42) & (np.abs(xx - w / 2.0) <= w * 0.42)
    mask = mask & central_roi
    if int(np.sum(mask)) < 4:
        mask = (np.abs(yy - h / 2.0) <= h * 0.22) & (np.abs(xx - w / 2.0) <= w * 0.22)
    return mask


def _extract_lbp_and_glcm_features(gray_01: np.ndarray) -> np.ndarray:
    """Compute 8-neighbor Local Binary Pattern histogram (10 dims) and 8-level Haralick/GLCM descriptors (8 dims)."""
    g = gray_01
    c = g[1:-1, 1:-1]
    neighbors = [
        g[:-2, :-2] >= c,
        g[:-2, 1:-1] >= c,
        g[:-2, 2:] >= c,
        g[1:-1, 2:] >= c,
        g[2:, 2:] >= c,
        g[2:, 1:-1] >= c,
        g[2:, :-2] >= c,
        g[1:-1, :-2] >= c,
    ]
    # Count bit transitions (uniform LBP pattern proxy: 0..8 active neighbors + uniformity)
    active_count = sum(n.astype(np.int32) for n in neighbors)
    transitions = sum((neighbors[i] != neighbors[(i + 1) % 8]).astype(np.int32) for i in range(8))
    bins = np.where(transitions <= 2, active_count, 9)
    lbp_hist = np.bincount(bins.ravel(), minlength=10).astype(np.float64)
    lbp_hist /= max(1.0, float(np.sum(lbp_hist)))

    # 8-level GLCM along horizontal (dx=1, dy=0) and vertical (dx=0, dy=1) offsets
    q = np.clip((gray_01 * 7.999).astype(np.int32), 0, 7)
    pairs_h = q[:, :-1] * 8 + q[:, 1:]
    pairs_v = q[:-1, :] * 8 + q[1:, :]
    glcm = (
        np.bincount(pairs_h.ravel(), minlength=64).reshape(8, 8).astype(np.float64)
        + np.bincount(pairs_v.ravel(), minlength=64).reshape(8, 8).astype(np.float64)
    )
    glcm = glcm + glcm.T
    glcm /= max(1.0, float(np.sum(glcm)))

    ii, jj = np.meshgrid(np.arange(8, dtype=np.float64), np.arange(8, dtype=np.float64), indexing="ij")
    contrast = float(np.sum(glcm * (ii - jj) ** 2)) / 49.0
    dissimilarity = float(np.sum(glcm * np.abs(ii - jj))) / 7.0
    homogeneity = float(np.sum(glcm / (1.0 + (ii - jj) ** 2)))
    asm = float(np.sum(glcm ** 2))
    energy = float(np.sqrt(asm))
    entropy = float(-np.sum(glcm * np.log(np.clip(glcm, 1e-12, 1.0)))) / np.log(64.0)

    mu_i = float(np.sum(ii * glcm))
    mu_j = float(np.sum(jj * glcm))
    std_i = float(np.sqrt(np.sum(glcm * (ii - mu_i) ** 2) + 1e-9))
    std_j = float(np.sqrt(np.sum(glcm * (jj - mu_j) ** 2) + 1e-9))
    corr = float(np.sum(glcm * (ii - mu_i) * (jj - mu_j)) / (std_i * std_j))
    max_prob = float(np.max(glcm))

    haralick = np.array([contrast, dissimilarity, homogeneity, asm, energy, entropy, corr, max_prob], dtype=np.float64)
    return np.concatenate([lbp_hist, haralick], axis=0)


def extract_classical_lesion_features(
    img: Image.Image | np.ndarray,
    *,
    size: int = 64,
) -> np.ndarray:
    """
    Extract 64-dim classical computer vision feature vector:
    - Colour histograms & moments in RGB, HSV, and CIE-Lab (`36` dims)
    - Texture: 8-neighbor LBP + 8-level Haralick/GLCM (`18` dims)
    - Shape / Border / Asymmetry / Color variegation on Otsu lesion mask (`10` dims)
    """
    rgb_u8, _, _ = _to_rgb_uint8(img)
    pil_rgb = Image.fromarray(rgb_u8).resize((size, size), Image.Resampling.BILINEAR)
    rgb = np.asarray(pil_rgb, dtype=np.float64) / 255.0
    hsv, lab = _rgb_to_hsv_and_lab(rgb)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    # 1. Colour histograms (4 bins per channel across RGB, HSV, Lab = 36 dims)
    color_feats: list[float] = []
    for space in (rgb, hsv, lab):
        for ch in range(3):
            hist, _ = np.histogram(space[..., ch], bins=4, range=(0.0, 1.0))
            hist_norm = hist.astype(np.float64) / float(max(1, size * size))
            color_feats.extend(hist_norm.tolist())

    # 2. LBP + Haralick/GLCM texture (18 dims)
    tex_feats = _extract_lbp_and_glcm_features(gray)

    # 3. Otsu lesion segmentation + ABCD border/asymmetry/color variegation features (10 dims)
    mask = _otsu_lesion_mask(gray)
    area_ratio = float(np.mean(mask))
    inv_mask = ~mask

    # Perimeter via 4-neighbor boundaryXOR
    pad_m = np.pad(mask, 1, mode="constant", constant_values=False)
    boundary = mask & (
        (~pad_m[:-2, 1:-1])
        | (~pad_m[2:, 1:-1])
        | (~pad_m[1:-1, :-2])
        | (~pad_m[1:-1, 2:])
    )
    perim = float(np.sum(boundary))
    area_px = max(1.0, float(np.sum(mask)))
    compactness = min(5.0, (perim ** 2) / (4.0 * np.pi * area_px)) / 5.0

    # Horizontal & vertical reflection asymmetry around image center
    asym_h = float(np.mean(mask != mask[:, ::-1]))
    asym_v = float(np.mean(mask != mask[::-1, :]))

    # Lesion vs background colour contrast & variegation
    lesion_rgb_mean = np.mean(rgb[mask], axis=0) if np.any(mask) else np.mean(rgb, axis=(0, 1))
    bg_rgb_mean = np.mean(rgb[inv_mask], axis=0) if np.any(inv_mask) else np.mean(rgb, axis=(0, 1))
    lesion_rgb_std = np.std(rgb[mask], axis=0) if np.any(mask) else np.std(rgb, axis=(0, 1))

    contrast_r = float(bg_rgb_mean[0] - lesion_rgb_mean[0])
    contrast_g = float(bg_rgb_mean[1] - lesion_rgb_mean[1])
    contrast_b = float(bg_rgb_mean[2] - lesion_rgb_mean[2])
    variegation = float(np.mean(lesion_rgb_std))
    lesion_lab_a = float(np.mean(lab[..., 1][mask])) if np.any(mask) else 0.5
    lesion_lab_b = float(np.mean(lab[..., 2][mask])) if np.any(mask) else 0.5

    shape_feats = np.array(
        [
            area_ratio,
            compactness,
            asym_h,
            asym_v,
            contrast_r,
            contrast_g,
            contrast_b,
            variegation,
            lesion_lab_a,
            lesion_lab_b,
        ],
        dtype=np.float64,
    )

    return np.concatenate([np.asarray(color_feats, dtype=np.float64), tex_feats, shape_feats], axis=0)


@dataclass
class ClassicalSkinClassifier:
    """
    Classical dermatology baseline classifier supporting Logistic Regression (`logreg`),
    calibrated Linear SVM (`svm`), and Histogram Gradient Boosting (`gbdt`).
    """

    variant: str = "logreg"
    include_metadata: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        self.scaler = StandardScaler()
        self.mal_clf: Any = None
        self.dx_clf: Any = None
        self.dx_classes_: list[int] = []

    def _build_estimator(self, *, is_multiclass: bool = False) -> Any:
        if self.variant == "svm":
            base = LinearSVC(C=0.5, class_weight="balanced", random_state=self.seed, max_iter=2000)
            return CalibratedClassifierCV(base, cv=2)
        if self.variant == "gbdt":
            return HistGradientBoostingClassifier(
                max_iter=60,
                max_depth=4,
                learning_rate=0.08,
                random_state=self.seed,
            )
        return LogisticRegression(
            C=0.8,
            class_weight="balanced",
            max_iter=500,
            random_state=self.seed,
        )

    def _featurize_batch(
        self, samples: list[tuple[np.ndarray, int, DermCatalogRecord]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats: list[np.ndarray] = []
        y_mal: list[int] = []
        y_dx: list[int] = []
        for img_arr, mal_label, rec in samples:
            f_vis = extract_classical_lesion_features(img_arr)
            if self.include_metadata:
                f_meta = encode_derm_metadata(rec, include_skin_tone=False)
                f_vis = np.concatenate([f_vis, f_meta], axis=0)
            feats.append(f_vis)
            y_mal.append(int(mal_label))
            y_dx.append(map_record_to_core_dx_index(rec))
        return np.stack(feats, axis=0), np.asarray(y_mal, dtype=np.int64), np.asarray(y_dx, dtype=np.int64)

    def fit(self, samples: list[tuple[np.ndarray, int, DermCatalogRecord]]) -> dict[str, Any]:
        t0 = time.perf_counter()
        X, y_mal, y_dx = self._featurize_batch(samples)
        X_scaled = self.scaler.fit_transform(X)

        self.mal_clf = self._build_estimator(is_multiclass=False)
        self.mal_clf.fit(X_scaled, y_mal)

        unique_dx = sorted(set(y_dx.tolist()))
        self.dx_classes_ = unique_dx
        if len(unique_dx) >= 2:
            self.dx_clf = LogisticRegression(
                C=0.8,
                class_weight="balanced",
                max_iter=500,
                random_state=self.seed,
            )
            self.dx_clf.fit(X_scaled, y_dx)
        else:
            self.dx_clf = None

        return {
            "variant": self.variant,
            "feature_dim": int(X.shape[1]),
            "train_samples": len(samples),
            "training_seconds": round(time.perf_counter() - t0, 3),
            "pretrained_weights_used": False,
        }

    def predict_proba(
        self, samples: list[tuple[np.ndarray, int, DermCatalogRecord]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(p_malignant [N], p_diagnosis [N, len(CORE_DX_CLASSES)])`."""
        X, _, _ = self._featurize_batch(samples)
        X_scaled = self.scaler.transform(X)
        mal_probs = self.mal_clf.predict_proba(X_scaled)[:, 1]

        dx_probs = np.full((len(samples), len(CORE_DX_CLASSES)), 1e-4, dtype=np.float64)
        if self.dx_clf is not None:
            raw_dx_p = self.dx_clf.predict_proba(X_scaled)
            for col_idx, cls_id in enumerate(self.dx_clf.classes_):
                dx_probs[:, int(cls_id)] = raw_dx_p[:, col_idx]
        else:
            dx_probs[:, 0] = mal_probs
            dx_probs[:, 4] = 1.0 - mal_probs
        dx_probs /= np.sum(dx_probs, axis=1, keepdims=True)
        return mal_probs, dx_probs
