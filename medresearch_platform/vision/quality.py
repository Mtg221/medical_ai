"""Bilingual (EN/FR) image-quality gate: blur (Laplacian variance), exposure, resolution, and inclusive I-VI skin area."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityThresholds:
    min_resolution_px: int = 48
    min_laplacian_variance: float = 16.0
    min_brightness: float = 0.12
    max_brightness: float = 0.92
    max_dark_fraction: float = 0.78
    max_bright_fraction: float = 0.72
    min_skin_area_ratio: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageQualityReport:
    quality_ok: bool
    quality_score: float
    width: int
    height: int
    laplacian_variance: float
    mean_brightness: float
    contrast_std: float
    underexposed_fraction: float
    overexposed_fraction: float
    skin_area_ratio: float
    reasons_code: list[str]
    reasons_en: list[str]
    reasons_fr: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_rgb_uint8(img: Image.Image | np.ndarray) -> tuple[np.ndarray, int, int]:
    if isinstance(img, Image.Image):
        rgb = img.convert("RGB")
        w, h = rgb.size
        arr = np.asarray(rgb, dtype=np.uint8)
        return arr, w, h
    arr_in = np.asarray(img)
    if arr_in.ndim == 2:
        arr_in = np.stack([arr_in, arr_in, arr_in], axis=-1)
    if arr_in.dtype in (np.float32, np.float64):
        scale = 255.0 if float(np.max(arr_in)) <= 1.05 else 1.0
        arr_u8 = np.clip(arr_in * scale, 0.0, 255.0).astype(np.uint8)
    else:
        arr_u8 = np.clip(arr_in, 0, 255).astype(np.uint8)
    h, w = arr_u8.shape[:2]
    return arr_u8[..., :3], w, h


def compute_laplacian_variance(gray_255: np.ndarray) -> float:
    """Compute variance of the 2D discrete Laplacian filter `[[0,1,0],[1,-4,1],[0,1,0]]`."""
    g = gray_255.astype(np.float64)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    center = g[1:-1, 1:-1]
    lap = (
        g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
        - 4.0 * center
    )
    return float(np.var(lap))


def estimate_skin_and_lesion_area_ratio(rgb_u8: np.ndarray) -> float:
    """
    Estimate fraction of pixels matching cutaneous tissue or pigmented lesion chrominance
    across Fitzpatrick skin tones I through VI (using YCbCr + RGB warm-channel ordering).
    """
    r = rgb_u8[..., 0].astype(np.float32)
    g = rgb_u8[..., 1].astype(np.float32)
    b = rgb_u8[..., 2].astype(np.float32)

    # YCbCr conversion (ITU-R BT.601)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    # Inclusive YCbCr bounds covering Fitzpatrick I-VI skin + pigmented lesions (brown/red/tan/dark)
    ycbcr_skin = (
        (y >= 18.0)
        & (y <= 245.0)
        & (cb >= 72.0)
        & (cb <= 142.0)
        & (cr >= 126.0)
        & (cr <= 185.0)
    )
    # Warm cutaneous / melanin dominance (R >= G >= B with non-zero chrominance, or dark central lesion)
    warm_skin = (
        (r >= g - 4.0)
        & (g >= b - 8.0)
        & ((r - b) >= 6.0)
        & (y >= 15.0)
        & (y <= 242.0)
    )
    combined = ycbcr_skin | warm_skin
    return float(np.mean(combined))


def assess_image_quality(
    img: Image.Image | np.ndarray,
    *,
    thresholds: QualityThresholds | None = None,
    check_skin_area: bool = True,
) -> ImageQualityReport:
    """
    Evaluate image quality (resolution, Laplacian focus, brightness/exposure, and skin area)
    and return bilingual EN/FR rejection reasons when any threshold fails.
    """
    thr = thresholds or QualityThresholds()
    rgb_u8, w, h = _to_rgb_uint8(img)

    # Standardize to <= 256x256 for consistent Laplacian variance across camera resolutions
    pil_rgb = Image.fromarray(rgb_u8)
    eval_w = min(256, max(32, w))
    eval_h = min(256, max(32, h))
    if (eval_w, eval_h) != (w, h):
        pil_eval = pil_rgb.resize((eval_w, eval_h), Image.Resampling.BILINEAR)
        rgb_eval = np.asarray(pil_eval, dtype=np.uint8)
    else:
        rgb_eval = rgb_u8

    gray_255 = (
        0.299 * rgb_eval[..., 0].astype(np.float64)
        + 0.587 * rgb_eval[..., 1].astype(np.float64)
        + 0.114 * rgb_eval[..., 2].astype(np.float64)
    )
    gray_01 = gray_255 / 255.0

    lap_var = compute_laplacian_variance(gray_255)
    mean_b = float(np.mean(gray_01))
    std_b = float(np.std(gray_01))
    under_frac = float(np.mean(gray_01 < 0.08))
    over_frac = float(np.mean(gray_01 > 0.96))
    skin_ratio = estimate_skin_and_lesion_area_ratio(rgb_eval)

    reasons_code: list[str] = []
    reasons_en: list[str] = []
    reasons_fr: list[str] = []

    if min(w, h) < thr.min_resolution_px:
        reasons_code.append("LOW_RESOLUTION")
        reasons_en.append(
            f"Image resolution is too low ({w}x{h} px; minimum {thr.min_resolution_px}x{thr.min_resolution_px} px required)."
        )
        reasons_fr.append(
            f"La résolution de l'image est trop faible ({w}x{h} px ; minimum {thr.min_resolution_px}x{thr.min_resolution_px} px requis)."
        )

    if mean_b < thr.min_brightness or under_frac > thr.max_dark_fraction:
        reasons_code.append("TOO_DARK")
        reasons_en.append(
            f"Image is underexposed / too dark (mean brightness {mean_b:.2f}, dark pixels {under_frac:.0%})."
        )
        reasons_fr.append(
            f"L'image est sous-exposée / trop sombre (luminosité moyenne {mean_b:.2f}, pixels sombres {under_frac:.0%})."
        )

    if mean_b > thr.max_brightness or over_frac > thr.max_bright_fraction:
        reasons_code.append("OVEREXPOSED")
        reasons_en.append(
            f"Image is overexposed / washed out (mean brightness {mean_b:.2f}, saturated pixels {over_frac:.0%})."
        )
        reasons_fr.append(
            f"L'image est surexposée / saturée (luminosité moyenne {mean_b:.2f}, pixels saturés {over_frac:.0%})."
        )

    if lap_var < thr.min_laplacian_variance:
        reasons_code.append("BLURRY")
        reasons_en.append(
            f"Image is out of focus / blurry (Laplacian variance {lap_var:.1f} < {thr.min_laplacian_variance:.1f})."
        )
        reasons_fr.append(
            f"L'image est floue / hors foyer (variance laplacienne {lap_var:.1f} < {thr.min_laplacian_variance:.1f})."
        )

    if check_skin_area and skin_ratio < thr.min_skin_area_ratio:
        reasons_code.append("INSUFFICIENT_SKIN_AREA")
        reasons_en.append(
            f"Insufficient visible skin or lesion area ({skin_ratio:.0%} < {thr.min_skin_area_ratio:.0%})."
        )
        reasons_fr.append(
            f"Surface cutanée ou lésionnelle visible insuffisante ({skin_ratio:.0%} < {thr.min_skin_area_ratio:.0%})."
        )

    focus_score = min(1.0, lap_var / max(1.0, thr.min_laplacian_variance * 3.0))
    exposure_score = max(0.0, 1.0 - abs(mean_b - 0.5) * 1.8)
    area_score = min(1.0, skin_ratio / max(0.05, thr.min_skin_area_ratio * 2.0)) if check_skin_area else 1.0
    raw_q = 0.45 * focus_score + 0.30 * exposure_score + 0.25 * area_score
    if reasons_code:
        raw_q = min(raw_q, 0.42)

    return ImageQualityReport(
        quality_ok=len(reasons_code) == 0,
        quality_score=round(float(np.clip(raw_q, 0.0, 1.0)), 4),
        width=w,
        height=h,
        laplacian_variance=round(lap_var, 3),
        mean_brightness=round(mean_b, 4),
        contrast_std=round(std_b, 4),
        underexposed_fraction=round(under_frac, 4),
        overexposed_fraction=round(over_frac, 4),
        skin_area_ratio=round(skin_ratio, 4),
        reasons_code=reasons_code,
        reasons_en=reasons_en,
        reasons_fr=reasons_fr,
    )


def calibrate_quality_thresholds_on_val(
    clean_val_images: Sequence[Image.Image | np.ndarray],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Calibrate quality thresholds on `val` images paired with degraded perturbations
    (severe Gaussian blur, underexposure, overexposure, low resolution) to maximize F1.
    """
    from PIL import ImageFilter

    rng = np.random.default_rng(seed)
    pos_imgs: list[Image.Image] = []
    neg_imgs: list[tuple[Image.Image, str]] = []

    for raw in clean_val_images:
        arr_u8, _, _ = _to_rgb_uint8(raw)
        pil_clean = Image.fromarray(arr_u8)
        if min(pil_clean.size) < 64:
            pil_clean = pil_clean.resize((128, 128), Image.Resampling.BICUBIC)
        pos_imgs.append(pil_clean)

        mode = int(rng.integers(0, 4))
        if mode == 0:
            # Blurry
            neg_imgs.append((pil_clean.filter(ImageFilter.GaussianBlur(radius=6.0)), "BLURRY"))
        elif mode == 1:
            # Too dark
            dark = np.clip(np.asarray(pil_clean, dtype=np.float32) * 0.12, 0, 255).astype(np.uint8)
            neg_imgs.append((Image.fromarray(dark), "TOO_DARK"))
        elif mode == 2:
            # Overexposed
            bright = np.clip(np.asarray(pil_clean, dtype=np.float32) * 1.9 + 185.0, 0, 255).astype(np.uint8)
            neg_imgs.append((Image.fromarray(bright), "OVEREXPOSED"))
        else:
            # Low resolution
            tiny = pil_clean.resize((24, 24), Image.Resampling.BILINEAR)
            neg_imgs.append((tiny, "LOW_RESOLUTION"))

    best_f1 = -1.0
    best_thr = QualityThresholds()
    for lap_cand in (10.0, 14.0, 16.0, 20.0, 25.0):
        cand = QualityThresholds(min_laplacian_variance=lap_cand)
        tp = sum(1 for im in pos_imgs if assess_image_quality(im, thresholds=cand).quality_ok)
        fp = sum(1 for im, _ in neg_imgs if assess_image_quality(im, thresholds=cand).quality_ok)
        fn = len(pos_imgs) - tp
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-9, prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = cand

    return {
        "calibrated_thresholds": best_thr.to_dict(),
        "val_clean_count": len(pos_imgs),
        "val_degraded_count": len(neg_imgs),
        "val_quality_f1": round(best_f1, 4),
    }
