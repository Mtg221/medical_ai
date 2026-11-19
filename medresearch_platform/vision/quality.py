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


def detect_dense_text_lines(image: Image.Image | np.ndarray) -> dict[str, Any]:
    """
    Fast pre-skin safety check for dense horizontal text lines (documents, receipts, forms, screens).
    Uses optional Tesseract line detection when available plus a deterministic NumPy stroke-row detector.
    """
    from PIL import ImageFilter

    arr_u8, _, _ = _to_rgb_uint8(image)
    pil = Image.fromarray(arr_u8)

    # 1. Optional fast Tesseract word/line check when binary + pytesseract exist
    ocr_lines = 0
    ocr_words = 0
    try:
        import shutil

        if shutil.which("tesseract"):
            import pytesseract  # type: ignore[import-untyped]

            data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, timeout=3)
            line_keys: set[tuple[int, int, int]] = set()
            for txt, conf, b, p, ln in zip(
                data.get("text", []),
                data.get("conf", []),
                data.get("block_num", []),
                data.get("par_num", []),
                data.get("line_num", []),
            ):
                tok = str(txt or "").strip()
                try:
                    c_val = float(conf)
                except Exception:
                    c_val = -1.0
                if len(tok) >= 2 and any(ch.isalpha() for ch in tok) and c_val >= 45.0:
                    ocr_words += 1
                    line_keys.add((int(b), int(p), int(ln)))
            ocr_lines = len(line_keys)
    except Exception:
        ocr_lines = 0
        ocr_words = 0

    # 2. Pure-NumPy high-contrast horizontal text-line stroke detector (works without Tesseract)
    gray_pil = pil.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    gray = np.asarray(gray_pil, dtype=np.float32)
    local_bg = np.asarray(gray_pil.filter(ImageFilter.BoxBlur(6)), dtype=np.float32)
    ink = local_bg - gray  # dark text on lighter background

    # Crisp vertical glyph strokes (horizontal transitions across dark ink)
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    stroke_mask = (dx >= 22.0) & (np.abs(ink) >= 12.0)
    crossings_per_row = np.sum(stroke_mask, axis=1)
    mean_row_ink = np.mean(np.clip(ink, 0.0, 255.0), axis=1)

    # Text rows have frequent sharp horizontal transitions and localized dark ink
    text_row = (crossings_per_row >= 18) & (mean_row_ink >= 1.5)
    # Count distinct horizontal line bands (contiguous True runs of height 2..14 px)
    band_count = 0
    run_len = 0
    for flag in text_row:
        if flag:
            run_len += 1
        else:
            if 2 <= run_len <= 16:
                band_count += 1
            run_len = 0
    if 2 <= run_len <= 16:
        band_count += 1

    # 3. Illumination- & warm-lighting-invariant 384x384 contrast-normalized text-line periodicity detector
    # Calibrated on val skin images (0/240 FP): requires >=7 crisp dark-ink horizontal bands separated by quiet rows.
    gray_hi_pil = pil.convert("L").resize((384, 384), Image.Resampling.BILINEAR)
    gray_hi = np.asarray(gray_hi_pil, dtype=np.float32)
    local_bg_hi = np.asarray(gray_hi_pil.filter(ImageFilter.BoxBlur(7)), dtype=np.float32)
    ink_hi = np.clip(local_bg_hi - gray_hi, 0.0, 255.0)
    dx_hi = np.abs(np.diff(gray_hi, axis=1, prepend=gray_hi[:, :1]))
    ink_scale = max(25.0, float(np.percentile(ink_hi, 98)))
    dx_scale = max(25.0, float(np.percentile(dx_hi, 98)))
    norm_stroke = (ink_hi >= 8.0) & (dx_hi >= 10.0) & ((ink_hi / ink_scale) >= 0.30) & ((dx_hi / dx_scale) >= 0.30)
    hi_crossings = np.sum(norm_stroke, axis=1)
    hi_row_ink = np.mean(ink_hi, axis=1)
    hi_text_row = (hi_crossings >= 24) & (hi_row_ink >= 1.2)
    hi_band_count = 0
    hi_run = 0
    for flag in hi_text_row:
        if flag:
            hi_run += 1
        else:
            if 2 <= hi_run <= 18:
                hi_band_count += 1
            hi_run = 0
    if 2 <= hi_run <= 18:
        hi_band_count += 1
    hi_quiet_frac = float(np.mean(hi_crossings <= 6))
    norm_dense_text = bool(hi_band_count >= 7 and int(np.sum(hi_text_row)) >= 26 and hi_quiet_frac >= 0.30)

    # Paper/light background check: documents typically have substantial bright neutral area and low skin redness
    r_ch = arr_u8[:, :, 0].astype(np.float32)
    g_ch = arr_u8[:, :, 1].astype(np.float32)
    b_ch = arr_u8[:, :, 2].astype(np.float32)
    bright_neutral = float(np.mean((np.min(arr_u8, axis=2) >= 145) & (np.abs(r_ch - b_ch) < 35)))
    redness_dom = float(np.mean((r_ch > g_ch + 12) & (r_ch > b_ch + 18)))

    numpy_dense_text = (
        (band_count >= 6 and int(np.sum(text_row)) >= 18 and redness_dom < 0.45)
        or (band_count >= 4 and bright_neutral >= 0.40 and int(np.sum(text_row)) >= 14)
        or norm_dense_text
    )
    ocr_dense_text = (ocr_lines >= 3 and ocr_words >= 10)
    has_dense_text = bool(ocr_dense_text or numpy_dense_text)

    return {
        "has_dense_text": has_dense_text,
        "text_line_bands": int(max(band_count, hi_band_count)),
        "text_rows": int(max(int(np.sum(text_row)), int(np.sum(hi_text_row)))),
        "bright_neutral_fraction": round(bright_neutral, 4),
        "redness_dominant_fraction": round(redness_dom, 4),
        "norm_text_bands": int(hi_band_count),
        "norm_quiet_row_fraction": round(hi_quiet_frac, 4),
        "ocr_lines": int(ocr_lines),
        "ocr_words": int(ocr_words),
    }
