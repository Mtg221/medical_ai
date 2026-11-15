"""Real-world OCR engine (OpenCV preprocessing + Tesseract 5 EN/FR) with word bounding boxes."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output


@dataclass(frozen=True)
class WordBox:
    """Single recognized or native-extracted word with page pixel coordinates and char offsets."""

    text: str
    char_start: int
    char_end: int
    x: int
    y: int
    width: int
    height: int
    confidence: float
    page_number: int


@dataclass(frozen=True)
class OCRPageResult:
    """Result of running OCR on a single rendered page or scanned image."""

    raw_text: str
    word_boxes: list[WordBox]
    mean_confidence: float
    mean_uncertainty: float
    width_px: int
    height_px: int
    skew_angle_deg: float
    engine: str


def _resolve_tesseract_cmd() -> str | None:
    """Locate the tesseract binary on PATH or in standard local user prefixes."""
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "tesseract",
        Path.home() / ".local" / "tesseract-env" / "bin" / "tesseract",
        Path("/usr/local/bin/tesseract"),
        Path("/opt/homebrew/bin/tesseract"),
    ):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def check_tesseract_available() -> tuple[bool, list[str]]:
    """Return whether the `tesseract` binary is available and which traineddata languages exist."""
    cmd = _resolve_tesseract_cmd()
    if cmd is None:
        return False, []
    pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        langs = list(pytesseract.get_languages(config=""))
        return True, langs
    except Exception:
        return False, []


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate document skew angle in degrees using foreground bounding-box orientation."""
    # Invert so dark text becomes foreground (255)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return 0.0
    # minAreaRect expects (x, y) order
    pts = coords[:, ::-1].astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = float(rect[-1])
    if angle < -45.0:
        angle = 90.0 + angle
    elif angle > 45.0:
        angle = angle - 90.0
    # Ignore tiny jitter or 90-degree aspect-ratio flips
    if abs(angle) < 0.3 or abs(angle) > 15.0:
        return 0.0
    return angle


def preprocess_scan_image(image: Image.Image) -> tuple[np.ndarray, float, float]:
    """Preprocess a scanned page or photo for Tesseract OCR.

    Steps: grayscale conversion, upscaling small scans, skew estimation & deskewing,
    light denoising, adaptive binarization, and border padding.

    Returns:
        (preprocessed_uint8_image, scale_factor, detected_skew_angle_deg)
    """
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape[:2]
    scale = 1.0
    if h < 1000 or w < 800:
        scale = 2.0
        gray = cv2.resize(
            gray,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    skew_angle = _estimate_skew_angle(gray)
    if skew_angle != 0.0:
        gh, gw = gray.shape[:2]
        center = (gw / 2.0, gh / 2.0)
        mat = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        gray = cv2.warpAffine(
            gray,
            mat,
            (gw, gh),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    denoised = cv2.medianBlur(gray, 3) if min(gray.shape[:2]) > 60 else gray
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    padded = cv2.copyMakeBorder(
        binary,
        16,
        16,
        16,
        16,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    return padded, scale, skew_angle


def _select_tesseract_lang(available_langs: list[str], lang_hint: str | None) -> str:
    """Choose the best available Tesseract language string for EN/FR clinical scans."""
    avail = set(available_langs)
    if lang_hint == "fr" and "fra" in avail and "eng" in avail:
        return "fra+eng"
    if lang_hint == "fr" and "fra" in avail:
        return "fra"
    if "eng" in avail and "fra" in avail:
        return "eng+fra"
    if "eng" in avail:
        return "eng"
    if "fra" in avail:
        return "fra"
    if available_langs:
        return available_langs[0]
    return "eng"


def detect_column_split(preprocessed: np.ndarray) -> int | None:
    """Detect a vertical column gutter in a preprocessed binary scan image (255=bg, 0=ink).

    Returns the x-coordinate of the gutter midpoint if a clear two-column layout
    exists, or None for single-column pages.
    """
    h, w = preprocessed.shape[:2]
    if h < 80 or w < 120:
        return None

    ink = (preprocessed < 128).astype(np.uint8) * 255
    # Suppress wide horizontal rules (e.g. table borders) before projection
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 12), 1))
    horiz_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, horiz_kernel)
    ink_clean = cv2.subtract(ink, horiz_lines)

    y0, y1 = int(0.08 * h), int(0.92 * h)
    roi = (ink_clean[y0:y1, :] > 0).astype(np.float32)
    col_density = roi.mean(axis=0)

    left_mean = float(col_density[int(0.08 * w) : int(0.42 * w)].mean())
    right_mean = float(col_density[int(0.58 * w) : int(0.92 * w)].mean())
    if left_mean < 0.004 or right_mean < 0.004:
        return None

    band_start = int(0.42 * w)
    band_end = int(0.58 * w)
    min_gutter_w = max(10, int(0.012 * w))
    threshold = max(0.004, 0.12 * min(left_mean, right_mean))

    best_start = -1
    best_len = 0
    cur_start = -1
    for x in range(band_start, band_end):
        if col_density[x] <= threshold:
            if cur_start < 0:
                cur_start = x
        else:
            if cur_start >= 0 and (x - cur_start) > best_len:
                best_start = cur_start
                best_len = x - cur_start
            cur_start = -1
    if cur_start >= 0 and (band_end - cur_start) > best_len:
        best_start = cur_start
        best_len = band_end - cur_start

    if best_len < min_gutter_w or best_start < 0:
        return None
    return best_start + best_len // 2


def _extract_region_words(
    region_img: np.ndarray,
    *,
    tess_lang: str,
    config: str,
    scale: float,
    x_offset_px: int,
    pad: int = 16,
) -> list[list[tuple[str, int, int, int, int, float]]]:
    """Run pytesseract.image_to_data on one cropped region and return ordered lines of words."""
    data: dict[str, list[Any]] = pytesseract.image_to_data(
        region_img,
        lang=tess_lang,
        config=config,
        output_type=Output.DICT,
    )
    n_items = len(data.get("text", []))
    lines_order: list[tuple[int, int, int]] = []
    grouped: dict[tuple[int, int, int], list[tuple[str, int, int, int, int, float]]] = {}

    for i in range(n_items):
        word = str(data["text"][i]).strip()
        if not word:
            continue
        conf_raw = float(data["conf"][i])
        if conf_raw < 0:
            continue
        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        if key not in grouped:
            grouped[key] = []
            lines_order.append(key)

        wx = max(0, int(round((int(data["left"][i]) - pad + x_offset_px) / scale)))
        wy = max(0, int(round((int(data["top"][i]) - pad) / scale)))
        ww = max(1, int(round(int(data["width"][i]) / scale)))
        wh = max(1, int(round(int(data["height"][i]) / scale)))
        conf = max(0.0, min(1.0, conf_raw / 100.0))
        grouped[key].append((word, wx, wy, ww, wh, conf))

    return [grouped[key] for key in lines_order]


def run_tesseract_ocr(
    image: Image.Image,
    *,
    page_number: int = 1,
    lang_hint: str | None = None,
    psm: int = 6,
    layout_aware: bool = True,
) -> OCRPageResult:
    """Run OpenCV preprocessing, optional column layout segmentation, and Tesseract OCR."""
    orig_w, orig_h = image.size
    available, langs = check_tesseract_available()
    if not available:
        return OCRPageResult(
            raw_text="",
            word_boxes=[],
            mean_confidence=0.0,
            mean_uncertainty=1.0,
            width_px=orig_w,
            height_px=orig_h,
            skew_angle_deg=0.0,
            engine="unavailable",
        )

    preprocessed, scale, skew_angle = preprocess_scan_image(image)
    tess_lang = _select_tesseract_lang(langs, lang_hint)
    config = f"--oem 3 --psm {psm}"
    pad = 16

    split_x = detect_column_split(preprocessed) if layout_aware else None
    all_lines: list[list[tuple[str, int, int, int, int, float]]] = []
    if split_x is not None:
        left_crop = cv2.copyMakeBorder(preprocessed[:, :split_x], 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=255)
        right_crop = cv2.copyMakeBorder(preprocessed[:, split_x:], 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=255)
        all_lines.extend(
            _extract_region_words(
                left_crop, tess_lang=tess_lang, config=config, scale=scale, x_offset_px=0, pad=pad
            )
        )
        all_lines.extend(
            _extract_region_words(
                right_crop,
                tess_lang=tess_lang,
                config=config,
                scale=scale,
                x_offset_px=split_x - pad,
                pad=pad,
            )
        )
    else:
        all_lines.extend(
            _extract_region_words(
                preprocessed, tess_lang=tess_lang, config=config, scale=scale, x_offset_px=0, pad=pad
            )
        )

    out_parts: list[str] = []
    word_boxes: list[WordBox] = []
    cursor = 0
    confs: list[float] = []

    for line_idx, words in enumerate(all_lines):
        if line_idx > 0:
            out_parts.append("\n")
            cursor += 1
        for w_idx, (word, wx, wy, ww, wh, conf) in enumerate(words):
            if w_idx > 0:
                out_parts.append(" ")
                cursor += 1
            start = cursor
            out_parts.append(word)
            cursor += len(word)
            word_boxes.append(
                WordBox(
                    text=word,
                    char_start=start,
                    char_end=cursor,
                    x=wx,
                    y=wy,
                    width=ww,
                    height=wh,
                    confidence=round(conf, 4),
                    page_number=page_number,
                )
            )
            confs.append(conf)

    raw_text = "".join(out_parts)
    mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return OCRPageResult(
        raw_text=raw_text,
        word_boxes=word_boxes,
        mean_confidence=round(mean_conf, 4),
        mean_uncertainty=round(1.0 - mean_conf, 4),
        width_px=orig_w,
        height_px=orig_h,
        skew_angle_deg=round(skew_angle, 3),
        engine=f"tesseract:{tess_lang}",
    )
