"""Phase 5 OCR robustness tests: EN/FR clean, faint photocopy, skewed, and noisy scans + offset preservation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from medresearch_platform.documents.ocr_engine import (
    check_tesseract_available,
    run_tesseract_ocr,
)


def _cer(ref: str, hyp: str) -> float:
    r = " ".join(ref.split())
    h = " ".join(hyp.split())
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev if r[i - 1] == h[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m] / max(1, n)


def _render_note(text: str, font_size: int = 22) -> Image.Image:
    im = Image.new("RGB", (1100, 380), "white")
    draw = ImageDraw.Draw(im)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except Exception:
            continue
    else:
        font = ImageFont.load_default()
    y = 35
    for line in text.splitlines():
        draw.text((40, y), line, fill="black", font=font)
        y += int(font_size * 1.7)
    return im


def test_phase5_ocr_en_fr_faint_copy_skew_and_offset_preservation() -> None:
    avail, langs = check_tesseract_available()
    if not avail or "eng" not in langs or "fra" not in langs:
        pytest.skip("Tesseract with eng+fra packs not available")

    en_txt = (
        "Discharge Summary - Date: 2025-03-15\n"
        "Medication: Metformin 500 mg twice daily for 14 days.\n"
        "No evidence of pneumonia. HbA1c 7.4 %."
    )
    fr_txt = (
        "Compte rendu médical - Date : 15 mars 2025\n"
        "Traitement : Metformine 500 mg deux fois par jour pendant 14 jours.\n"
        "Absence de pneumonie."
    )

    for lang, txt in (("en", en_txt), ("fr", fr_txt)):
        base = _render_note(txt)
        arr = np.array(base, dtype=np.float32)
        faint = Image.fromarray(np.clip(215 - (255 - arr) * 0.18, 0, 255).astype(np.uint8))
        skewed = base.rotate(6.0, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")

        for label, img in (("clean", base), ("faint_copy", faint), ("skew6", skewed)):
            res = run_tesseract_ocr(img, lang_hint=lang)
            cer = _cer(txt, res.raw_text)
            assert cer < 0.08, f"{lang}/{label} CER too high: {cer:.4f} (text={res.raw_text!r})"
            assert "500" in res.raw_text, f"Dosage number lost in {lang}/{label}: {res.raw_text!r}"
            # Strict character-offset invariant for evidence highlighting
            for wb in res.word_boxes:
                assert res.raw_text[wb.char_start : wb.char_end] == wb.text
            if label == "skew6":
                assert abs(res.skew_angle_deg) >= 4.0
