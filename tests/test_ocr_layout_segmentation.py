"""Tests for two-column OCR layout segmentation and reading-order reconstruction."""

from __future__ import annotations

import pytest

from medresearch_platform.documents.ocr_benchmark import (
    compute_cer,
    render_noisy_scan_image,
    tesseract_available,
)
from medresearch_platform.documents.ocr_engine import (
    detect_column_split,
    preprocess_scan_image,
    run_tesseract_ocr,
)


def test_detect_column_split_single_vs_two_column() -> None:
    single_text = (
        "Patient presented with acute dyspnoea and fever.\n"
        "Started amoxicillin 1g TID for 7 days.\n"
        "Follow up scheduled next week in cardiology clinic."
    )
    img_single = render_noisy_scan_image(single_text, two_columns=False, dpi=200, skew_deg=0.0, noise_std=2.0, seed=1)
    prep_single, _, _ = preprocess_scan_image(img_single)
    assert detect_column_split(prep_single) is None

    two_col_text = "\n".join(
        [f"Left line {i:02d} patient takes aspirin daily." for i in range(1, 9)]
        + [f"Right line {i:02d} blood pressure is normal." for i in range(1, 9)]
    )
    img_two = render_noisy_scan_image(
        two_col_text,
        two_columns=True,
        stamps_stains=True,
        dpi=200,
        skew_deg=1.5,
        noise_std=4.0,
        seed=2,
    )
    prep_two, _, _ = preprocess_scan_image(img_two)
    split_x = detect_column_split(prep_two)
    assert split_x is not None
    w = prep_two.shape[1]
    assert int(w * 0.35) <= split_x <= int(w * 0.65)


def test_two_column_ocr_reading_order_reduces_cer() -> None:
    ok, reason = tesseract_available(("eng",))
    if not ok:
        pytest.skip(reason)

    left_lines = [f"LEFT_COL_{i:02d} aspirin 100mg daily" for i in range(1, 7)]
    right_lines = [f"RIGHT_COL_{i:02d} metformin 500mg BID" for i in range(1, 7)]
    ref_text = "\n".join(left_lines + right_lines)
    img = render_noisy_scan_image(
        ref_text,
        two_columns=True,
        dpi=200,
        skew_deg=1.0,
        noise_std=3.0,
        seed=3,
    )
    res_flat = run_tesseract_ocr(img, lang_hint="en", layout_aware=False)
    res_layout = run_tesseract_ocr(img, lang_hint="en", layout_aware=True)
    cer_flat = compute_cer(ref_text, res_flat.raw_text)
    cer_layout = compute_cer(ref_text, res_layout.raw_text)
    assert cer_layout < cer_flat
    assert cer_layout < 0.15
