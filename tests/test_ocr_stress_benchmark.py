"""Tests for Phase 11/11B OCR stress benchmark (G9)."""

from __future__ import annotations

import pytest

from medresearch_platform.documents.ocr_benchmark import (
    map_ocr_spans_to_reference,
    parse_and_benchmark_scan,
    render_noisy_scan_image,
    tesseract_available,
)


def test_tesseract_available_reports_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("medresearch_platform.documents.ocr_benchmark.shutil.which", lambda _: None)
    ok, reason = tesseract_available(("eng", "fra"))
    assert ok is False
    assert "Tesseract" in reason


def test_render_noisy_scan_supports_all_degradations() -> None:
    sample = "Header: Admission\nSymptom: fever for 3 days.\nMedication: Amoxicillin 500 mg oral."
    for font_idx in (0, 1, 2):
        img = render_noisy_scan_image(
            sample,
            dpi=150,
            skew_deg=4.0,
            noise_std=10.0,
            blur_radius=0.8,
            jpeg_quality=50,
            perspective=0.4,
            uneven_lighting=0.3,
            stamps_stains=True,
            two_columns=(font_idx == 1),
            table_mode=(font_idx == 2),
            font_index=font_idx,
        )
        assert img.width > 500
        assert img.height > 800


def test_map_ocr_spans_to_reference() -> None:
    ref = "Patient has acute pneumonia and fever."
    ocr = "Pat1ent has acute pneumonia and fever."
    s = ocr.index("acute pneumonia")
    e = s + len("acute pneumonia")
    mapped = map_ocr_spans_to_reference(ocr, ref, [(s, e)])
    assert mapped == [(ref.index("acute pneumonia"), ref.index("acute pneumonia") + len("acute pneumonia"))]


def test_ocr_stress_tiers_show_realistic_degradation() -> None:
    ok, reason = tesseract_available(("eng", "fra"))
    if not ok:
        pytest.skip(reason)

    text = (
        "INPATIENT CONSULTATION NOTE — Date: 2025-03-12 | Patient ID: PAT-EN-01001\n\n"
        "Chief complaint: substernal chest pain for 7 days.\n"
        "Medications: Atorvastatin 40 mg oral once daily.\n"
        "Lab results: High-sensitivity troponin I 42.5 ng/L."
    )
    clean = parse_and_benchmark_scan(
        "ocr_clean",
        "en",
        text,
        dpi=300,
        skew_deg=0.5,
        noise_std=3.0,
        blur_radius=0.0,
        jpeg_quality=None,
        contrast=1.0,
        as_pdf=False,
    )
    heavy = parse_and_benchmark_scan(
        "ocr_heavy",
        "en",
        text,
        dpi=150,
        skew_deg=4.5,
        noise_std=22.0,
        blur_radius=1.2,
        jpeg_quality=40,
        contrast=0.78,
        perspective=0.3,
        uneven_lighting=0.3,
        stamps_stains=True,
        as_pdf=False,
    )
    assert clean.word_box_count >= 25
    assert heavy.cer >= clean.cer
    assert heavy.cer > 0.0
