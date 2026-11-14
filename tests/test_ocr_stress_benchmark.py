"""Tests for Phase 11 Step 5: realistic OCR stress benchmark (F6)."""

from __future__ import annotations

from medresearch_platform.documents.ocr_benchmark import parse_and_benchmark_scan


def test_ocr_stress_tiers_show_realistic_degradation() -> None:
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
        skew_deg=4.5,
        noise_std=22.0,
        blur_radius=1.2,
        jpeg_quality=40,
        contrast=0.78,
        as_pdf=False,
    )
    assert clean.word_box_count >= 25
    assert heavy.cer >= clean.cer
    assert heavy.cer > 0.0
