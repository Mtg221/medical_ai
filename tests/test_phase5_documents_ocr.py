"""Phase 5 tests: real-world digital PDF word boxes/tables, OpenCV scan preprocessing, Tesseract OCR, and CER/WER."""

from __future__ import annotations

import io

import numpy as np
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from medresearch_platform.documents.ocr_benchmark import (
    build_digital_pdf_with_table,
    compute_cer,
    compute_wer,
    parse_and_benchmark_scan,
    render_noisy_scan_image,
)
from medresearch_platform.documents.ocr_engine import (
    check_tesseract_available,
    preprocess_scan_image,
)
from medresearch_platform.documents.parsers import parse_document_bytes


def test_cer_and_wer_metrics() -> None:
    assert compute_cer("Patient denies chest pain", "Patient denies chest pain") == 0.0
    assert compute_wer("Patient denies chest pain", "Patient denies chest pain") == 0.0
    assert 0.0 < compute_cer("metformin 500 mg", "metformin 500 ml") < 0.2
    assert compute_wer("metformin 500 mg", "metformin 500 ml") == pytest.approx(0.3333, abs=1e-3)


def test_digital_pdf_word_boxes_and_tables() -> None:
    pdf_bytes = build_digital_pdf_with_table(
        "LABORATORY RESULTS: Metabolic Panel",
        ["Test", "Value", "Unit"],
        [["HbA1c", "7.4", "%"], ["CRP", "18.2", "mg/L"]],
    )
    bundle = parse_document_bytes(pdf_bytes, "PDF")
    assert bundle.page_count == 1
    assert bundle.extraction_mode == "NATIVE_TEXT"
    page = bundle.pages[0]
    assert page.extraction_source == "NATIVE_PARSER"
    assert len(page.word_boxes) >= 6
    for wb in page.word_boxes[:10]:
        assert page.raw_text[wb.char_start:wb.char_end] == wb.text
        assert wb.width > 0 and wb.height > 0
    assert len(page.tables) >= 1
    assert page.tables[0].headers == ["Test", "Value", "Unit"]
    assert ["HbA1c", "7.4", "%"] in page.tables[0].rows


def test_opencv_preprocessing_deskews_and_binarizes() -> None:
    sample = "CHIEF COMPLAINT: Chest pain.\nMEDICATIONS: Aspirin 81 mg daily."
    img = render_noisy_scan_image(sample, skew_deg=3.0, noise_std=10.0, seed=7)
    pre, scale, skew = preprocess_scan_image(img)
    assert pre.dtype == np.uint8
    assert scale >= 1.0
    assert set(np.unique(pre)).issubset({0, 255})
    assert abs(skew) <= 15.0


def test_mixed_digital_and_scanned_pdf_hybrid_mode() -> None:
    scan_img = render_noisy_scan_image(
        "MEDICATIONS: Metformin 500 mg daily.",
        skew_deg=1.0,
        noise_std=4.0,
        seed=11,
    )
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(72, 720, "CHIEF COMPLAINT: Shortness of breath and fatigue for 3 days.")
    c.showPage()
    img_buf = io.BytesIO()
    scan_img.save(img_buf, format="PNG")
    img_buf.seek(0)
    pw, ph = letter
    c.drawImage(ImageReader(img_buf), 0, 0, width=pw, height=ph)
    c.showPage()
    c.save()

    bundle = parse_document_bytes(buf.getvalue(), "PDF")
    assert bundle.page_count == 2
    assert bundle.pages[0].extraction_source == "NATIVE_PARSER"
    assert bundle.pages[1].extraction_source == "TESSERACT_OCR"
    assert bundle.extraction_mode == "HYBRID"


def test_tesseract_ocr_en_and_fr_scanned_pdf_and_image() -> None:
    available, _ = check_tesseract_available()
    if not available:
        pytest.skip("tesseract binary not installed on host")

    en_ref = (
        "CHIEF COMPLAINT: Chest pain and shortness of breath.\n"
        "MEDICATIONS: Metformin 500 mg daily.\n"
        "ASSESSMENT: Type 2 diabetes mellitus."
    )
    en_pdf = parse_and_benchmark_scan("scan_en_pdf", "en", en_ref, skew_deg=1.5, noise_std=5.0, as_pdf=True)
    assert en_pdf.extraction_mode == "TESSERACT_OCR"
    assert en_pdf.word_box_count >= 10
    assert en_pdf.cer <= 0.25
    assert en_pdf.wer <= 0.35

    fr_ref = (
        "MOTIF DE CONSULTATION: Douleur thoracique et dyspnee.\n"
        "TRAITEMENT: Metformine 500 mg par jour.\n"
        "CONCLUSION: Diabete de type 2."
    )
    fr_png = parse_and_benchmark_scan("scan_fr_png", "fr", fr_ref, skew_deg=-1.5, noise_std=5.0, as_pdf=False)
    assert fr_png.extraction_mode == "TESSERACT_OCR"
    assert fr_png.word_box_count >= 10
    assert fr_png.cer <= 0.25
    assert fr_png.wer <= 0.35
