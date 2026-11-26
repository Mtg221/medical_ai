# ADR-0007: Tesseract 5 + OpenCV + PyMuPDF for Real-World Document OCR

- **Status:** Accepted
- **Date:** 2026-11-14
- **References:** `docs/MASTER_PROMPT.md` Parts B.1, C.1, Phase 5; `docs/EXTERNAL_COMPONENTS.md`

## Context

`CustomOCRConvNet` in `medresearch_platform/ai/ocr_model.py` only classifies synthetic rendered glyph patches and cannot read real scanned PDFs or document photographs. Real-world bilingual (EN/FR) scanned medical documents require robust page rendering, image preprocessing, and word-level bounding boxes for source-linked UI highlighting.

## Decision

1. **External OCR & PDF Tooling:** Adopt Tesseract 5 (`eng+fra`) via `pytesseract`, OpenCV preprocessing (deskew, denoise, adaptive thresholding), and PyMuPDF (`fitz`) for page-by-page text/image detection, 300 dpi rendering, and word bounding-box mapping (Phase 5).
2. **Documentation & Transparency:** Document all external document/OCR utilities in `docs/EXTERNAL_COMPONENTS.md`. Keep `CustomOCRConvNet` isolated in `extras/` as a research-only artifact.

## Consequences

- **Positive:** Real scanned PDFs and photos in EN and FR produce usable text with bounding boxes for source-linked highlighting.
- **Negative / Limitations:** Requires `tesseract` (`eng` + `fra` traineddata) installed on the host OS; OCR-induced errors must be tracked separately from digital-text extraction errors.
