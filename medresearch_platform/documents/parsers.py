"""Native & OCR document parsing (PDF, DOCX, TXT, PNG/JPEG/TIFF/WEBP), language detection, and section segmentation."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import docx
import pdfplumber
import pypdfium2 as pdfium
from PIL import Image

from medresearch_platform.documents.ocr_engine import WordBox, run_tesseract_ocr
from medresearch_platform.documents.tables import ExtractedTable, extract_pdf_tables

DetectedLanguage = Literal["en", "fr", "mixed_en_fr", "unknown"]
ExtractionSource = Literal["NATIVE_PARSER", "TESSERACT_OCR", "CUSTOM_OCR_MODEL"]
ExtractionMode = Literal["NATIVE_TEXT", "TESSERACT_OCR", "CUSTOM_OCR", "HYBRID"]


@dataclass(frozen=True)
class ParsedSection:
    page_number: int
    section_type: str
    heading_text: str
    body_text: str
    char_start: int
    char_end: int
    bbox: dict[str, int]


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    width_px: int
    height_px: int
    raw_text: str
    extraction_source: ExtractionSource
    mean_confidence: float
    mean_uncertainty: float
    sections: list[ParsedSection]
    word_boxes: list[WordBox] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedDocumentBundle:
    file_format: str
    page_count: int
    detected_language: DetectedLanguage
    document_type: str
    specialty: str
    extraction_mode: ExtractionMode
    pages: list[ParsedPage]
    image_object: Image.Image | None = None


_EN_MARKERS = frozenset({
    "patient", "history", "presents", "with", "denies", "ruled", "out", "prescribed",
    "daily", "blood", "pressure", "chest", "pain", "shortness", "breath", "discharge",
    "medications", "laboratory", "assessment", "diagnosis",
})
_FR_MARKERS = frozenset({
    "patient", "antécédents", "antecedents", "présente", "presente", "avec", "sans",
    "écarté", "ecarte", "prescrit", "jour", "douleur", "thoracique", "dyspnée",
    "traitement", "biologie", "conclusion", "hospitalisation", "diagnostic", "mère",
})

_SECTION_HEADINGS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("CHIEF_COMPLAINT", re.compile(r"^(?:CHIEF COMPLAINT|REASON FOR VISIT|MOTIF D'HOSPITALISATION|MOTIF DE CONSULTATION)\s*:", re.IGNORECASE | re.MULTILINE)),
    ("HISTORY_PRESENT_ILLNESS", re.compile(r"^(?:HISTORY OF PRESENT ILLNESS|CLINICAL HISTORY|HISTOIRE DE LA MALADIE)\s*:", re.IGNORECASE | re.MULTILINE)),
    ("PAST_MEDICAL_HISTORY", re.compile(r"^(?:PAST MEDICAL HISTORY|FAMILY HISTORY|ANTÉCÉDENTS|ANTECEDENTS)\s*:", re.IGNORECASE | re.MULTILINE)),
    ("MEDICATIONS", re.compile(r"^(?:MEDICATIONS|CURRENT MEDICATIONS|TRAITEMENT|ORDONNANCE)\s*:", re.IGNORECASE | re.MULTILINE)),
    ("LABORATORY_RESULTS", re.compile(r"^(?:LABORATORY RESULTS|LABS|BIOLOGIE|RÉSULTATS BIOLOGIQUES)\s*:", re.IGNORECASE | re.MULTILINE)),
    ("IMAGING_PROCEDURES", re.compile(r"^(?:IMAGING|PROCEDURES|IMAGERIE|EXAMENS COMPLÉMENTAIRES)\s*:", re.IGNORECASE | re.MULTILINE)),
    ("ASSESSMENT_PLAN", re.compile(r"^(?:ASSESSMENT AND PLAN|ASSESSMENT|CONCLUSION|SYNTHÈSE CLINIQUE)\s*:", re.IGNORECASE | re.MULTILINE)),
)

_MIN_DIGITAL_CHARS_PER_PAGE = 20


def detect_document_language(text: str) -> DetectedLanguage:
    """Detect whether clinical text is English, French, or mixed English/French."""
    tokens = [t.strip(".,:;()[]{}\"'").lower() for t in text.split()]
    if not tokens:
        return "unknown"
    en_hits = sum(1 for t in tokens if t in _EN_MARKERS and t != "patient")
    fr_hits = sum(1 for t in tokens if t in _FR_MARKERS and t != "patient")
    if en_hits >= 2 and fr_hits >= 2:
        return "mixed_en_fr"
    if fr_hits > en_hits and fr_hits >= 1:
        return "fr"
    if en_hits >= 1:
        return "en"
    return "unknown"


def segment_page_sections(page_text: str, page_number: int = 1) -> list[ParsedSection]:
    """Segment page text into clinical sections with character spans and layout coordinates."""
    matches: list[tuple[int, int, str, str]] = []
    for sec_type, pattern in _SECTION_HEADINGS:
        for m in pattern.finditer(page_text):
            matches.append((m.start(), m.end(), sec_type, m.group(0).strip()))

    matches.sort(key=lambda x: x[0])
    if not matches:
        return [
            ParsedSection(
                page_number=page_number,
                section_type="CLINICAL_BODY",
                heading_text="DOCUMENT_BODY",
                body_text=page_text,
                char_start=0,
                char_end=len(page_text),
                bbox={"x": 24, "y": 24, "width": 552, "height": max(60, page_text.count("\n") * 20)},
            )
        ]

    sections: list[ParsedSection] = []
    if matches[0][0] > 0:
        preamble = page_text[: matches[0][0]].strip()
        if preamble:
            sections.append(
                ParsedSection(
                    page_number=page_number,
                    section_type="HEADER_METADATA",
                    heading_text="HEADER",
                    body_text=preamble,
                    char_start=0,
                    char_end=matches[0][0],
                    bbox={"x": 24, "y": 16, "width": 552, "height": 48},
                )
            )

    for idx, (h_start, h_end, sec_type, heading_str) in enumerate(matches):
        next_start = matches[idx + 1][0] if (idx + 1) < len(matches) else len(page_text)
        body_str = page_text[h_end:next_start].strip()
        sections.append(
            ParsedSection(
                page_number=page_number,
                section_type=sec_type,
                heading_text=heading_str,
                body_text=body_str,
                char_start=h_start,
                char_end=next_start,
                bbox={
                    "x": 24,
                    "y": 72 + idx * 90,
                    "width": 552,
                    "height": max(40, (body_str.count("\n") + 1) * 22),
                },
            )
        )
    return sections


def _align_pdfplumber_word_boxes(
    raw_text: str,
    raw_words: list[dict[str, Any]],
    page_number: int,
) -> list[WordBox]:
    """Align pdfplumber extracted words to character offsets in `raw_text`."""
    boxes: list[WordBox] = []
    search_from = 0
    for w in raw_words:
        word_str = str(w.get("text", "")).strip()
        if not word_str:
            continue
        pos = raw_text.find(word_str, search_from)
        if pos == -1:
            pos = raw_text.find(word_str)
        if pos == -1:
            continue
        search_from = pos + len(word_str)
        x0 = float(w.get("x0", 0.0) or 0.0)
        top = float(w.get("top", 0.0) or 0.0)
        x1 = float(w.get("x1", x0 + 1.0) or (x0 + 1.0))
        bottom = float(w.get("bottom", top + 1.0) or (top + 1.0))
        boxes.append(
            WordBox(
                text=word_str,
                char_start=pos,
                char_end=pos + len(word_str),
                x=int(round(x0)),
                y=int(round(top)),
                width=max(1, int(round(x1 - x0))),
                height=max(1, int(round(bottom - top))),
                confidence=1.0,
                page_number=page_number,
            )
        )
    return boxes


def _render_pdf_page_300dpi(pdf_doc: pdfium.PdfDocument, page_index_zero: int) -> Image.Image:
    """Render a single PDF page at 300 DPI using pypdfium2."""
    page = pdf_doc[page_index_zero]
    bitmap = page.render(scale=300.0 / 72.0)
    pil_image = bitmap.to_pil()
    page.close()
    return pil_image


def parse_document_bytes(
    raw_bytes: bytes,
    file_format: str,
    *,
    document_type: str = "CLINICAL_SUMMARY",
    specialty: str = "INTERNAL_MEDICINE",
    ocr_dpi_render: bool = True,
) -> ParsedDocumentBundle:
    """Parse digital or scanned PDF, DOCX, TXT, or Image payloads without mutating original bytes."""
    fmt = file_format.upper()

    if fmt == "TXT":
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1", errors="replace")
        sections = segment_page_sections(text, page_number=1)
        lang = detect_document_language(text)
        return ParsedDocumentBundle(
            file_format="TXT",
            page_count=1,
            detected_language=lang,
            document_type=document_type,
            specialty=specialty,
            extraction_mode="NATIVE_TEXT",
            pages=[
                ParsedPage(
                    page_number=1,
                    width_px=600,
                    height_px=800,
                    raw_text=text,
                    extraction_source="NATIVE_PARSER",
                    mean_confidence=1.0,
                    mean_uncertainty=0.0,
                    sections=sections,
                )
            ],
        )

    if fmt == "PDF":
        tables_by_page = extract_pdf_tables(raw_bytes)
        pages: list[ParsedPage] = []
        all_texts: list[str] = []
        pdf_doc: pdfium.PdfDocument | None = None
        has_native = False
        has_ocr = False

        try:
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                for idx, ppage in enumerate(pdf.pages, start=1):
                    extracted_txt = (ppage.extract_text() or "").strip()
                    page_w = int(round(float(ppage.width)))
                    page_h = int(round(float(ppage.height)))
                    page_tables = tables_by_page.get(idx, [])

                    if len(extracted_txt) >= _MIN_DIGITAL_CHARS_PER_PAGE:
                        has_native = True
                        raw_words = ppage.extract_words() or []
                        word_boxes = _align_pdfplumber_word_boxes(extracted_txt, raw_words, page_number=idx)
                        sections = segment_page_sections(extracted_txt, page_number=idx)
                        all_texts.append(extracted_txt)
                        pages.append(
                            ParsedPage(
                                page_number=idx,
                                width_px=page_w,
                                height_px=page_h,
                                raw_text=extracted_txt,
                                extraction_source="NATIVE_PARSER",
                                mean_confidence=1.0,
                                mean_uncertainty=0.0,
                                sections=sections,
                                word_boxes=word_boxes,
                                tables=page_tables,
                            )
                        )
                    else:
                        has_ocr = True
                        if ocr_dpi_render:
                            if pdf_doc is None:
                                pdf_doc = pdfium.PdfDocument(raw_bytes)
                            rendered = _render_pdf_page_300dpi(pdf_doc, idx - 1)
                            ocr_res = run_tesseract_ocr(rendered, page_number=idx)
                            ocr_text = ocr_res.raw_text.strip()
                            sections = segment_page_sections(ocr_text, page_number=idx) if ocr_text else []
                            all_texts.append(ocr_text)
                            pages.append(
                                ParsedPage(
                                    page_number=idx,
                                    width_px=ocr_res.width_px,
                                    height_px=ocr_res.height_px,
                                    raw_text=ocr_text,
                                    extraction_source="TESSERACT_OCR",
                                    mean_confidence=ocr_res.mean_confidence,
                                    mean_uncertainty=ocr_res.mean_uncertainty,
                                    sections=sections,
                                    word_boxes=ocr_res.word_boxes,
                                    tables=page_tables,
                                )
                            )
                        else:
                            all_texts.append("")
                            pages.append(
                                ParsedPage(
                                    page_number=idx,
                                    width_px=page_w,
                                    height_px=page_h,
                                    raw_text="",
                                    extraction_source="TESSERACT_OCR",
                                    mean_confidence=0.0,
                                    mean_uncertainty=1.0,
                                    sections=[],
                                    tables=page_tables,
                                )
                            )
        finally:
            if pdf_doc is not None:
                pdf_doc.close()

        combined = "\n".join(all_texts)
        if has_native and has_ocr:
            mode: ExtractionMode = "HYBRID"
        elif has_ocr:
            mode = "TESSERACT_OCR"
        else:
            mode = "NATIVE_TEXT"

        return ParsedDocumentBundle(
            file_format="PDF",
            page_count=max(1, len(pages)),
            detected_language=detect_document_language(combined),
            document_type=document_type,
            specialty=specialty,
            extraction_mode=mode,
            pages=pages,
        )

    if fmt == "DOCX":
        doc = docx.Document(io.BytesIO(raw_bytes))
        lines: list[str] = [p.text for p in doc.paragraphs if p.text.strip()]
        docx_tables: list[ExtractedTable] = []
        for table in doc.tables:
            cleaned: list[list[str]] = []
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))
                    cleaned.append(cells)
            if cleaned:
                docx_tables.append(
                    ExtractedTable(
                        page_number=1,
                        headers=cleaned[0],
                        rows=cleaned[1:] if len(cleaned) > 1 else [],
                        bbox={"x": 24, "y": 400, "width": 552, "height": max(40, len(cleaned) * 24)},
                    )
                )
        full_text = "\n".join(lines)
        sections = segment_page_sections(full_text, page_number=1)
        return ParsedDocumentBundle(
            file_format="DOCX",
            page_count=1,
            detected_language=detect_document_language(full_text),
            document_type=document_type,
            specialty=specialty,
            extraction_mode="NATIVE_TEXT",
            pages=[
                ParsedPage(
                    page_number=1,
                    width_px=600,
                    height_px=800,
                    raw_text=full_text,
                    extraction_source="NATIVE_PARSER",
                    mean_confidence=1.0,
                    mean_uncertainty=0.0,
                    sections=sections,
                    tables=docx_tables,
                )
            ],
        )

    if fmt in ("PNG", "JPEG", "TIFF", "WEBP"):
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
        w, h = img.size
        ocr_res = run_tesseract_ocr(img, page_number=1)
        ocr_text = ocr_res.raw_text.strip()
        lang = detect_document_language(ocr_text) if ocr_text else "unknown"
        sections = segment_page_sections(ocr_text, page_number=1) if ocr_text else []
        return ParsedDocumentBundle(
            file_format=fmt,
            page_count=1,
            detected_language=lang,
            document_type=document_type,
            specialty=specialty,
            extraction_mode="TESSERACT_OCR",
            pages=[
                ParsedPage(
                    page_number=1,
                    width_px=w,
                    height_px=h,
                    raw_text=ocr_text,
                    extraction_source="TESSERACT_OCR",
                    mean_confidence=ocr_res.mean_confidence,
                    mean_uncertainty=ocr_res.mean_uncertainty,
                    sections=sections,
                    word_boxes=ocr_res.word_boxes,
                )
            ],
            image_object=img,
        )

    raise ValueError(f"Unsupported format for parsing: {file_format}")
