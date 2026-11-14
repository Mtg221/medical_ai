"""Native document parsing (PDF, DOCX, TXT, PNG/JPEG/TIFF), language detection, and section segmentation."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Literal

import docx
from PIL import Image
from pypdf import PdfReader

DetectedLanguage = Literal["en", "fr", "mixed_en_fr", "unknown"]


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
    extraction_source: Literal["NATIVE_PARSER", "CUSTOM_OCR_MODEL"]
    mean_confidence: float
    mean_uncertainty: float
    sections: list[ParsedSection]


@dataclass(frozen=True)
class ParsedDocumentBundle:
    file_format: str
    page_count: int
    detected_language: DetectedLanguage
    document_type: str
    specialty: str
    extraction_mode: Literal["NATIVE_TEXT", "CUSTOM_OCR", "HYBRID"]
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


def parse_document_bytes(
    raw_bytes: bytes,
    file_format: str,
    *,
    document_type: str = "CLINICAL_SUMMARY",
    specialty: str = "INTERNAL_MEDICINE",
) -> ParsedDocumentBundle:
    """Parse native PDF, DOCX, TXT, or Image payloads without mutating original bytes."""
    fmt = file_format.upper()

    if fmt == "TXT":
        text = raw_bytes.decode("utf-8", errors="replace")
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
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages: list[ParsedPage] = []
        all_texts: list[str] = []
        for idx, pdf_page in enumerate(reader.pages, start=1):
            extracted_txt = (pdf_page.extract_text() or "").strip()
            all_texts.append(extracted_txt)
            sections = segment_page_sections(extracted_txt, page_number=idx)
            pages.append(
                ParsedPage(
                    page_number=idx,
                    width_px=612,
                    height_px=792,
                    raw_text=extracted_txt,
                    extraction_source="NATIVE_PARSER",
                    mean_confidence=1.0 if extracted_txt else 0.0,
                    mean_uncertainty=0.0 if extracted_txt else 1.0,
                    sections=sections,
                )
            )
        combined = "\n".join(all_texts)
        return ParsedDocumentBundle(
            file_format="PDF",
            page_count=max(1, len(pages)),
            detected_language=detect_document_language(combined),
            document_type=document_type,
            specialty=specialty,
            extraction_mode="NATIVE_TEXT" if combined.strip() else "CUSTOM_OCR",
            pages=pages,
        )

    if fmt == "DOCX":
        doc = docx.Document(io.BytesIO(raw_bytes))
        lines: list[str] = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))
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
                )
            ],
        )

    if fmt in ("PNG", "JPEG", "TIFF", "WEBP"):
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
        w, h = img.size
        return ParsedDocumentBundle(
            file_format=fmt,
            page_count=1,
            detected_language="unknown",
            document_type=document_type,
            specialty=specialty,
            extraction_mode="CUSTOM_OCR",
            pages=[
                ParsedPage(
                    page_number=1,
                    width_px=w,
                    height_px=h,
                    raw_text="",
                    extraction_source="CUSTOM_OCR_MODEL",
                    mean_confidence=0.0,
                    mean_uncertainty=1.0,
                    sections=[],
                )
            ],
            image_object=img,
        )

    raise ValueError(f"Unsupported format for parsing: {file_format}")
