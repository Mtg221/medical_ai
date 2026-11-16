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
        text = text.replace("\r\n", "\n").replace("\r", "\n")
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


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file's extension or magic bytes are not in the allow-list (HTTP 415)."""


class PayloadTooLargeError(ValueError):
    """Raised when an uploaded file exceeds size, page, or character limits (HTTP 413)."""


_ALLOWED_EXTENSIONS: dict[str, str] = {
    ".txt": "TXT",
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}

_BINARY_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"MZ", "Windows PE executable (MZ)"),
    (b"\x7fELF", "ELF executable"),
    (b"\xfe\xed\xfa\xce", "Mach-O binary"),
    (b"\xfe\xed\xfa\xcf", "Mach-O 64-bit binary"),
    (b"\xce\xfa\xed\xfe", "Mach-O binary"),
    (b"\xcf\xfa\xed\xfe", "Mach-O 64-bit binary"),
    (b"\xca\xfe\xba\xbe", "Mach-O universal / Java class binary"),
)


def validate_upload_payload(
    raw_bytes: bytes,
    filename: str,
    *,
    max_upload_bytes: int = 10 * 1024 * 1024,
) -> str:
    """Validate file size, extension allow-list, and magic bytes before parsing or queueing."""
    if len(raw_bytes) > max_upload_bytes:
        raise PayloadTooLargeError(
            f"File '{filename}' ({len(raw_bytes):,} bytes) exceeds the maximum allowed upload size "
            f"({max_upload_bytes:,} bytes). / La taille du fichier dépasse la limite maximale autorisée."
        )
    if len(raw_bytes) == 0:
        raise ValueError(
            f"File '{filename}' is empty (0 bytes). / Le fichier téléversé est vide."
        )

    for sig, desc in _BINARY_SIGNATURES:
        if raw_bytes.startswith(sig):
            raise UnsupportedFileTypeError(
                f"Rejected executable or binary payload ({desc}) in '{filename}'. "
                f"Allowed formats: TXT, PDF, DOCX, PNG, JPEG, TIFF, WEBP. / "
                f"Fichier exécutable ou binaire non autorisé ({desc})."
            )

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file extension '{ext or '(none)'}' for '{filename}'. "
            f"Allowed formats: TXT, PDF, DOCX, PNG, JPEG, TIFF, WEBP. / "
            f"Extension de fichier non prise en charge."
        )

    expected_fmt = _ALLOWED_EXTENSIONS[ext]
    if expected_fmt == "PDF":
        if not raw_bytes.startswith(b"%PDF-"):
            raise UnsupportedFileTypeError(
                f"File '{filename}' has a .pdf extension but does not start with %PDF- magic bytes. / "
                f"Signature PDF (%PDF-) invalide."
            )
    elif expected_fmt == "DOCX":
        if not raw_bytes.startswith(b"PK\x03\x04"):
            raise UnsupportedFileTypeError(
                f"File '{filename}' has a .docx extension but is not a valid ZIP/OOXML container. / "
                f"Conteneur DOCX invalide."
            )
        import zipfile
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                if "[Content_Types].xml" not in zf.namelist():
                    raise UnsupportedFileTypeError(
                        f"File '{filename}' is a ZIP archive without [Content_Types].xml (not a valid DOCX)."
                    )
        except zipfile.BadZipFile as exc:
            raise UnsupportedFileTypeError(f"File '{filename}' is a corrupted ZIP/DOCX container.") from exc
    elif expected_fmt == "PNG":
        if not raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise UnsupportedFileTypeError(f"File '{filename}' does not have valid PNG magic bytes.")
    elif expected_fmt == "JPEG":
        if not raw_bytes.startswith(b"\xff\xd8\xff"):
            raise UnsupportedFileTypeError(f"File '{filename}' does not have valid JPEG magic bytes.")
    elif expected_fmt == "TIFF":
        if not (raw_bytes.startswith(b"II*\x00") or raw_bytes.startswith(b"MM\x00*")):
            raise UnsupportedFileTypeError(f"File '{filename}' does not have valid TIFF magic bytes.")
    elif expected_fmt == "WEBP":
        if not (raw_bytes.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP"):
            raise UnsupportedFileTypeError(f"File '{filename}' does not have valid WEBP magic bytes.")
    elif expected_fmt == "TXT":
        if b"\x00" in raw_bytes[:8192]:
            raise UnsupportedFileTypeError(
                f"File '{filename}' contains binary null bytes and is not valid plain text (UTF-8/Latin-1). / "
                f"Le fichier contient des octets binaires nuls."
            )
        sample = raw_bytes[:4096]
        non_text = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
        if len(sample) > 0 and (non_text / len(sample)) > 0.05:
            raise UnsupportedFileTypeError(
                f"File '{filename}' contains excessive binary control bytes and is not valid plain text."
            )

    return expected_fmt
