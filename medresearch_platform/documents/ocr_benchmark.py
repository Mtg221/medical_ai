"""OCR quality metrics (CER, WER, word-box coverage) and digital-vs-scanned benchmarking."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from medresearch_platform.documents.parsers import ParsedDocumentBundle, parse_document_bytes


def _levenshtein(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    """Compute exact edit distance between two sequences."""
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == seq_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[m]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (CER) on whitespace-normalized strings."""
    ref_norm = " ".join(reference.split())
    hyp_norm = " ".join(hypothesis.split())
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    dist = _levenshtein(list(ref_norm), list(hyp_norm))
    return round(dist / len(ref_norm), 4)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate (WER) on whitespace-split tokens."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = _levenshtein(ref_words, hyp_words)
    return round(dist / len(ref_words), 4)


def render_noisy_scan_image(
    text: str,
    *,
    width: int = 1654,
    height: int = 2339,
    skew_deg: float = 2.0,
    noise_std: float = 8.0,
    seed: int = 42,
) -> Image.Image:
    """Render clinical text onto a realistic skewed, lightly noisy 200-DPI-like scan image."""
    img = Image.new("RGB", (width, height), color=(252, 251, 248))
    draw = ImageDraw.Draw(img)

    font: ImageFont.ImageFont | ImageFont.FreeTypeFont
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, size=34)
            break
        except Exception:
            continue
    else:
        font = ImageFont.load_default()

    y = 110
    for line in text.splitlines():
        if y > height - 100:
            break
        draw.text((100, y), line, fill=(20, 20, 24), font=font)
        y += 50

    if abs(skew_deg) > 0.01:
        img = img.rotate(skew_deg, resample=Image.Resampling.BICUBIC, fillcolor=(252, 251, 248))

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        arr = np.array(img, dtype=np.float32)
        arr += rng.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def build_scanned_pdf_bytes(images: list[Image.Image]) -> bytes:
    """Wrap one or more rendered scan images into an image-only PDF (no text layer)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    page_w, page_h = letter
    for img in images:
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")
        img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), 0, 0, width=page_w, height=page_h)
        c.showPage()
    c.save()
    return buf.getvalue()


@dataclass(frozen=True)
class OCRBenchmarkRow:
    """Single-document OCR benchmark result."""

    doc_id: str
    language: str
    extraction_mode: str
    cer: float
    wer: float
    mean_confidence: float
    word_box_count: int


def evaluate_bundle_against_reference(
    doc_id: str,
    language: str,
    reference_text: str,
    bundle: ParsedDocumentBundle,
) -> OCRBenchmarkRow:
    """Measure CER, WER, confidence, and word-box count for a parsed bundle."""
    hyp = "\n".join(p.raw_text for p in bundle.pages)
    confs = [p.mean_confidence for p in bundle.pages if p.raw_text.strip()]
    mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
    total_boxes = sum(len(p.word_boxes) for p in bundle.pages)
    return OCRBenchmarkRow(
        doc_id=doc_id,
        language=language,
        extraction_mode=bundle.extraction_mode,
        cer=compute_cer(reference_text, hyp),
        wer=compute_wer(reference_text, hyp),
        mean_confidence=round(mean_conf, 4),
        word_box_count=total_boxes,
    )


def build_digital_pdf_with_table(title: str, headers: list[str], rows: list[list[str]]) -> bytes:
    """Generate a digital PDF containing a bordered table for table-extraction tests."""
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Heading2"]), Spacer(1, 12)]
    tbl = Table([headers] + rows)
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(tbl)
    doc.build(story)
    return buf.getvalue()


def parse_and_benchmark_scan(
    doc_id: str,
    language: str,
    reference_text: str,
    *,
    skew_deg: float = 2.0,
    noise_std: float = 8.0,
    as_pdf: bool = True,
) -> OCRBenchmarkRow:
    """Render a synthetic scan (PNG or image-only PDF), parse it, and score CER/WER."""
    img = render_noisy_scan_image(reference_text, skew_deg=skew_deg, noise_std=noise_std)
    if as_pdf:
        raw_bytes = build_scanned_pdf_bytes([img])
        bundle = parse_document_bytes(raw_bytes, "PDF")
    else:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        bundle = parse_document_bytes(buf.getvalue(), "PNG")
    return evaluate_bundle_against_reference(doc_id, language, reference_text, bundle)
