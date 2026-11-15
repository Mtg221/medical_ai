"""OCR quality metrics (CER, WER, word-box coverage) and digital-vs-scanned benchmarking."""

from __future__ import annotations

import difflib
import io
import shutil
import subprocess
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
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


def tesseract_available(required_langs: tuple[str, ...] = ("eng", "fra")) -> tuple[bool, str]:
    """Check whether Tesseract binary and required language packs are installed."""
    exe = shutil.which("tesseract")
    if not exe:
        return False, "Tesseract OCR binary not found in PATH."
    try:
        proc = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, check=False, timeout=10)
    except Exception as exc:
        return False, f"Failed to invoke tesseract --list-langs: {exc}"
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    installed = {line.strip() for line in output.splitlines() if line.strip() and not line.startswith("List of")}
    missing = [lang for lang in required_langs if lang not in installed]
    if missing:
        return False, f"Missing Tesseract language pack(s): {', '.join(missing)}"
    return True, "ok"


_FONT_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ("/System/Library/Fonts/Supplemental/Times New Roman.ttf", "/System/Library/Fonts/Times.ttc", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
    ("/System/Library/Fonts/Supplemental/Courier New.ttf", "/System/Library/Fonts/Courier.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"),
)


def _load_font(font_index: int, size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    group = _FONT_CANDIDATES[font_index % len(_FONT_CANDIDATES)]
    for candidate in group + tuple(p for g in _FONT_CANDIDATES for p in g):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_noisy_scan_image(
    text: str,
    *,
    width: int | None = None,
    height: int | None = None,
    dpi: int = 200,
    skew_deg: float = 2.0,
    noise_std: float = 8.0,
    blur_radius: float = 0.0,
    jpeg_quality: int | None = None,
    contrast: float = 1.0,
    perspective: float = 0.0,
    uneven_lighting: float = 0.0,
    stamps_stains: bool = False,
    two_columns: bool = False,
    table_mode: bool = False,
    font_index: int = 0,
    wrap_chars: int = 72,
    seed: int = 42,
) -> Image.Image:
    """Render clinical text onto a realistic degraded scan image supporting DPI, perspective, lighting, stamps, 2-col, tables, and fonts."""
    scale = max(0.5, dpi / 200.0)
    w = width if width is not None else int(round(1654 * scale))
    h = height if height is not None else int(round(2339 * scale))
    font_size = max(16, int(round(28 * scale)))
    line_step = max(24, int(round(40 * scale)))
    margin = int(round(80 * scale))

    img = Image.new("RGB", (w, h), color=(252, 251, 248))
    draw = ImageDraw.Draw(img)
    font = _load_font(font_index, font_size)

    raw_lines = text.splitlines()
    if two_columns and len(raw_lines) > 4:
        col_w = max(28, wrap_chars // 2 - 2)
        mid = (len(raw_lines) + 1) // 2
        for col_idx, col_lines in enumerate((raw_lines[:mid], raw_lines[mid:])):
            x_off = margin if col_idx == 0 else (w // 2 + margin // 2)
            y = margin
            for rl in col_lines:
                wrapped = textwrap.wrap(rl, width=col_w) if rl.strip() else [""]
                for line in wrapped:
                    if y > h - margin:
                        break
                    draw.text((x_off, y), line, fill=(20, 20, 24), font=font)
                    y += line_step
    elif table_mode:
        y = margin
        for rl in raw_lines:
            wrapped = textwrap.wrap(rl, width=wrap_chars) if rl.strip() else [""]
            for line in wrapped:
                if y > h - margin:
                    break
                if ":" in line or "|" in line:
                    draw.rectangle([margin - 8, y - 4, w - margin, y + line_step - 4], outline=(90, 90, 95), width=1)
                draw.text((margin, y), line, fill=(20, 20, 24), font=font)
                y += line_step
    else:
        y = margin
        for rl in raw_lines:
            wrapped = textwrap.wrap(rl, width=wrap_chars) if rl.strip() else [""]
            for line in wrapped:
                if y > h - margin:
                    break
                draw.text((margin, y), line, fill=(20, 20, 24), font=font)
                y += line_step

    if stamps_stains:
        overlay = img.copy()
        odraw = ImageDraw.Draw(overlay)
        odraw.ellipse([int(w * 0.65), int(h * 0.12), int(w * 0.88), int(h * 0.24)], fill=(230, 212, 185))
        odraw.rectangle([int(w * 0.62), int(h * 0.06), int(w * 0.90), int(h * 0.11)], outline=(180, 45, 45), width=3)
        odraw.text((int(w * 0.64), int(h * 0.072)), "CONFIDENTIAL", fill=(180, 45, 45), font=font)
        img = Image.blend(img, overlay, alpha=0.28)

    if abs(skew_deg) > 0.01:
        img = img.rotate(skew_deg, resample=Image.Resampling.BICUBIC, fillcolor=(252, 251, 248))

    if perspective > 0.0:
        dx = perspective * w * 0.06
        dy = perspective * h * 0.03
        coeffs = _perspective_coeffs(
            [(0, 0), (w, 0), (w, h), (0, h)],
            [(dx, dy), (w - dx * 0.6, 0.0), (w, h - dy), (dx * 0.4, h)],
        )
        img = img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, resample=Image.Resampling.BICUBIC, fillcolor=(248, 246, 240))

    if uneven_lighting > 0.0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grad = 1.0 - uneven_lighting * (0.35 * (xx / max(1, w)) + 0.25 * (yy / max(1, h)))
        arr_l = np.clip(np.array(img, dtype=np.float32) * grad[:, :, None], 0, 255).astype(np.uint8)
        img = Image.fromarray(arr_l)

    if blur_radius > 0.0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if abs(contrast - 1.0) > 0.01:
        img = ImageEnhance.Contrast(img).enhance(contrast)

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        arr = np.array(img, dtype=np.float32)
        arr += rng.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    if jpeg_quality is not None:
        qbuf = io.BytesIO()
        img.save(qbuf, format="JPEG", quality=int(jpeg_quality))
        qbuf.seek(0)
        img = Image.open(qbuf).convert("RGB")

    return img


def _perspective_coeffs(
    src_pts: list[tuple[float, float]], dst_pts: list[tuple[float, float]]
) -> tuple[float, ...]:
    """Compute 8-parameter perspective transform coefficients for PIL Image.transform."""
    matrix = []
    for p1, p2 in zip(dst_pts, src_pts):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(src_pts, dtype=np.float64).reshape(8)
    res = np.linalg.solve(a, b)
    return tuple(float(x) for x in res)


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
    dpi: int = 200,
    skew_deg: float = 2.0,
    noise_std: float = 8.0,
    blur_radius: float = 0.0,
    jpeg_quality: int | None = None,
    contrast: float = 1.0,
    perspective: float = 0.0,
    uneven_lighting: float = 0.0,
    stamps_stains: bool = False,
    two_columns: bool = False,
    table_mode: bool = False,
    font_index: int = 0,
    as_pdf: bool = True,
) -> OCRBenchmarkRow:
    """Render a synthetic scan (PNG or image-only PDF), parse it, and score CER/WER."""
    img = render_noisy_scan_image(
        reference_text,
        dpi=dpi,
        skew_deg=skew_deg,
        noise_std=noise_std,
        blur_radius=blur_radius,
        jpeg_quality=jpeg_quality,
        contrast=contrast,
        perspective=perspective,
        uneven_lighting=uneven_lighting,
        stamps_stains=stamps_stains,
        two_columns=two_columns,
        table_mode=table_mode,
        font_index=font_index,
    )
    if as_pdf:
        raw_bytes = build_scanned_pdf_bytes([img])
        bundle = parse_document_bytes(raw_bytes, "PDF")
    else:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        bundle = parse_document_bytes(buf.getvalue(), "PNG")
    return evaluate_bundle_against_reference(doc_id, language, reference_text, bundle)


def map_ocr_spans_to_reference(
    ocr_text: str,
    reference_text: str,
    spans: Sequence[tuple[int, int]],
) -> list[tuple[int, int] | None]:
    """Map character spans [start, end) from OCR text back to reference_text offsets via token alignment."""
    import re

    ref_toks = [(m.group(0).lower(), m.start(), m.end()) for m in re.finditer(r"\S+", reference_text)]
    ocr_toks = [(m.group(0).lower(), m.start(), m.end()) for m in re.finditer(r"\S+", ocr_text)]
    if not ref_toks or not ocr_toks:
        return [None for _ in spans]

    sm = difflib.SequenceMatcher(
        a=[t[0] for t in ocr_toks],
        b=[t[0] for t in ref_toks],
        autojunk=False,
    )
    ocr_to_ref: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ocr_to_ref[i1 + k] = j1 + k
        elif tag == "replace":
            span_len = min(i2 - i1, j2 - j1)
            for k in range(span_len):
                ocr_to_ref[i1 + k] = j1 + k

    mapped: list[tuple[int, int] | None] = []
    for s, e in spans:
        covered = [idx for idx, (_, ts, te) in enumerate(ocr_toks) if te > s and ts < e]
        ref_indices = [ocr_to_ref[i] for i in covered if i in ocr_to_ref]
        if not ref_indices:
            mapped.append(None)
            continue
        r_start = ref_toks[min(ref_indices)][1]
        r_end = ref_toks[max(ref_indices)][2]
        mapped.append((r_start, r_end))
    return mapped
