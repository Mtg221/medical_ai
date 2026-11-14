"""Untrusted document validation, archive-bomb protection, and prompt-injection defenses (Sections 2, 24)."""

from __future__ import annotations

from dataclasses import dataclass
import io
import re
import zipfile

SUPPORTED_FORMATS = frozenset({"TXT", "PDF", "DOCX", "PNG", "JPEG", "TIFF", "WEBP"})

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "IGNORE_INSTRUCTIONS",
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions", re.IGNORECASE),
    ),
    (
        "IGNORER_CONSIGNES_FR",
        re.compile(r"ignore[rz]?\s+(?:toutes\s+les\s+)?(?:instructions|consignes)\s+pr[ée]c[ée]dentes", re.IGNORECASE),
    ),
    (
        "SYSTEM_PROMPT_OVERRIDE",
        re.compile(r"(?:system\s*prompt|you\s+are\s+now\s+an?\s+unrestricted|execute\s+shell\s+command)", re.IGNORECASE),
    ),
    (
        "SCRIPT_TAG_EMBEDDED",
        re.compile(r"<\s*script\b[^>]*>", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class DocumentSafetyAssessment:
    is_safe_to_parse: bool
    detected_format: str
    byte_size: int
    untrusted_flags: list[str]
    rejection_reason: str | None = None


def inspect_untrusted_bytes(
    raw_bytes: bytes,
    filename: str,
    *,
    max_upload_bytes: int = 25 * 1024 * 1024,
    max_zip_expansion_ratio: float = 40.0,
) -> DocumentSafetyAssessment:
    """Validate file size, magic bytes, zip-bomb ratios (for DOCX), and untrusted instruction patterns."""
    byte_size = len(raw_bytes)
    if byte_size == 0:
        return DocumentSafetyAssessment(
            is_safe_to_parse=False,
            detected_format="UNKNOWN",
            byte_size=0,
            untrusted_flags=["EMPTY_PAYLOAD"],
            rejection_reason="Uploaded document is empty.",
        )

    if byte_size > max_upload_bytes:
        return DocumentSafetyAssessment(
            is_safe_to_parse=False,
            detected_format="UNKNOWN",
            byte_size=byte_size,
            untrusted_flags=["SIZE_LIMIT_EXCEEDED"],
            rejection_reason=f"Payload size {byte_size} exceeds limit {max_upload_bytes} bytes.",
        )

    ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else "TXT"
    if ext == "JPG":
        ext = "JPEG"
    if ext == "TIF":
        ext = "TIFF"

    flags: list[str] = []
    if raw_bytes.startswith(b"%PDF-"):
        detected_format = "PDF"
    elif raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        detected_format = "PNG"
    elif raw_bytes.startswith(b"\xff\xd8\xff"):
        detected_format = "JPEG"
    elif raw_bytes.startswith((b"II*\x00", b"MM\x00*")):
        detected_format = "TIFF"
    elif len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        detected_format = "WEBP"
    elif raw_bytes.startswith(b"PK\x03\x04"):
        detected_format = "DOCX"
        # Check for archive bomb inside DOCX zip container
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                total_uncompressed = sum(info.file_size for info in zf.infolist())
                if byte_size > 0 and (total_uncompressed / byte_size) > max_zip_expansion_ratio:
                    return DocumentSafetyAssessment(
                        is_safe_to_parse=False,
                        detected_format="DOCX",
                        byte_size=byte_size,
                        untrusted_flags=["ARCHIVE_BOMB_DETECTED"],
                        rejection_reason="DOCX archive expansion ratio exceeds safety threshold.",
                    )
        except zipfile.BadZipFile:
            return DocumentSafetyAssessment(
                is_safe_to_parse=False,
                detected_format="DOCX",
                byte_size=byte_size,
                untrusted_flags=["CORRUPTED_ZIP_CONTAINER"],
                rejection_reason="Corrupted ZIP/DOCX container.",
            )
    else:
        if ext in ("PDF", "PNG", "JPEG", "TIFF", "WEBP", "DOCX"):
            return DocumentSafetyAssessment(
                is_safe_to_parse=False,
                detected_format=ext,
                byte_size=byte_size,
                untrusted_flags=["MAGIC_BYTE_MISMATCH"],
                rejection_reason=f"File extension .{ext.lower()} does not match binary header.",
            )
        detected_format = "TXT"

    if detected_format not in SUPPORTED_FORMATS:
        return DocumentSafetyAssessment(
            is_safe_to_parse=False,
            detected_format=detected_format,
            byte_size=byte_size,
            untrusted_flags=["UNSUPPORTED_FORMAT"],
            rejection_reason=f"Unsupported file format: {detected_format}",
        )

    return DocumentSafetyAssessment(
        is_safe_to_parse=True,
        detected_format=detected_format,
        byte_size=byte_size,
        untrusted_flags=flags,
        rejection_reason=None,
    )


def scan_text_for_untrusted_instructions(text: str) -> list[str]:
    """Flag adversarial prompt-injection or embedded script strings inside untrusted document text.
    Note: The platform treats all document text strictly as inert data, never executable instructions.
    """
    flags: list[str] = []
    for flag_name, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            flags.append(f"UNTRUSTED_CONTENT_FLAG:{flag_name}")
    return flags
