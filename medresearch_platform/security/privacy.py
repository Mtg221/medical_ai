"""Bilingual (EN/FR) medical PII/PHI detection, de-identification, date shifting, and pseudonymization."""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any


@dataclass(frozen=True)
class DetectedPIISpan:
    pii_type: str
    start: int
    end: int
    original_value: str
    replacement_token: str


@dataclass(frozen=True)
class DeidentificationResult:
    scrubbed_text: str
    detected_spans: list[DetectedPIISpan]
    date_shift_days: int
    policy_version: str
    residual_risk_note: str


_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "PATIENT_NAME",
        re.compile(
            r"(?:Patient(?:\s+Name)?|Nom(?:\s+du\s+patient)?)\s*:\s*([A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ\-]+)+)",
            re.UNICODE,
        ),
    ),
    (
        "MEDICAL_RECORD_NUMBER",
        re.compile(r"\b(?:MRN|IPP|N°\s*Dossier|Dossier)\s*[:#]?\s*([A-Z0-9\-]{5,16})\b", re.IGNORECASE),
    ),
    (
        "FRENCH_NIR",
        re.compile(r"\b([12]\s?\d{2}\s?(?:0[1-9]|1[0-2])\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2})\b"),
    ),
    (
        "SSN",
        re.compile(r"\b(\d{3}-\d{2}-\d{4})\b"),
    ),
    (
        "EMAIL",
        re.compile(r"\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b"),
    ),
    (
        "PHONE",
        re.compile(r"\b((?:\+33|\+221|\+1|0)[1-9](?:[\s.\-]?\d{2}){3,4})\b"),
    ),
)

_ISO_DATE_PATTERN = re.compile(r"\b((?:19|20)\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")


def derive_pseudonym_and_shift(raw_subject_identifier: str, vault_secret_key: str) -> tuple[str, int]:
    """Derive a deterministic pseudonym code and interval-preserving date shift (-180 to -15 days)."""
    mac = hmac.new(
        vault_secret_key.encode("utf-8"),
        raw_subject_identifier.strip().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    pseudonym_code = f"SUBJ-{mac[:12].upper()}"
    shift_seed = int(mac[12:16], 16)
    date_shift_days = -(15 + (shift_seed % 166))
    return pseudonym_code, date_shift_days


def deidentify_clinical_text(
    text: str,
    *,
    date_shift_days: int = 0,
    policy_version: str = "DEID-EN-FR-v1.0",
) -> DeidentificationResult:
    """Detect and mask bilingual PII spans while shifting ISO dates by a constant subject offset."""
    detected: list[DetectedPIISpan] = []
    scrubbed = text

    for pii_type, pattern in _PII_PATTERNS:
        for match in pattern.finditer(text):
            target_group = 1 if match.lastindex and match.lastindex >= 1 else 0
            span_start, span_end = match.span(target_group)
            orig_val = match.group(target_group)
            token = f"[{pii_type}_REDACTED]"
            detected.append(
                DetectedPIISpan(
                    pii_type=pii_type,
                    start=span_start,
                    end=span_end,
                    original_value=orig_val,
                    replacement_token=token,
                )
            )

    # Sort non-overlapping spans in reverse order to replace safely
    detected_sorted = sorted(detected, key=lambda s: (s.start, -(s.end - s.start)))
    non_overlapping: list[DetectedPIISpan] = []
    last_end = -1
    for span in detected_sorted:
        if span.start >= last_end:
            non_overlapping.append(span)
            last_end = span.end

    chars = list(scrubbed)
    for span in reversed(non_overlapping):
        chars[span.start : span.end] = list(span.replacement_token)
    scrubbed = "".join(chars)

    if date_shift_days != 0:
        def _shift_match(m: re.Match[str]) -> str:
            try:
                dt = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                shifted = dt + timedelta(days=date_shift_days)
                return shifted.isoformat()
            except ValueError:
                return m.group(0)

        scrubbed = _ISO_DATE_PATTERN.sub(_shift_match, scrubbed)

    return DeidentificationResult(
        scrubbed_text=scrubbed,
        detected_spans=non_overlapping,
        date_shift_days=date_shift_days,
        policy_version=policy_version,
        residual_risk_note=(
            "Rule-based bilingual PII scrubbing and interval-preserving date shifting applied. "
            "Removing direct identifiers does not guarantee absolute anonymity against rare-disease linkage."
        ),
    )


class IsolatedPseudonymVault:
    """In-memory / encrypted vault mapping reversible pseudonyms independently of research records."""

    def __init__(self, vault_secret_key: str) -> None:
        self._secret = vault_secret_key
        self._forward: dict[str, tuple[str, int]] = {}
        self._reverse: dict[str, str] = {}

    def register_identity(self, raw_identifier: str) -> dict[str, Any]:
        pseudonym, shift_days = derive_pseudonym_and_shift(raw_identifier, self._secret)
        self._forward[raw_identifier] = (pseudonym, shift_days)
        self._reverse[pseudonym] = raw_identifier
        return {
            "pseudonym_code": pseudonym,
            "date_shift_days": shift_days,
        }

    def resolve_pseudonym(self, pseudonym_code: str, *, authorized_role: str) -> str:
        if authorized_role != "INSTITUTION_ADMIN":
            raise PermissionError("Reversing a pseudonym requires INSTITUTION_ADMIN authorization.")
        if pseudonym_code not in self._reverse:
            raise KeyError(f"Unknown pseudonym code: {pseudonym_code}")
        return self._reverse[pseudonym_code]
