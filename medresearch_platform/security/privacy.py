"""Bilingual (EN/FR) medical PII/PHI detection, de-identification, date shifting, and pseudonymization."""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from medresearch_platform.security.person_age_detector import (
    detect_age_spans_with_values,
    detect_person_spans,
)


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


_MONTH_NAMES = (
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|sep\.?|sept\.?|oct\.?|nov\.?|dec\.?|"
    r"janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[ûu]t|septembre|octobre|novembre|d[ée]cembre"
)

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "DOB",
        re.compile(
            rf"(?:n[ée]e?\s+le|date\s+de\s+naissance|date\s+of\s+birth|DOB|born(?:\s+on)?)\s*[:#]?\s*"
            rf"(\d{{1,2}}[/\-.]\d{{1,2}}[/\-.]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}"
            rf"|\d{{1,2}}[\s\-](?:{_MONTH_NAMES})[\s\-,]+\d{{4}}"
            rf"|(?:{_MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}})",
            re.IGNORECASE | re.UNICODE,
        ),
    ),
    (
        "MEDICAL_RECORD_NUMBER",
        re.compile(
            r"\b(?:MRN|IPP|IEP|CNI|N°\s*Dossier|Dossier\s*n°?|ID\s*patient|Patient\s*ID|Health\s*ID|"
            r"Record\s*#?|Case\s*#?|Insurance\s+(?:policy|ID)\s*#?|Policy\s*#?|"
            r"Mutuelle\s*(?:n°|#|no\.?)?|Lab\s+specimen\s+barcode|Specimen\s+(?:ID|barcode)|Barcode)"
            r"\s*[:#]?\s*([A-Z]{0,6}\-?\d[\dA-Z\-/]{3,19})\b",
            re.IGNORECASE,
        ),
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
        re.compile(
            r"(?:(?:t[ée]l(?:[ée]phone)?|phone|tel|mobile|fax|portable)\s*[:#.]?\s*(\+?[\d\s.\-()]{7,20}\d)"
            r"|\(\d{3}\)\s*\d{3}[\s.\-]\d{4}"
            r"|\b[2-9]\d{2}[\-.]\d{3}[\-.]\d{4}\b"
            r"|\+221[\s.\-]?(?:7[05678]|3[03])(?:[\s.\-]?\d{2,3}){3}"
            r"|\b(?:7[05678]|33)(?:[\s.\-]?\d{3}[\s.\-]?\d{2}[\s.\-]?\d{2}|[\s.\-]?\d{2}[\s.\-]?\d{3}[\s.\-]?\d{2})\b"
            r"|\+33[\s.\-]?(?:\(0\)[\s.\-]?)?[1-9](?:[\s.\-]?\d{2}){4}"
            r"|\b0[1-9](?:[\s.\-]?\d{2}){4}"
            r"|\+\d{1,3}[\s.\-]?(?:\(\d+\)[\s.\-]?)?(?:\d[\s.\-]?){7,12}\d)",
            re.IGNORECASE,
        ),
    ),
    (
        "ADDRESS",
        re.compile(
            r"\b("
            r"Lot\s+\d{1,5}\s+Cit[ée]\s+[A-Za-zÀ-ÖØ-öø-ÿ0-9'\-\s]{2,40}(?:,\s*[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'\-\s]{1,25})?"
            r"|\d{1,5}(?:\s*(?:bis|ter))?\s+(?:rue|avenue|av\.?|boulevard|bd\.?|blvd\.?|impasse|all[ée]e|place|chemin|route)\s+[A-Za-zÀ-ÖØ-öø-ÿ0-9'\-\s]{2,45}"
            r"(?:,\s*(?:B[âa]t\.?|Apt\.?|Esc\.?|App\.?)\s*[A-Za-z0-9\-]+)?"
            r"(?:,\s*\d{4,5}\s+[A-Za-zÀ-ÖØ-öø-ÿ\-]+|(?:,\s*[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'\-]{2,25}){1,2})?"
            r"|\d{1,5}\s+(?-i:[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ'\-]+\s+){1,4}"
            r"(?:Street|St\.|Avenue|Ave\.|Boulevard|Blvd\.?|Road|Rd\.|Lane|Ln\.|Drive|Dr\.|Way|Terrace|Ter\.|Crescent|Court|Ct\.|Place|Pl\.)\b"
            r"(?:,\s*(?:Apt\.?|Suite|Ste\.?|Unit)\s*[A-Za-z0-9\-]+)?"
            r"(?:,\s*(?-i:[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'\-\s]{1,25}),\s*(?-i:[A-Z]{2})\s+(?:\d{5}(?:-\d{4})?|(?-i:[A-Z]\d[A-Z])\s?\d(?-i:[A-Z])\d))?"
            r")",
            re.IGNORECASE | re.UNICODE,
        ),
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
    policy_version: str = "DEID-EN-FR-v1.1",
    redact_ages_over_89: bool = True,
    redact_all_ages: bool = False,
) -> DeidentificationResult:
    """Detect and mask bilingual PII spans (including HIPAA >89 age rule) while shifting ISO dates."""
    detected: list[DetectedPIISpan] = []
    scrubbed = text

    for sp in detect_person_spans(text):
        detected.append(
            DetectedPIISpan(
                pii_type=sp.pii_type,
                start=sp.start,
                end=sp.end,
                original_value=sp.text,
                replacement_token=f"[{sp.pii_type}_REDACTED]",
            )
        )

    if redact_all_ages or redact_ages_over_89:
        for a_start, a_end, age_years, raw_age in detect_age_spans_with_values(text):
            if redact_all_ages:
                detected.append(
                    DetectedPIISpan(
                        pii_type="AGE",
                        start=a_start,
                        end=a_end,
                        original_value=raw_age,
                        replacement_token="[AGE_REDACTED]",
                    )
                )
            elif redact_ages_over_89 and age_years > 89:
                detected.append(
                    DetectedPIISpan(
                        pii_type="AGE_OVER_89",
                        start=a_start,
                        end=a_end,
                        original_value=raw_age,
                        replacement_token="[AGE_OVER_89_REDACTED]",
                    )
                )

    for pii_type, pattern in _PII_PATTERNS:
        for match in pattern.finditer(text):
            if match.lastindex and match.lastindex >= 1 and match.group(1) is not None:
                target_group = 1
            else:
                target_group = 0
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
