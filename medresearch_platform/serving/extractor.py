"""Production serving service backed by saved evaluated models (Hybrid v2 / Classical CRF)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import DocumentPrediction
from medresearch_platform.extraction.contract import (
    AssertionStatus,
    ExtractedRecord,
    ExtractedRelationship,
    ExtractionResult,
    NegationStatus,
    RegionCoordinates,
    ReviewStatus,
)
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.hybrid.stage4_temporal import resolve_document_date

_ASSERTION_TO_CONTRACT: dict[str, AssertionStatus] = {
    "present": "present",
    "absent": "absent",
    "possible": "uncertain",
    "hypothetical": "hypothetical",
    "conditional": "uncertain",
    "family": "family_history",
}

_ASSERTION_TO_NEGATION: dict[str, NegationStatus] = {
    "present": "affirmed",
    "absent": "negated",
    "possible": "uncertain",
    "hypothetical": "uncertain",
    "conditional": "uncertain",
    "family": "affirmed",
}


_NON_CLINICAL_BY_TYPE: dict[str, frozenset[str]] = {
    "LAB_TEST": frozenset(
        {"vs", "dates", "date", "ocr", "illegible", "t5h", "reference", "reference <", "ref", "mmhg", "bpm", "ng/l", "mg/dl"}
    ),
    "PROCEDURE": frozenset({"ocr", "illegible", "reflexes"}),
    "SYMPTOM": frozenset({"pharmacy record", "alert", "oriented", "oriented x3", "positional", "taking", "dry"}),
    "FREQUENCY": frozenset({"old", "child"}),
    "MEDICATION": frozenset({"says", "taking", "reported", "noted", "prescribed", "air"}),
    "DIAGNOSIS": frozenset({"as", "in", "of", "to", "for", "and", "or"}),
}
_LAB_NAMES_NOT_PROCEDURE: frozenset[str] = frozenset(
    {
        "hb",
        "hgb",
        "hemoglobin",
        "hba1c",
        "wbc",
        "glucose",
        "platelets",
        "tsh",
        "b12",
        "creatinine",
        "creat",
        "urea",
        "ldl",
        "crp",
        "d-dimer",
        "bnp",
        "troponin",
        "hs-tni",
        "spo2",
        "bp",
        "hr",
        "rr",
        "temp",
        "na",
        "k",
    }
)
_FREQUENCY_NOT_ROUTE: frozenset[str] = frozenset({"prn", "od", "qd", "bid", "tid", "qid", "qhs", "daily", "nightly"})
_DIAGNOSIS_NOT_SYMPTOM: frozenset[str] = frozenset({"diabetes", "type 2 diabetes", "hypertension", "diabète"})
_RECORD_ID_RE = re.compile(r"^[A-Z]{2,10}[-_]\d{1,8}$")
_PURE_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_AGE_LIKE_RE = re.compile(r"\b\d+\s*(?:years?|yrs?|y\.?o\.?|y\b|ans?)\b|\b\d+\?\s*/\s*\d+", re.IGNORECASE)
_ROOM_AIR_PREFIX_RE = re.compile(r"(?:%|spo2|o2|room\s+air)\s*$", re.IGNORECASE)
_PAIN_SCORE_RE = re.compile(r"^\d{1,2}/10$")
_PAIN_PREFIX_RE = re.compile(r"(?:pain|cp|douleur|eva|score|→|->)\s*[:\-]?\s*$", re.IGNORECASE)
_TRUNCATED_CONTRACTION_RE = re.compile(r"^[a-z]\s+[a-z]")
_UNIT_ONLY_RE = re.compile(r"^(?:m[1il]u/l|mg/dl|g/dl|mmol/l|x10\^9/l|bpm|mmhg|%|[^\w\d]+)$", re.IGNORECASE)
_OCR_CORRUPTED_VAL_RE = re.compile(r"\?|[0-9][OBA]|[OBA][0-9]")
_OCR_ALPHA_NOISE_RE = re.compile(r"[A-Za-z][013@][A-Za-z]|[A-Za-z]{2,}[013@]\b|\b[013@][A-Za-z]{2,}")
_STRUCTURED_VITAL_LAB_RE = re.compile(
    r"(?:^|\n|\||;|,\s*)\s*(?:[-•*]\s*)?"
    r"(?:(?P<time>\d{1,2}:\d{2})\s*[—–\-:]\s*)?"
    r"(?P<name>"
    r"BP(?:\s+repeat)?|HR(?:\s+repeat)?|Pulse|RR|Temp(?:erature|érature)?|SpO2(?:\s+repeat)?|"
    r"Weight|Wt|Poids|Height|Ht|Taille|BMI|Tension(?:\s+art[ée]rielle)?|"
    r"hs-TnI|Troponin(?:\s+[IT])?|TnI|HbA1c|Hb|WBC|Platelets|Plaquettes|Na|Sodium|K|Potassium|"
    r"Creatinine|Creat|Cr[ée]atinine|Urea|Ur[ée]e|Glucose|LDL|CRP|D-dimer|BNP|B12|TSH"
    r")"
    r"(?:\s*(?:\|\s*|\t+|:\s*|(?:\bde\b|\bà\b)\s+|\s+)|(?=\d))"
    r"(?P<val>(?:\d[0-9OBA?./]{0,11})(?:\s*/\s*[0-9OBA?]{1,4})?)"
    r"(?:\s*(?P<unit>mmHg|bpm|/min|°C|C|%|kg|cm|m\b|g/dL|mg/dL|mmol/L|µmol/L|mg/L|ng/L|ng/mL|pg/mL|mIU/L|[×x]10\^9/L))?",
    re.IGNORECASE,
)
_REF_RANGE_RE = re.compile(
    r"^\s*\((?:ref(?:erence)?|norm(?:al|ale)?|N)\s*[:=]?\s*(?P<spec>[^)]{1,40})\)",
    re.IGNORECASE,
)
_OXYGEN_COND_RE = re.compile(
    r"^\s*(?:on\s+)?(?P<cond>room\s+air|RA\b|air\s+ambiant|AA\b|O2\s*\d+(?:\.\d+)?\s*L(?:/min)?|nasal\s+cannula|lunettes(?:\s+\d+\s*L)?)",
    re.IGNORECASE,
)
_MED_ORDERED_NOT_ADMIN_RE = re.compile(
    r"\b(?:ordered\s+(?:but\s+)?not\s+(?:yet\s+)?administered|ordered,?\s*not\s+given|not\s+administered|non\s+administr[ée])\b",
    re.IGNORECASE,
)
_MED_NOT_TAKING_RE = re.compile(
    r"\b(?:not\s+taking|no\s+active\s+prescription|denies\s+taking|ne\s+prend\s+pas)\b",
    re.IGNORECASE,
)
_MED_DISCONTINUED_RE = re.compile(
    r"\b(?:stopped|discontinued|arr[êe]t[ée]?|interrompu)\b",
    re.IGNORECASE,
)
_MED_PRN_RE = re.compile(r"\b(?:prn|as\s+needed|si\s+besoin)\b", re.IGNORECASE)
_MED_LAST_TAKEN_PAST_RE = re.compile(
    r"\b(?:last\s+taken\s+(?:yesterday|last\s+night|\d+\s+(?:days?|hours?)\s+ago)|last\s+dose|pris\s+hier)\b",
    re.IGNORECASE,
)
_MED_UNCERTAIN_RE = re.compile(
    r"\b(?:active\?|current\?|when\s+I\s+remember|sometimes\s+forgets|verify)\b|\?",
    re.IGNORECASE,
)
_FREQ_NORM_MAP: dict[str, str] = {
    "od": "once daily",
    "qd": "once daily",
    "daily": "once daily",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "qhs": "at bedtime",
    "nightly": "at bedtime",
    "prn": "as needed",
}
_FUNCTIONAL_TAIL_RELS: frozenset[str] = frozenset(
    {"HAS_DOSAGE", "HAS_FREQUENCY", "HAS_ROUTE", "HAS_VALUE", "HAS_UNIT"}
)


def _normalize_ocr_clinical_token(raw_text: str, entity_type: str) -> tuple[str | None, bool]:
    """Return (normalized_candidate, is_ocr_corrupted) for noisy OCR tokens without fabricating digits for '?'."""
    txt = raw_text.strip()
    if not txt:
        return None, False
    if entity_type in {"LAB_VALUE", "DOSAGE"}:
        if not _OCR_CORRUPTED_VAL_RE.search(txt):
            return None, False
        if "?" in txt:
            return None, True
        cand = re.sub(r"(?<=\d)[Oo]|[Oo](?=\d)", "0", txt)
        cand = re.sub(r"(?<=\d)B|B(?=\d)", "8", cand)
        cand = re.sub(r"(?<=\d)A|A(?=\d)", "4", cand)
        return (cand if cand != txt else None), True
    if _OCR_ALPHA_NOISE_RE.search(txt) or "?" in txt:
        if "?" in txt:
            return None, True
        cand = txt.replace("@", "a")
        cand = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])|(?<=[A-Za-z]{2})0\b", "o", cand)
        cand = re.sub(r"(?<=[A-Za-z])3(?=[A-Za-z])", "e", cand)
        cand = re.sub(r"(?<=[A-Za-z])1(?=[A-Za-z])", "i", cand)
        return (cand if cand != txt else None), True
    return None, False


def _interpret_reference_range(val_txt: str, ref_spec: str) -> str:
    """Interpret a numeric lab value against a parsed reference range specification."""
    if _OCR_CORRUPTED_VAL_RE.search(val_txt):
        return "uncertain"
    m_num = re.search(r"-?\d+(?:\.\d+)?", val_txt)
    if not m_num:
        return "uncertain"
    val = float(m_num.group(0))
    m_lt = re.search(r"<\s*=?\s*(\d+(?:\.\d+)?)", ref_spec)
    if m_lt:
        hi = float(m_lt.group(1))
        return "within_range" if val <= hi else "high"
    m_gt = re.search(r">\s*=?\s*(\d+(?:\.\d+)?)", ref_spec)
    if m_gt:
        lo = float(m_gt.group(1))
        return "within_range" if val >= lo else "low"
    m_rng = re.search(r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)", ref_spec)
    if m_rng:
        lo, hi = float(m_rng.group(1)), float(m_rng.group(2))
        if val < lo:
            return "low"
        if val > hi:
            return "high"
        return "within_range"
    return "uncertain"


def _extract_structured_vitals_and_labs(
    text: str,
    existing_entities: list[Any],
    meta_by_span: dict[tuple[int, int], dict[str, Any]] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Deterministically extract EN/FR tabular/bulleted vital & lab measurements without adding any ML model."""
    from medresearch_platform.eval_harness.metrics import PredictedEntity, PredictedRelation

    new_ents: list[Any] = []
    new_rels: list[Any] = []
    occupied: list[tuple[int, int]] = [(e.start, e.end) for e in existing_entities]

    def _overlaps(s: int, e: int) -> bool:
        return any(max(s, os_) < min(e, oe_) for os_, oe_ in occupied)

    for m in _STRUCTURED_VITAL_LAB_RE.finditer(text):
        ns, ne = m.span("name")
        vs, ve = m.span("val")
        val_txt = m.group("val").strip()
        if not val_txt or not any(ch.isdigit() for ch in val_txt):
            continue
        while ve > vs and text[ve - 1] in ".,;":
            ve -= 1
        val_txt = text[vs:ve]
        if not val_txt:
            continue

        head_ent = next(
            (e for e in existing_entities if e.type == "LAB_TEST" and max(e.start, ns) < min(e.end, ne)),
            None,
        )
        if head_ent is None:
            if _overlaps(ns, ne):
                continue
            head_ent = PredictedEntity(
                type="LAB_TEST",
                start=ns,
                end=ne,
                text=text[ns:ne],
                assertion="present",
                confidence=0.92,
                needs_review=False,
            )
            new_ents.append(head_ent)
            occupied.append((ns, ne))

        is_ocr_bad = bool(_OCR_CORRUPTED_VAL_RE.search(val_txt))
        val_ent = next(
            (e for e in existing_entities if e.type == "LAB_VALUE" and max(e.start, vs) < min(e.end, ve)),
            None,
        )
        if val_ent is None and not _overlaps(vs, ve):
            val_ent = PredictedEntity(
                type="LAB_VALUE",
                start=vs,
                end=ve,
                text=val_txt,
                assertion="possible" if is_ocr_bad else "present",
                confidence=0.62 if is_ocr_bad else 0.92,
                needs_review=is_ocr_bad,
                high_uncertainty=is_ocr_bad,
                certainty="possible" if is_ocr_bad else "confirmed",
            )
            new_ents.append(val_ent)
            occupied.append((vs, ve))

        if val_ent is not None:
            new_rels.append(
                PredictedRelation(
                    type="HAS_VALUE",
                    head_start=head_ent.start,
                    head_end=head_ent.end,
                    tail_start=val_ent.start,
                    tail_end=val_ent.end,
                    confidence=0.92,
                )
            )

        tail_pos = ve
        if m.group("unit"):
            us, ue = m.span("unit")
            tail_pos = max(tail_pos, ue)
            unit_ent = next(
                (e for e in existing_entities if e.type == "LAB_UNIT" and max(e.start, us) < min(e.end, ue)),
                None,
            )
            if unit_ent is None and not _overlaps(us, ue):
                unit_ent = PredictedEntity(
                    type="LAB_UNIT",
                    start=us,
                    end=ue,
                    text=text[us:ue],
                    assertion="present",
                    confidence=0.92,
                    needs_review=False,
                )
                new_ents.append(unit_ent)
                occupied.append((us, ue))
            if unit_ent is not None:
                new_rels.append(
                    PredictedRelation(
                        type="HAS_UNIT",
                        head_start=head_ent.start,
                        head_end=head_ent.end,
                        tail_start=unit_ent.start,
                        tail_end=unit_ent.end,
                        confidence=0.92,
                    )
                )

        if meta_by_span is not None:
            meta: dict[str, Any] = {}
            if m.group("time"):
                meta["measurement_time"] = m.group("time")
            after = text[tail_pos : min(len(text), tail_pos + 60)]
            m_ref = _REF_RANGE_RE.match(after)
            if m_ref:
                spec = m_ref.group("spec").strip()
                meta["reference_range"] = spec
                meta["interpretation"] = _interpret_reference_range(val_txt, spec)
            if head_ent.text.lower().startswith("spo2"):
                m_o2 = _OXYGEN_COND_RE.match(after)
                if m_o2:
                    raw_c = m_o2.group("cond").strip()
                    meta["oxygen_condition"] = (
                        "room_air" if raw_c.lower() in {"room air", "ra", "air ambiant", "aa"} else raw_c
                    )
            if meta:
                meta_by_span[(head_ent.start, head_ent.end)] = meta
                if val_ent is not None:
                    meta_by_span[(val_ent.start, val_ent.end)] = dict(meta)

    return new_ents, new_rels


_NKDA_RE = re.compile(r"\b(?:nkda|no\s+known\s+(?:drug\s+)?allergies|aucune\s+allergie\s+connue)\b", re.IGNORECASE)
_ALLERGY_POS_RE = re.compile(
    r"\b(?:allerg(?:y|ic|ies|ique)\s*(?:to|:|-|—)?\s*(?!(?:nkda|none|no\b|nil|unknown|aucune|pas\s+d)\b)(?:penicillin|latex|aspirin|amoxicillin|sulfa|iodine|peanuts?|shellfish|contrast|[a-z]{4,})|"
    r"(?:penicillin|latex|amoxicillin|aspirin)\s*(?:-|—|:)\s*(?:rash|itching|hives|anaphylaxis|urticaria))\b",
    re.IGNORECASE,
)


def _canon_clinical_concept(raw: str) -> str:
    low = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
    aliases = {
        "htn": "hypertension",
        "t2dm": "type 2 diabetes",
        "dm2": "type 2 diabetes",
        "diabetes": "type 2 diabetes",
        "type 2 diabetes": "type 2 diabetes",
        "sob": "shortness of breath",
        "cp": "chest pain",
        "temp": "temperature",
        "temperature": "temperature",
        "bp": "blood pressure",
        "hr": "heart rate",
        "pulse": "heart rate",
        "spo2": "spo2",
    }
    return aliases.get(low, low)


def detect_clinical_contradictions(
    records: list[ExtractedRecord] | tuple[ExtractedRecord, ...],
    text: str,
) -> list[dict[str, Any]]:
    """Detect explicit clinical contradictions in a document while preserving both sides and spans."""
    contradictions: list[dict[str, Any]] = []

    # 1. NKDA vs affirmed allergy mention
    m_nkda = _NKDA_RE.search(text)
    m_allergy = _ALLERGY_POS_RE.search(text)
    if m_nkda and m_allergy:
        contradictions.append(
            {
                "topic": "allergy_status",
                "status": "unresolved",
                "side_a": {"text": m_nkda.group(0), "span": [m_nkda.start(), m_nkda.end()], "assertion": "absent"},
                "side_b": {"text": m_allergy.group(0), "span": [m_allergy.start(), m_allergy.end()], "assertion": "present"},
            }
        )

    # 2. Same patient diagnosis/symptom affirmed AND negated
    by_concept: dict[str, list[ExtractedRecord]] = {}
    for rec in records:
        if rec.entity_type in {"DIAGNOSIS", "SYMPTOM"} and rec.experiencer == "patient":
            canon = _canon_clinical_concept(rec.original_text)
            if len(canon) >= 3:
                by_concept.setdefault(canon, []).append(rec)
    for canon, recs in by_concept.items():
        pos = [r for r in recs if r.assertion_status == "present"]
        neg = [r for r in recs if r.assertion_status == "absent"]
        if pos and neg:
            contradictions.append(
                {
                    "topic": f"polarity:{canon}",
                    "status": "unresolved",
                    "side_a": {"text": pos[0].original_text, "span": [pos[0].char_start, pos[0].char_end], "assertion": "present"},
                    "side_b": {"text": neg[0].original_text, "span": [neg[0].char_start, neg[0].char_end], "assertion": "absent"},
                }
            )

    # 3. Conflicting doses for the same medication (including inline "X 10 mg ... 5 mg" clauses)
    med_doses: dict[str, list[tuple[str, int, int]]] = {}
    for rec in records:
        if rec.entity_type != "MEDICATION":
            continue
        med_key = _canon_clinical_concept(rec.original_text)
        for rel in rec.relationships:
            if rel.relation_type == "HAS_DOSAGE":
                med_doses.setdefault(med_key, []).append((rel.target_value.strip().lower(), rec.char_start, rec.char_end))
        # Also scan same line for alternative dose mentions (e.g., "10 mg daily; patient says 5 mg")
        line_end = text.find("\n", rec.char_end)
        line_txt = text[rec.char_start : (line_end if line_end != -1 else len(text))]
        for dm in re.finditer(r"\b(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?))\b", line_txt, re.IGNORECASE):
            d_norm = re.sub(r"\s+", " ", dm.group(1).strip().lower())
            med_doses.setdefault(med_key, []).append(
                (d_norm, rec.char_start + dm.start(), rec.char_start + dm.end())
            )
    for med_key, items in med_doses.items():
        distinct: dict[str, tuple[int, int]] = {}
        for d_val, s, e in items:
            if d_val not in distinct:
                distinct[d_val] = (s, e)
        if len(distinct) >= 2:
            vals = list(distinct.items())
            contradictions.append(
                {
                    "topic": f"medication_dose:{med_key}",
                    "status": "unresolved",
                    "side_a": {"text": vals[0][0], "span": list(vals[0][1])},
                    "side_b": {"text": vals[1][0], "span": list(vals[1][1])},
                }
            )

    # 4. Unperformed scan vs negative imaging statement
    for para in text.split("\n\n"):
        if re.search(r"\b(?:study\s+not\s+performed|not\s+performed|cancelled|canceled|non\s+r[ée]alis[ée])\b", para, re.IGNORECASE):
            m_neg = re.search(r"\b(?:no\s+evidence\s+of\s+[A-Za-z0-9 \-]{2,30}|normal\s+study)\b", para, re.IGNORECASE)
            if m_neg:
                p_off = text.find(para)
                contradictions.append(
                    {
                        "topic": "imaging_unperformed_vs_negative",
                        "status": "unresolved",
                        "side_a": {"text": m_neg.group(0), "span": [p_off + m_neg.start(), p_off + m_neg.end()]},
                        "side_b": {"text": "study_not_performed", "span": [p_off, p_off + len(para)]},
                    }
                )

    return contradictions


def _retype_entity(ent: Any, new_type: str) -> Any:
    from medresearch_platform.eval_harness.metrics import PredictedEntity

    return PredictedEntity(
        type=new_type,
        start=ent.start,
        end=ent.end,
        text=ent.text,
        assertion=ent.assertion,
        confidence=ent.confidence,
        needs_review=ent.needs_review,
        confidence_variance=getattr(ent, "confidence_variance", 0.0),
        high_uncertainty=getattr(ent, "high_uncertainty", False),
        certainty=getattr(ent, "certainty", "confirmed"),
        temporality=getattr(ent, "temporality", "current"),
        experiencer=getattr(ent, "experiencer", "patient"),
    )


def _filter_non_clinical_artifacts(entities: list[Any], text: str) -> list[Any]:
    """Remove generalizable OCR/template/non-clinical false positives and fix obvious clinical misclassifications."""
    out: list[Any] = []
    for ent in entities:
        raw = str(ent.text).strip()
        low = raw.lower()
        etype = str(ent.type)

        if low in _NON_CLINICAL_BY_TYPE.get(etype, frozenset()):
            continue
        # Truncated contraction spans like "t catch breath" (from "can't catch breath")
        if etype == "SYMPTOM" and _TRUNCATED_CONTRACTION_RE.match(raw) and ent.start > 0 and text[ent.start - 1] in "'’":
            continue
        # Room Air ("SpO2 98% RA", "94% RA") is NOT Rheumatoid Arthritis
        if etype == "DIAGNOSIS" and raw == "RA":
            prefix = text[max(0, ent.start - 18) : ent.start]
            if _ROOM_AIR_PREFIX_RE.search(prefix):
                continue
        # "mass" inside "mass effect" is a radiology finding, not a patient symptom
        if etype == "SYMPTOM" and low == "mass":
            post = text[ent.end : min(len(text), ent.end + 12)].lower()
            if post.startswith(" effect"):
                continue
        # Pain scores ("7/10", "3/10", "6/10") are NOT DATE
        if etype == "DATE" and _PAIN_SCORE_RE.match(raw):
            prefix = text[max(0, ent.start - 30) : ent.start]
            if _PAIN_PREFIX_RE.search(prefix):
                continue
        # Record IDs (NOISE-0027, SYN-00481, ER-2), ages (4? / 47 years, 6O y), and vital prefixes (SpO2 9) are NOT DATE
        if etype == "DATE" and (
            _RECORD_ID_RE.match(raw)
            or _AGE_LIKE_RE.search(raw)
            or re.match(r"^(?:spo2|bp|hr|rr|temp|tni)\b", raw, re.IGNORECASE)
        ):
            continue
        # Multi-line spans in LAB_UNIT or LAB_VALUE are malformed OCR/CRF bleed-through
        if etype in {"LAB_UNIT", "LAB_VALUE"} and "\n" in raw:
            continue
        # 4-digit calendar years (2021) are NOT DOSAGE
        if etype == "DOSAGE" and _PURE_YEAR_RE.match(raw):
            continue
        # Pure unit strings or punctuation noise ("+? /", "m1U/L") are NOT LAB_VALUE
        if etype == "LAB_VALUE" and (_UNIT_ONLY_RE.match(raw) or not any(ch.isdigit() for ch in raw)):
            continue
        # Core lab/vital analytes misclassified as PROCEDURE -> retype to LAB_TEST
        if etype == "PROCEDURE" and low in _LAB_NAMES_NOT_PROCEDURE:
            ent = _retype_entity(ent, "LAB_TEST")
        # Medication frequencies (PRN, OD, BID, etc.) misclassified as ROUTE -> retype to FREQUENCY
        elif etype == "ROUTE" and low in _FREQUENCY_NOT_ROUTE:
            ent = _retype_entity(ent, "FREQUENCY")
        # Diagnoses (diabetes, hypertension) misclassified as SYMPTOM -> retype to DIAGNOSIS
        elif etype == "SYMPTOM" and low in _DIAGNOSIS_NOT_SYMPTOM:
            ent = _retype_entity(ent, "DIAGNOSIS")
        out.append(ent)
    return out


def _merge_and_dedup_relations(
    entities: list[Any],
    rel_lists: list[list[Any]],
) -> list[Any]:
    """Merge relation lists, keep only relations whose endpoints exist, and deduplicate functional modifier tails."""
    valid_spans = {(e.start, e.end) for e in entities}
    seen_exact: set[tuple[str, int, int, int, int]] = set()
    non_func: list[Any] = []
    best_func: dict[tuple[str, int, int], tuple[int, int, float, Any]] = {}

    for rlist in rel_lists:
        for r in rlist:
            if (r.head_start, r.head_end) not in valid_spans or (r.tail_start, r.tail_end) not in valid_spans:
                continue
            key = (r.type, r.head_start, r.head_end, r.tail_start, r.tail_end)
            if key in seen_exact:
                continue
            seen_exact.add(key)
            if r.type in _FUNCTIONAL_TAIL_RELS:
                tkey = (r.type, r.tail_start, r.tail_end)
                is_prec = 0 if r.head_end <= r.tail_start else 1
                gap = abs(r.tail_start - r.head_end) if r.head_end <= r.tail_start else abs(r.head_start - r.tail_end)
                rank = (is_prec, gap, -float(r.confidence))
                prev = best_func.get(tkey)
                if prev is None or rank < (prev[0], prev[1], prev[2]):
                    best_func[tkey] = (rank[0], rank[1], rank[2], r)
            else:
                non_func.append(r)

    return non_func + [entry[3] for entry in best_func.values()]


def _resolve_page_and_bbox(
    char_start: int,
    char_end: int,
    page_map: list[dict[str, Any]] | None,
) -> tuple[int, RegionCoordinates | None]:
    if not page_map:
        return 1, None
    for page in page_map:
        p_num = int(page.get("page_number", 1))
        p_start = int(page.get("char_start", 0))
        p_end = int(page.get("char_end", 0))
        if p_start <= char_start < max(p_end, p_start + 1):
            word_boxes = page.get("word_boxes") or []
            overlapping = []
            for wb in word_boxes:
                if isinstance(wb, dict):
                    wb_start = int(wb.get("char_start", -1))
                    wb_end = int(wb.get("char_end", -1))
                    x0 = float(wb["x0"]) if wb.get("x0") is not None else float(wb.get("x") or 0.0)
                    y0 = float(wb["y0"]) if wb.get("y0") is not None else float(wb.get("y") or 0.0)
                    x1 = float(wb["x1"]) if wb.get("x1") is not None else x0 + float(wb.get("width") or 0.0)
                    y1 = float(wb["y1"]) if wb.get("y1") is not None else y0 + float(wb.get("height") or 0.0)
                else:
                    wb_start = int(getattr(wb, "char_start", -1))
                    wb_end = int(getattr(wb, "char_end", -1))
                    x0 = float(getattr(wb, "x", 0.0))
                    y0 = float(getattr(wb, "y", 0.0))
                    x1 = x0 + float(getattr(wb, "width", 0.0))
                    y1 = y0 + float(getattr(wb, "height", 0.0))
                if max(wb_start, char_start) < min(wb_end, char_end):
                    overlapping.append((x0, y0, x1, y1))
            if overlapping:
                return p_num, RegionCoordinates(
                    x0=round(min(b[0] for b in overlapping), 4),
                    y0=round(min(b[1] for b in overlapping), 4),
                    x1=round(max(b[2] for b in overlapping), 4),
                    y1=round(max(b[3] for b in overlapping), 4),
                )
            return p_num, None
    return int(page_map[0].get("page_number", 1)), None


class ExtractionService:
    """Loads saved evaluated models from disk and serves ExtractionResult records."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        classical_crf_dir: Path | str | None = None,
    ) -> None:
        default_dir = Path(os.environ.get("MEDRESEARCH_MODEL_DIR", "models/hybrid_v2_seed1"))
        self.model_dir = Path(model_dir) if model_dir is not None else default_dir
        self.classical_crf_dir = (
            Path(classical_crf_dir)
            if classical_crf_dir is not None
            else Path(os.environ.get("MEDRESEARCH_CLASSICAL_CRF_DIR", "models/classical_crf_seed1"))
        )
        raw_torch_env = os.environ.get("MEDRESEARCH_TORCH_NER_DIR", "").strip()
        if raw_torch_env:
            self.torch_ner_dir = Path(raw_torch_env.split(",")[0].strip())
        elif model_dir is not None:
            # Keep torch NER discovery scoped to the caller-supplied model directory root
            self.torch_ner_dir = self.model_dir.parent / "hospital_char_cnn_bilstm_crf_seed1"
        else:
            self.torch_ner_dir = Path("models/hospital_char_cnn_bilstm_crf_seed1")
        self._hybrid: HybridPipeline | None = None
        self._classical_crf: ClassicalCRFModel | None = None
        self._torch_ner_models: list[Any] | None = None

    def _discover_torch_ner_dirs(self) -> list[Path]:
        raw_env = os.environ.get("MEDRESEARCH_TORCH_NER_DIR", "").strip()
        if "," in raw_env:
            return [Path(p.strip()) for p in raw_env.split(",") if p.strip() and Path(p.strip()).exists()]
        if raw_env:
            p = Path(raw_env)
            return [p] if p.exists() else []
        # Auto-discover multi-seed hospital checkpoints (seed1, seed2, seed3, ...)
        base_parent = self.torch_ner_dir.parent
        discovered = sorted(
            d for d in base_parent.glob("hospital_char_cnn_bilstm_crf_seed*")
            if d.is_dir() and (d / "manifest.json").exists()
        )
        if discovered:
            return discovered
        return [self.torch_ner_dir] if self.torch_ner_dir.exists() else []

    def _get_torch_ner_models(self) -> list[Any]:
        if self._torch_ner_models is None:
            dirs = self._discover_torch_ner_dirs()
            if not dirs:
                self._torch_ner_models = []
            else:
                from medresearch_platform.approaches.torch_ner import (
                    PyTorchCharCNNBiLSTMCRF,
                    is_torch_available,
                )

                if is_torch_available():
                    self._torch_ner_models = [PyTorchCharCNNBiLSTMCRF.load(d, device="cpu") for d in dirs]
                else:
                    self._torch_ner_models = []
        return self._torch_ner_models

    def _get_torch_ner(self) -> Any:
        models = self._get_torch_ner_models()
        return models[0] if models else None

    def _get_hybrid(self) -> HybridPipeline:
        if self._hybrid is None:
            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"Saved model not found at '{self.model_dir}'. "
                    f"Train and save it first with: medextract train --config configs/exp/v3_2/hybrid_v2.yaml "
                    f"--seed 1 --out {self.model_dir}"
                )
            self._hybrid = HybridPipeline.load(self.model_dir)
        return self._hybrid

    def _get_classical_crf(self) -> ClassicalCRFModel:
        if self._classical_crf is None:
            if not self.classical_crf_dir.exists():
                raise FileNotFoundError(
                    f"Saved classical_crf model not found at '{self.classical_crf_dir}'. "
                    f"Train and save it first with: medextract train --config configs/exp/v3_1/classical_crf.yaml "
                    f"--seed 1 --out {self.classical_crf_dir}"
                )
            self._classical_crf = ClassicalCRFModel.load(self.classical_crf_dir)
        return self._classical_crf

    @staticmethod
    def get_review_threshold(cfg: dict[str, Any] | None = None) -> float:
        env_val = os.environ.get("MEDRESEARCH_REVIEW_THRESHOLD")
        if env_val is not None and env_val.strip() != "":
            return float(env_val)
        if cfg and cfg.get("review_threshold") is not None:
            return float(cfg["review_threshold"])
        return 0.60

    def list_loaded_models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.model_dir.exists():
            h = self._get_hybrid()
            cfg = h.manifest.get("cfg", h.cfg)
            out.append({
                "model_id": "hybrid_v2",
                "name": "hybrid_v2",
                "model_type": h.manifest.get("model_type", "hybrid"),
                "seed": h.manifest.get("seed", h.seed),
                "cfg": cfg,
                "review_threshold": self.get_review_threshold(cfg if isinstance(cfg, dict) else None),
                "corpus_id": h.manifest.get("corpus_id", h.corpus_id),
                "corpus_hash": h.manifest.get("corpus_hash", h.corpus_hash),
                "git_commit": h.manifest.get("git_commit", "unknown"),
                "saved_at_utc": h.manifest.get("saved_at_utc", ""),
                "files_sha256": h.manifest.get("files", {}),
                "model_dir": str(self.model_dir),
            })
        if self.classical_crf_dir.exists():
            c = self._get_classical_crf()
            cfg = c.manifest.get("cfg", c.cfg)
            out.append({
                "model_id": "classical_crf",
                "name": "classical_crf",
                "model_type": c.manifest.get("model_type", "classical_crf"),
                "seed": c.manifest.get("seed", c.seed),
                "cfg": cfg,
                "review_threshold": self.get_review_threshold(cfg if isinstance(cfg, dict) else None),
                "corpus_id": c.manifest.get("corpus_id", c.corpus_id),
                "corpus_hash": c.manifest.get("corpus_hash", c.corpus_hash),
                "git_commit": c.manifest.get("git_commit", "unknown"),
                "saved_at_utc": c.manifest.get("saved_at_utc", ""),
                "files_sha256": c.manifest.get("files", {}),
                "model_dir": str(self.classical_crf_dir),
            })
        t_models = self._get_torch_ner_models()
        if t_models:
            t_ner = t_models[0]
            out.append({
                "model_id": "hospital_char_cnn_bilstm_crf",
                "name": "hospital_char_cnn_bilstm_crf",
                "model_type": "pytorch_char_cnn_bilstm_crf",
                "seed": t_ner.seed,
                "num_seeds": len(t_models),
                "cfg": {
                    "pretrained_weights_used": False,
                    "partial_label_marginal_crf": True,
                    "ensemble_seeds": [m.seed for m in t_models],
                },
                "review_threshold": self.get_review_threshold(None),
                "corpus_id": "SYNTH-MED-V3.1+open_clinical",
                "corpus_hash": t_ner.trained_weights_sha256[:16],
                "git_commit": "local",
                "saved_at_utc": "",
                "files_sha256": {"char_cnn_bilstm_crf.pt": t_ner.trained_weights_sha256},
                "model_dir": str(self.torch_ner_dir),
            })
        return out

    def extract(
        self,
        text: str,
        language: str = "en",
        document_date: str | None = None,
        page_map: list[dict[str, Any]] | None = None,
        *,
        document_id: str = "doc_serving",
        source_record_id: str = "rec_serving",
        approach: str = "hybrid_v2",
    ) -> ExtractionResult:
        doc_date, _doc_date_source = resolve_document_date(text, fallback_date=document_date)
        doc = AnnotatedDocument(
            doc_id=document_id,
            patient_id="PAT-SERVING",
            language=language if language in ("en", "fr") else "en",
            doc_type="discharge_summary",
            document_date=doc_date,
            source=document_id,
            text=text,
        )

        source_stage_by_key: dict[tuple[str, int, int], str] = {}
        eff_approach = os.environ.get("MEDRESEARCH_DEFAULT_APPROACH", approach) if approach == "hybrid_v2" else approach
        t_models = self._get_torch_ner_models() if eff_approach == "hospital_ensemble" else []
        if eff_approach == "hospital_ensemble" and t_models:
            from medresearch_platform.approaches.context_negex import apply_context_and_relations
            from medresearch_platform.approaches.torch_ner import ensemble_torch_ner_predictions
            from medresearch_platform.eval_harness.metrics import PredictedEntity

            model = self._get_hybrid()
            pred_h, payload = model.extract_document(doc)
            torch_ents = ensemble_torch_ner_predictions(t_models, doc.text)
            merged: dict[tuple[str, int, int], PredictedEntity] = {}
            for e in torch_ents:
                merged[(e.type, e.start, e.end)] = e
                source_stage_by_key[(e.type, e.start, e.end)] = "torch_char_cnn_bilstm_crf"
            for he in pred_h.entities:
                key = (he.type, he.start, he.end)
                if key in merged:
                    prev = merged[key]
                    merged[key] = PredictedEntity(
                        type=prev.type,
                        start=prev.start,
                        end=prev.end,
                        text=prev.text,
                        assertion=he.assertion,
                        confidence=round(min(0.99, max(prev.confidence, he.confidence) + 0.05), 4),
                        needs_review=prev.needs_review,
                        confidence_variance=prev.confidence_variance,
                        high_uncertainty=prev.high_uncertainty,
                    )
                    source_stage_by_key[key] = "hospital_ensemble(torch+hybrid_v2)"
                else:
                    overlaps = [k for k, m in merged.items() if max(he.start, m.start) < min(he.end, m.end)]
                    if not overlaps and he.confidence >= 0.72:
                        merged[key] = he
                        source_stage_by_key[key] = "hybrid_v2"
                    elif overlaps and all(he.confidence > merged[k].confidence + 0.12 for k in overlaps):
                        for k in overlaps:
                            del merged[k]
                        merged[key] = he
                        source_stage_by_key[key] = "hybrid_v2"
            meta_by_span: dict[tuple[int, int], dict[str, Any]] = {}
            ens_ents = _filter_non_clinical_artifacts(sorted(merged.values(), key=lambda x: x.start), doc.text)
            struct_ents, struct_rels = _extract_structured_vitals_and_labs(doc.text, ens_ents, meta_by_span=meta_by_span)
            combined_ents = sorted(ens_ents + struct_ents, key=lambda x: x.start)
            enriched_ents, ens_rels, _ = apply_context_and_relations(doc, combined_ents)
            merged_rels = _merge_and_dedup_relations(enriched_ents, [pred_h.relations, struct_rels, ens_rels])
            pred = DocumentPrediction(
                doc_id=doc.doc_id,
                language=doc.language,
                doc_type=doc.doc_type,
                predicted_doc_type=pred_h.predicted_doc_type,
                entities=enriched_ents,
                relations=merged_rels,
                timex=pred_h.timex,
            )
            manifest = model.manifest
            model_name = "hospital_ensemble"
            model_version = f"seed{model.seed}+torch_partial_crf"
            warnings = tuple(payload.get("validation", {}).get("warnings", []))
        elif eff_approach == "hybrid_v2" or approach == "hybrid_v2":
            meta_by_span = {}
            model = self._get_hybrid()
            pred_raw, payload = model.extract_document(doc)
            filtered_ents = _filter_non_clinical_artifacts(pred_raw.entities, doc.text)
            struct_ents, struct_rels = _extract_structured_vitals_and_labs(
                doc.text, filtered_ents, meta_by_span=meta_by_span
            )
            combined_ents = sorted(filtered_ents + struct_ents, key=lambda x: x.start)
            filtered_rels = _merge_and_dedup_relations(combined_ents, [pred_raw.relations, struct_rels])
            pred = DocumentPrediction(
                doc_id=pred_raw.doc_id,
                language=pred_raw.language,
                doc_type=pred_raw.doc_type,
                predicted_doc_type=pred_raw.predicted_doc_type,
                entities=combined_ents,
                relations=filtered_rels,
                timex=pred_raw.timex,
            )
            for item in payload.get("records", []):
                prov = item.get("provenance", {})
                key = (str(item.get("entity_type")), int(item.get("char_start", 0)), int(item.get("char_end", 0)))
                source_stage_by_key[key] = str(prov.get("source_stage", "hybrid_v2"))
            manifest = model.manifest
            model_name = "hybrid_v2"
            model_version = f"seed{model.seed}@{manifest.get('git_commit', 'unknown')}"
            warnings = tuple(payload.get("validation", {}).get("warnings", []))
        elif approach == "classical_crf":
            meta_by_span = {}
            crf_model = self._get_classical_crf()
            pred = crf_model.predict_document(doc)
            manifest = crf_model.manifest
            model_name = "classical_crf"
            model_version = f"seed{crf_model.seed}@{manifest.get('git_commit', 'unknown')}"
            warnings = ()
        else:
            raise ValueError(f"Unsupported serving approach: {approach!r}")

        return self._prediction_to_result(
            pred=pred,
            text=text,
            page_map=page_map,
            document_id=document_id,
            source_record_id=source_record_id,
            model_name=model_name,
            model_version=model_version,
            manifest=manifest,
            source_stage_by_key=source_stage_by_key,
            warnings=warnings,
            meta_by_span=meta_by_span,
        )

    def _prediction_to_result(
        self,
        *,
        pred: DocumentPrediction,
        text: str,
        page_map: list[dict[str, Any]] | None,
        document_id: str,
        source_record_id: str,
        model_name: str,
        model_version: str,
        manifest: dict[str, Any],
        source_stage_by_key: dict[tuple[str, int, int], str],
        warnings: tuple[str, ...],
        meta_by_span: dict[tuple[int, int], dict[str, Any]] | None = None,
    ) -> ExtractionResult:
        from medresearch_platform.approaches.context_negex import classify_assertion_attributes

        ent_by_span = {(e.start, e.end): e for e in pred.entities}
        rels_by_head: dict[tuple[int, int], list[ExtractedRelationship]] = {}
        for rel in pred.relations:
            tail = ent_by_span.get((rel.tail_start, rel.tail_end))
            tail_text = tail.text if tail is not None else text[rel.tail_start : rel.tail_end]
            tail_type = tail.type if tail is not None else "UNKNOWN"
            rels_by_head.setdefault((rel.head_start, rel.head_end), []).append(
                ExtractedRelationship(
                    relation_type=rel.type,
                    target_value=tail_text,
                    target_entity_type=tail_type,
                    confidence=float(rel.confidence),
                    evidence_text=text[min(rel.head_start, rel.tail_start) : max(rel.head_end, rel.tail_end)][:160],
                )
            )

        cfg_dict = manifest.get("cfg") if isinstance(manifest.get("cfg"), dict) else None
        review_threshold = self.get_review_threshold(cfg_dict)
        span_meta_map = meta_by_span or {}

        records: list[ExtractedRecord] = []
        for idx, ent in enumerate(pred.entities):
            page_num, bbox = _resolve_page_and_bbox(ent.start, ent.end, page_map)
            matched_timex = next(
                (t.normalized for t in pred.timex if max(t.start, ent.start) < min(t.end, ent.end)),
                None,
            )
            stage_name = source_stage_by_key.get((ent.type, ent.start, ent.end), model_name)
            ctx_attrs = classify_assertion_attributes(text, ent.start, ent.end, language=pred.language)
            assertion_contract = _ASSERTION_TO_CONTRACT.get(ent.assertion, "present")
            negation_contract = _ASSERTION_TO_NEGATION.get(ent.assertion, "affirmed")
            ent_conf = float(ent.confidence)
            ent_var = round(float(getattr(ent, "confidence_variance", 0.0)), 4)
            is_high_unc = bool(getattr(ent, "high_uncertainty", False))
            ocr_cand, is_ocr_corrupted = _normalize_ocr_clinical_token(ent.text, ent.type)
            if is_ocr_corrupted:
                is_high_unc = True
                ent_conf = min(ent_conf, 0.65)
                assertion_contract = "uncertain"
                negation_contract = "uncertain"

            # P2: Medication status & normalized frequency enrichment
            med_meta: dict[str, Any] = {}
            if ent.type == "MEDICATION":
                line_s = text.rfind("\n", 0, ent.start)
                line_e = text.find("\n", ent.end)
                med_ctx = text[(line_s + 1 if line_s != -1 else 0) : (line_e if line_e != -1 else len(text))]
                if _MED_ORDERED_NOT_ADMIN_RE.search(med_ctx):
                    med_meta["medication_status"] = "ordered_not_administered"
                    assertion_contract = "uncertain"
                    negation_contract = "uncertain"
                elif _MED_NOT_TAKING_RE.search(med_ctx):
                    med_meta["medication_status"] = "not_taking"
                    assertion_contract = "absent"
                    negation_contract = "negated"
                elif _MED_DISCONTINUED_RE.search(med_ctx):
                    med_meta["medication_status"] = "discontinued"
                elif _MED_PRN_RE.search(med_ctx):
                    med_meta["medication_status"] = "prn_active"
                    if _MED_LAST_TAKEN_PAST_RE.search(med_ctx):
                        med_meta["last_administered_temporality"] = "historical"
                elif _MED_UNCERTAIN_RE.search(med_ctx):
                    med_meta["medication_status"] = "uncertain"
                else:
                    med_meta["medication_status"] = "active"
                for rel in rels_by_head.get((ent.start, ent.end), []):
                    if rel.relation_type == "HAS_FREQUENCY":
                        norm_f = _FREQ_NORM_MAP.get(rel.target_value.strip().lower())
                        if norm_f:
                            med_meta["normalized_frequency"] = norm_f
            elif ent.type == "FREQUENCY":
                norm_f = _FREQ_NORM_MAP.get(ent.text.strip().lower())
                if norm_f:
                    med_meta["normalized_frequency"] = norm_f

            is_low_conf = ent_conf < review_threshold
            needs_rev = is_low_conf or is_high_unc or is_ocr_corrupted
            record_review_status: ReviewStatus = "needs_review" if needs_rev else "unreviewed"
            reasons: list[str] = []
            if is_low_conf:
                reasons.append("low_confidence")
            if is_high_unc:
                reasons.append("high_ensemble_uncertainty")
            if is_ocr_corrupted:
                reasons.append("ocr_corrupted")
            review_reason = ",".join(reasons)

            raw_cert = getattr(ent, "certainty", ctx_attrs.certainty)
            raw_temp = getattr(ent, "temporality", ctx_attrs.temporality)
            raw_exp = getattr(ent, "experiencer", ctx_attrs.experiencer)
            ent_certainty: Literal["confirmed", "possible", "absent"] = (
                "possible"
                if (is_ocr_corrupted or med_meta.get("medication_status") == "ordered_not_administered")
                else ("absent" if raw_cert == "absent" else ("possible" if raw_cert == "possible" else "confirmed"))
            )
            ent_temporality: Literal["current", "historical"] = (
                "historical"
                if (raw_temp == "historical" or med_meta.get("medication_status") == "discontinued")
                else "current"
            )
            ent_experiencer: Literal["patient", "family"] = (
                "family" if raw_exp == "family" else "patient"
            )
            extra_span_meta = span_meta_map.get((ent.start, ent.end), {})
            prov_dict: dict[str, Any] = {
                "source_stage": stage_name,
                "raw_assertion": ent.assertion,
                "section": ctx_attrs.section,
                "experiencer_detail": ctx_attrs.experiencer_detail,
                "assertion_detail": ctx_attrs.assertion_detail,
                "temporality_detail": ctx_attrs.temporality_detail,
                "predicted_doc_type": pred.predicted_doc_type,
                "review_reason": review_reason,
                "review_threshold": review_threshold,
                "confidence_variance": ent_var,
                "high_uncertainty": is_high_unc,
                "ocr_corrupted": is_ocr_corrupted,
                "corpus_id": manifest.get("corpus_id", "unknown"),
                "corpus_hash": manifest.get("corpus_hash", ""),
                "git_commit": manifest.get("git_commit", "unknown"),
                "files_sha256": manifest.get("files", {}),
                "relations": [
                    {
                        "type": r.type,
                        "head_start": r.head_start,
                        "head_end": r.head_end,
                        "tail_start": r.tail_start,
                        "tail_end": r.tail_end,
                        "confidence": r.confidence,
                    }
                    for r in pred.relations
                    if r.head_start == ent.start and r.head_end == ent.end
                ],
                "timex": [
                    {"type": t.type, "start": t.start, "end": t.end, "text": t.text, "normalized": t.normalized}
                    for t in pred.timex
                ],
            }
            if ocr_cand is not None:
                prov_dict["normalized_candidate"] = ocr_cand
            if extra_span_meta:
                prov_dict.update(extra_span_meta)
            if med_meta:
                prov_dict.update(med_meta)

            records.append(
                ExtractedRecord(
                    source_record_id=f"{source_record_id}_{idx}",
                    document_id=document_id,
                    original_text=ent.text,
                    extracted_value=ent.text,
                    entity_type=ent.type,
                    normalized_concept=matched_timex or ocr_cand,
                    char_start=ent.start,
                    char_end=ent.end,
                    page_number=page_num,
                    region_coordinates=bbox,
                    language=pred.language,
                    negation_status=negation_contract,
                    assertion_status=assertion_contract,
                    uncertainty=round(max(0.0, 1.0 - ent_conf), 4),
                    temporal_information=matched_timex,
                    relationships=tuple(rels_by_head.get((ent.start, ent.end), [])),
                    confidence_score=round(ent_conf, 4),
                    model_name=model_name,
                    model_version=model_version,
                    review_status=record_review_status,
                    confidence_variance=ent_var,
                    high_uncertainty=is_high_unc,
                    certainty=ent_certainty,
                    temporality=ent_temporality,
                    experiencer=ent_experiencer,
                    provenance=prov_dict,
                )
            )

        contradictions = detect_clinical_contradictions(records, text)
        all_warnings = list(warnings)
        for c in contradictions:
            all_warnings.append(
                f"contradiction:{c['topic']}:{c['side_a']['text']} vs {c['side_b']['text']}"
            )

        return ExtractionResult(
            source_record_id=source_record_id,
            document_id=document_id,
            language=pred.language,
            model_name=model_name,
            model_version=model_version,
            records=tuple(records),
            modality_status={"text": "evaluated_model", "ocr": "tesseract_layout"},
            warnings=tuple(all_warnings),
        )
