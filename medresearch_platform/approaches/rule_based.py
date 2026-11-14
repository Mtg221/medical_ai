"""Approach 1: Versioned EN/FR Rule-Based + Regex + ConText/NegEx Extractor."""

from __future__ import annotations

import re
from typing import Any

from medresearch_platform.approaches.context_negex import apply_context_and_relations
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.data.synthetic_v2 import (
    ANATOMIES,
    DIAGNOSES,
    LAB_TESTS,
    MEDICATIONS,
    PROCEDURES,
    SYMPTOMS,
)
from medresearch_platform.eval_harness.metrics import DocumentPrediction, PredictedEntity

_REGEX_RULES: tuple[tuple[str, str, re.Pattern[str], float], ...] = (
    ("R_PATIENT_ID", "PATIENT_ID", re.compile(r"\bPAT-(?:EN|FR)-\d{4}\b"), 0.99),
    ("R_DATE_ISO", "DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), 0.98),
    ("R_DOSAGE", "DOSAGE", re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|UI)\b", re.IGNORECASE), 0.94),
    (
        "R_FREQUENCY",
        "FREQUENCY",
        re.compile(
            r"\b(?:once daily|twice daily|three times daily|every \d+ hours|at bedtime|as needed"
            r"|une fois par jour|deux fois par jour|trois fois par jour|toutes les \d+ heures|au coucher|si besoin)\b",
            re.IGNORECASE,
        ),
        0.92,
    ),
    (
        "R_ROUTE",
        "ROUTE",
        re.compile(
            r"\b(?:orally|intravenously|subcutaneously|inhaled"
            r"|par voie orale|par voie intraveineuse|par voie sous-cutanée|par inhalation)\b",
            re.IGNORECASE,
        ),
        0.92,
    ),
    (
        "R_DURATION",
        "DURATION",
        re.compile(
            r"\b(?:\d+\s+days\s+ago|for\s+\d+\s+days|il y a\s+\d+\s+jours|pendant\s+\d+\s+jours)\b",
            re.IGNORECASE,
        ),
        0.91,
    ),
    ("R_LAB_UNIT", "LAB_UNIT", re.compile(r"(?:µmol/L|mg/L|g/dL|ng/mL|mmol/L|G/L|%)"), 0.90),
    ("R_LAB_VAL", "LAB_VALUE", re.compile(r"(?<=:\s)\d+(?:\.\d+)?(?=\s*(?:µmol/L|mg/L|g/dL|ng/mL|mmol/L|G/L|%))"), 0.90),
)


def _build_lexicon_patterns() -> list[tuple[str, str, re.Pattern[str], float]]:
    entries: list[tuple[str, str, str]] = []
    for item in SYMPTOMS:
        entries.extend([("LEX_SYM", "SYMPTOM", item.en), ("LEX_SYM", "SYMPTOM", item.fr)])
    for item in DIAGNOSES:
        entries.extend([("LEX_DX", "DIAGNOSIS", item.en), ("LEX_DX", "DIAGNOSIS", item.fr)])
    for m_item, *_ in MEDICATIONS:
        entries.extend([("LEX_MED", "MEDICATION", m_item.en), ("LEX_MED", "MEDICATION", m_item.fr)])
    for l_item, *_ in LAB_TESTS:
        entries.extend([("LEX_LAB", "LAB_TEST", l_item.en), ("LEX_LAB", "LAB_TEST", l_item.fr)])
    for item in PROCEDURES:
        entries.extend([("LEX_PROC", "PROCEDURE", item.en), ("LEX_PROC", "PROCEDURE", item.fr)])
    for item in ANATOMIES:
        entries.extend([("LEX_ANAT", "ANATOMY", item.en), ("LEX_ANAT", "ANATOMY", item.fr)])

    # Longest match first, deduplicated
    seen: set[tuple[str, str]] = set()
    patterns: list[tuple[str, str, re.Pattern[str], float]] = []
    for rid, etype, phrase in sorted(entries, key=lambda x: -len(x[2])):
        key = (etype, phrase.lower())
        if key in seen:
            continue
        seen.add(key)
        pat = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
        patterns.append((f"{rid}_{len(patterns)}", etype, pat, 0.93))
    return patterns


_LEXICON_PATTERNS = _build_lexicon_patterns()


_DOC_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "discharge_summary",
        (
            "discharge summary",
            "hospital discharge",
            "hospital course",
            "discharge medications",
            "compte rendu d'hospitalisation",
            "lettre de sortie",
            "synthèse d'hospitalisation",
            "traitement de sortie",
            "bilan de sortie",
        ),
    ),
    (
        "lab_report",
        (
            "laboratory results",
            "biochemistry panel",
            "hematology & chemistry",
            "laboratory panel",
            "biologie médicale",
            "bilan biologique",
            "résultats de laboratoire",
            "résultats biologiques",
            "interprétation biologique",
        ),
    ),
    (
        "prescription",
        (
            "medical prescription",
            "medication order",
            "pharmacy prescription",
            "prescribed medications",
            "ordonnance médicale",
            "prescription médicamenteuse",
            "ordonnance de sortie",
            "traitement prescrit",
        ),
    ),
    (
        "radiology_report",
        (
            "radiology imaging",
            "diagnostic imaging",
            "radiology findings",
            "radiologist recommendation",
            "imagerie médicale",
            "rapport de radiologie",
            "interprétation radiologique",
            "résultats radiologiques",
        ),
    ),
    (
        "operative_note",
        (
            "operative procedure",
            "surgical intervention",
            "post-operative summary",
            "operating room",
            "operative findings",
            "compte rendu opératoire",
            "intervention chirurgicale",
            "protocole opératoire",
            "geste opératoire",
        ),
    ),
    (
        "consultation",
        (
            "outpatient consultation",
            "specialist consultation",
            "ambulatory clinic",
            "follow-up consultation",
            "reason for consultation",
            "compte rendu de consultation",
            "avis spécialisé",
            "consultation externe",
            "motif de consultation",
        ),
    ),
)


def _predict_doc_type_rules(text: str) -> str:
    lowered = text.lower()
    best_type = "discharge_summary"
    best_hits = 0
    for dtype, keywords in _DOC_TYPE_RULES:
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits > best_hits:
            best_hits = hits
            best_type = dtype
    return best_type


def predict_rule_based(
    train_docs: list[AnnotatedDocument],
    eval_docs: list[AnnotatedDocument],
    seed: int,
    cfg: dict[str, Any],
) -> list[DocumentPrediction]:
    """Run deterministic EN/FR rule-based + regex + ConText/NegEx extraction."""
    preds: list[DocumentPrediction] = []
    all_rules = list(_REGEX_RULES) + _LEXICON_PATTERNS

    for doc in eval_docs:
        candidates: list[tuple[int, int, str, float]] = []
        for _rule_id, etype, pat, conf in all_rules:
            for m in pat.finditer(doc.text):
                candidates.append((m.start(), m.end(), etype, conf))

        # Non-overlapping longest-first greedy selection
        candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
        accepted: list[PredictedEntity] = []
        occupied: list[tuple[int, int]] = []
        for start, end, etype, conf in candidates:
            if any(max(start, s) < min(end, e) for s, e in occupied):
                continue
            occupied.append((start, end))
            accepted.append(
                PredictedEntity(
                    type=etype,
                    start=start,
                    end=end,
                    text=doc.text[start:end],
                    assertion="present",
                    confidence=conf,
                )
            )
        accepted.sort(key=lambda e: e.start)
        enriched, relations, timex = apply_context_and_relations(doc, accepted)
        preds.append(
            DocumentPrediction(
                doc_id=doc.doc_id,
                language=doc.language,
                doc_type=doc.doc_type,
                predicted_doc_type=_predict_doc_type_rules(doc.text),
                entities=enriched,
                relations=relations,
                timex=timex,
            )
        )
    return preds
