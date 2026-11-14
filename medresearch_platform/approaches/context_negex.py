"""Bilingual (EN/FR) ConText/NegEx assertion classifier, TIMEX extractor, and proximity relation linker."""

from __future__ import annotations

import re

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import (
    PredictedEntity,
    PredictedRelation,
    PredictedTimex,
)

# Language-parameterized ConText/NegEx cue tables
CONTEXT_CUES: dict[str, dict[str, tuple[str, ...]]] = {
    "en": {
        "pseudo_neg": ("no increase", "no change", "not ruled out"),
        "neg_pre": ("no ", "denies ", "without ", "negative for ", "absence of ", "no evidence of "),
        "neg_post": (" ruled out", " excluded", " negative", " absent"),
        "possible": ("suspected ", "possible ", "probable ", "likely ", "question of "),
        "hypothetical": ("if ", "return if ", "should ", "in case of "),
        "family": ("family history of ", "mother had ", "father had "),
        "historical": ("history of ", "past history of ", "prior ", "previous "),
        "terminators": (".", ";", "\n", " but ", " however ", " except "),
    },
    "fr": {
        "pseudo_neg": ("sans changement", "pas d'augmentation", "non éliminé"),
        "neg_pre": ("absence de ", "absence d'", "sans ", "pas de ", "pas d'", "nie ", "négatif pour "),
        "neg_post": (" éliminé", " écarté", " écartée", " exclu", " exclue"),
        "possible": ("suspicion de ", "suspicion d'", "probable ", "possible ", "évoquant "),
        "hypothetical": ("si ", "si apparition de ", "en cas de "),
        "family": ("antécédents familiaux de ", "antécédents familiaux d'"),
        "historical": ("antécédent de ", "antécédent d'", "antécédents de ", "ancien "),
        "terminators": (".", ";", "\n", " mais ", " cependant ", " toutefois "),
    },
}


def classify_assertion_context(text: str, start: int, end: int, language: str = "en") -> str:
    """ConText/NegEx-style assertion classification with scope termination and pseudo-negation."""
    cues = CONTEXT_CUES.get(language, CONTEXT_CUES["en"])
    # Restrict scope to the current sentence/clause
    pre_window = text[max(0, start - 90) : start].lower()
    post_window = text[end : min(len(text), end + 45)].lower()

    for term in cues["terminators"]:
        if term in pre_window:
            pre_window = pre_window.rsplit(term, 1)[-1]
        if term in post_window:
            post_window = post_window.split(term, 1)[0]

    full_clause = pre_window + " " + text[start:end].lower() + " " + post_window
    if any(p in full_clause for p in cues["pseudo_neg"]):
        return "present"
    if any(c in pre_window for c in cues["family"]):
        return "family"
    if any(c in pre_window for c in cues["hypothetical"]):
        return "hypothetical"
    if any(c in pre_window for c in cues["neg_pre"]) or any(c in post_window for c in cues["neg_post"]):
        return "absent"
    if any(c in pre_window for c in cues["possible"]):
        return "possible"
    if any(c in pre_window for c in cues["historical"]):
        return "historical"
    return "present"


def apply_context_and_relations(
    doc: AnnotatedDocument,
    entities: list[PredictedEntity],
) -> tuple[list[PredictedEntity], list[PredictedRelation], list[PredictedTimex]]:
    """Apply ConText assertion, extract proximity relations, and normalize simple TIMEX expressions."""
    enriched: list[PredictedEntity] = [
        PredictedEntity(
            type=e.type,
            start=e.start,
            end=e.end,
            text=e.text,
            assertion=classify_assertion_context(doc.text, e.start, e.end, language=doc.language),
            confidence=e.confidence,
        )
        for e in entities
    ]

    relations: list[PredictedRelation] = []
    # Same-line / short-window pairing for standard medical relations
    for i, head in enumerate(enriched):
        for j, tail in enumerate(enriched):
            if i == j or abs(head.start - tail.start) > 140:
                continue
            between = doc.text[min(head.end, tail.end) : max(head.start, tail.start)]
            if "\n" in between:
                continue
            rtype: str | None = None
            if head.type == "MEDICATION" and tail.type == "DOSAGE" and head.end <= tail.start:
                rtype = "HAS_DOSAGE"
            elif head.type == "MEDICATION" and tail.type == "FREQUENCY" and head.end <= tail.start:
                rtype = "HAS_FREQUENCY"
            elif head.type == "MEDICATION" and tail.type == "ROUTE" and head.end <= tail.start:
                rtype = "HAS_ROUTE"
            elif head.type == "LAB_TEST" and tail.type == "LAB_VALUE" and head.end <= tail.start and tail.start - head.end <= 8:
                rtype = "HAS_VALUE"
            elif head.type == "LAB_VALUE" and tail.type == "LAB_UNIT" and head.end <= tail.start and tail.start - head.end <= 5:
                rtype = "HAS_UNIT"
            elif head.type == "MEDICATION" and tail.type == "DIAGNOSIS" and tail.end <= head.start:
                rtype = "TREATS"
            elif head.type == "PROCEDURE" and tail.type == "DIAGNOSIS" and head.end <= tail.start:
                rtype = "REVEALS"
            elif head.type == "SYMPTOM" and tail.type == "ANATOMY" and head.end <= tail.start:
                rtype = "LOCATED_IN"
            if rtype is not None:
                relations.append(
                    PredictedRelation(
                        type=rtype,
                        head_start=head.start,
                        head_end=head.end,
                        tail_start=tail.start,
                        tail_end=tail.end,
                        confidence=min(head.confidence, tail.confidence),
                    )
                )

    timex: list[PredictedTimex] = []
    for e in enriched:
        if e.type == "DATE" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", e.text):
            timex.append(PredictedTimex(type="DATE", start=e.start, end=e.end, text=e.text, normalized=e.text))
        elif e.type == "FREQUENCY":
            timex.append(PredictedTimex(type="FREQUENCY", start=e.start, end=e.end, text=e.text))
        elif e.type == "DURATION":
            ttype = "RELATIVE" if ("ago" in e.text.lower() or "il y a" in e.text.lower()) else "DURATION"
            timex.append(PredictedTimex(type=ttype, start=e.start, end=e.end, text=e.text))

    return enriched, relations, timex
