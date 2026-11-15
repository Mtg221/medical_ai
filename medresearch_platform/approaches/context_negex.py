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
        "neg_post": (" ruled out", " excluded", " negative", " absent", " was not observed", " not observed"),
        "possible": ("suspected ", "possible ", "probable ", "likely ", "question of ", "rule out ", "suspect ", "uncertain whether "),
        "hypothetical": ("if ", "return if ", "should ", "in case of ", "watch for ", "return precautions given for "),
        "family": (
            "family history of ",
            "family medical history notable for ",
            "family history:",
            "family history reveals ",
            "paternal history includes ",
            "hereditary risk: ",
            "sibling diagnosed with ",
            "mother had ",
            "mother with ",
            "father had ",
        ),
        "historical": (
            "history of ",
            "past history of ",
            "prior ",
            "previous ",
            "previous diagnoses include ",
            "history notable for ",
            "pmh includes ",
            "historical conditions: ",
            "known ",
        ),
        "terminators": (".", ";", "\n", " but ", " however ", " except ", " and takes ", " started "),
    },
    "fr": {
        "pseudo_neg": ("sans changement", "pas d'augmentation", "non éliminé"),
        "neg_pre": ("absence de ", "absence d'", "sans ", "pas de ", "pas d'", "nie ", "négatif pour ", "indemne de ", "aucune ", "ni de ", "ni d'", "ni "),
        "neg_post": (" éliminé", " écarté", " écartée", " exclu", " exclue", " soit absente"),
        "possible": (
            "suspicion de ",
            "suspicion d'",
            "probable ",
            "possible ",
            "évoquant ",
            "à confirmer",
            "doute sur ",
            "éliminer ",
            "visant à écarter ",
        ),
        "hypothetical": (
            "si ",
            "si apparition de ",
            "si survenue de ",
            "en cas de ",
            "surveiller l'apparition éventuelle de ",
            "surveiller l'éventuelle apparition de ",
            "guetter ",
        ),
        "family": (
            "antécédents familiaux de ",
            "antécédents familiaux d'",
            "histoire familiale marquée par ",
            "prédisposition familiale à ",
            "du côté paternel, on note ",
            "enquête familiale retrouvant ",
            "mère ",
            "père ",
        ),
        "historical": (
            "antécédent de ",
            "antécédent d'",
            "antécédents de ",
            "ancien ",
            "antécédents documentés : ",
            "connu pour ",
            "antécédent ancien de ",
            "antécédents médicaux : ",
            "pathologie chronique connue : ",
        ),
        "terminators": (".", ";", "\n", " mais ", " cependant ", " toutefois ", " et prend ", " traitée par ", " traité par "),
    },
}


def _compile_cue_pattern(cue: str) -> re.Pattern[str]:
    core = cue.strip()
    prefix = r"(?<!\w)" if core and (core[0].isalnum() or core[0] == "_") else ""
    suffix = r"(?!\w)" if core and (core[-1].isalnum() or core[-1] == "_") else ""
    return re.compile(f"{prefix}{re.escape(core)}{suffix}", flags=re.UNICODE | re.IGNORECASE)


_COMPILED_CUES: dict[str, dict[str, tuple[re.Pattern[str], ...]]] = {
    lang: {
        cat: tuple(_compile_cue_pattern(c) for c in cues)
        for cat, cues in groups.items()
        if cat != "terminators"
    }
    for lang, groups in CONTEXT_CUES.items()
}

_COMPILED_TERMINATORS: dict[str, tuple[re.Pattern[str], ...]] = {
    lang: tuple(
        _compile_cue_pattern(t) if t.strip().isalnum() or " " in t.strip() else re.compile(re.escape(t), flags=re.IGNORECASE)
        for t in groups["terminators"]
    )
    for lang, groups in CONTEXT_CUES.items()
}


def _matches_any(patterns: tuple[re.Pattern[str], ...], window: str) -> bool:
    return any(p.search(window) is not None for p in patterns)


def classify_assertion_context(text: str, start: int, end: int, language: str = "en") -> str:
    """ConText/NegEx-style assertion classification with Unicode word-boundary cue matching."""
    lang = language if language in CONTEXT_CUES else "en"
    compiled = _COMPILED_CUES[lang]
    term_patterns = _COMPILED_TERMINATORS[lang]

    # Restrict scope to the current sentence/clause
    pre_window = text[max(0, start - 90) : start].lower()
    post_window = text[end : min(len(text), end + 45)].lower()

    for term_pat in term_patterns:
        matches_pre = list(term_pat.finditer(pre_window))
        if matches_pre:
            pre_window = pre_window[matches_pre[-1].end() :]
        m_post = term_pat.search(post_window)
        if m_post:
            post_window = post_window[: m_post.start()]

    full_clause = pre_window + " " + text[start:end].lower() + " " + post_window
    if _matches_any(compiled["pseudo_neg"], full_clause):
        return "present"
    if _matches_any(compiled["family"], pre_window):
        return "family"
    if _matches_any(compiled["hypothetical"], pre_window):
        return "hypothetical"
    if _matches_any(compiled["neg_pre"], pre_window) or _matches_any(compiled["neg_post"], post_window):
        return "absent"
    if _matches_any(compiled["possible"], pre_window) or _matches_any(compiled["possible"], post_window):
        return "possible"
    if _matches_any(compiled["historical"], pre_window):
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
