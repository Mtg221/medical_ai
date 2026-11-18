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
        "neg_pre": (
            "no ",
            "not ",
            "never ",
            "denies ",
            "without ",
            "free of ",
            "negative for ",
            "absence of ",
            "no evidence of ",
            "rules out ",
            "ruled out ",
            "excludes ",
            "excluded ",
        ),
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
        "terminators": (".", ";", "\n", " but ", " however ", " except ", " although ", " whereas "),
    },
    "fr": {
        "pseudo_neg": ("sans changement", "pas d'augmentation", "non éliminé"),
        "neg_pre": (
            "absence de ",
            "absence d'",
            "sans ",
            "pas de ",
            "pas d'",
            "pas d'argument pour ",
            "pas de signe en faveur de ",
            "pas de signe en faveur d'",
            "nie ",
            "négatif pour ",
            "indemne de ",
            "aucun ",
            "aucune ",
            "ni de ",
            "ni d'",
            "ni ",
            "élimine ",
            "éliminé ",
            "éliminée ",
            "écarte ",
            "écarté ",
            "écartée ",
            "exclut ",
            "exclu ",
            "exclue ",
        ),
        "neg_post": (
            " éliminé",
            " éliminée",
            " écarté",
            " écartée",
            " exclu",
            " exclue",
            " soit absente",
            " non retrouvé",
            " non retrouvée",
            " non retrouvés",
            " non retrouvées",
        ),
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
        "terminators": (".", ";", "\n", " mais ", " cependant ", " toutefois ", " sauf ", " alors que "),
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


_CONJ_PREDICATE_BREAK: dict[str, re.Pattern[str]] = {
    "en": re.compile(
        r"(?<!\w)(?:and|or)\s+(?:has|have|had|takes|take|took|taking|reports|presents|started|receives|is|was|are|were)\b",
        flags=re.UNICODE | re.IGNORECASE,
    ),
    "fr": re.compile(
        r"(?<!\w)(?:et|ou)\s+(?:a|ont|prend|prennent|présente|rapporte|reçoit|est|sont|était|étaient|sous)\b",
        flags=re.UNICODE | re.IGNORECASE,
    ),
}

_COMMA_LIST_LEAD_RE: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"^\s*(?:or|nor|and)\b", flags=re.UNICODE | re.IGNORECASE),
    "fr": re.compile(r"^\s*(?:ni|ou|et)\b", flags=re.UNICODE | re.IGNORECASE),
}

_COMMA_CLAUSE_PREDICATE_RE: dict[str, re.Pattern[str]] = {
    "en": re.compile(
        r"^\s*(?:has|have|had|takes|take|took|taking|reports|presents|started|receives|is|was|are|were|on)\b",
        flags=re.UNICODE | re.IGNORECASE,
    ),
    "fr": re.compile(
        r"^\s*(?:a|ont|prend|prennent|présente|rapporte|reçoit|est|sont|était|étaient|sous|traité|traitée|opéré|opérée)\b",
        flags=re.UNICODE | re.IGNORECASE,
    ),
}

_LIST_CONJ_RE: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"(?<!\w)(?:or|nor|and)(?!\w)", flags=re.UNICODE | re.IGNORECASE),
    "fr": re.compile(r"(?<!\w)(?:ni|ou|et)(?!\w)", flags=re.UNICODE | re.IGNORECASE),
}


_PASSIVE_EXCLUDING_PREP_RE: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"(?<!\w)(?:excluded|ruled\s+out)\s+(?:on|by)\b", flags=re.UNICODE | re.IGNORECASE),
    "fr": re.compile(r"(?<!\w)(?:écartée?|éliminée?|exclue?)\s+par\b", flags=re.UNICODE | re.IGNORECASE),
}


def _apply_general_clause_scope(
    pre_window: str,
    target_lower: str,
    post_window: str,
    lang: str,
) -> str:
    """Break scope on conjunction+predicate, passive 'excluded on/by <test>', or non-list commas."""
    prep_matches = list(_PASSIVE_EXCLUDING_PREP_RE[lang].finditer(pre_window))
    if prep_matches:
        pre_window = pre_window[prep_matches[-1].end() :]

    conj_pat = _CONJ_PREDICATE_BREAK[lang]
    m_conj = list(conj_pat.finditer(pre_window))
    if m_conj:
        pre_window = pre_window[m_conj[-1].start() :]

    if "," in pre_window:
        last_comma = pre_window.rfind(",")
        after_comma = pre_window[last_comma + 1 :]
        clause_span = f"{after_comma} {target_lower} {post_window}"
        starts_with_list_marker = _COMMA_LIST_LEAD_RE[lang].search(after_comma) is not None
        starts_with_predicate = _COMMA_CLAUSE_PREDICATE_RE[lang].search(after_comma) is not None
        has_list_conj = _LIST_CONJ_RE[lang].search(clause_span) is not None
        # Continue across comma only if it introduces a coordinated list item (e.g. ", ni X" or ", chills or sweats")
        # and does not start a new verbal/prepositional clause.
        if starts_with_predicate or (not starts_with_list_marker and not has_list_conj):
            pre_window = after_comma

    return pre_window


def classify_assertion_context(text: str, start: int, end: int, language: str = "en") -> str:
    """ConText/NegEx-style assertion classification with Unicode word-boundary cue matching and general clause scope."""
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

    target_lower = text[start:end].lower()
    pre_window = _apply_general_clause_scope(pre_window, target_lower, post_window, lang)

    full_clause = pre_window + " " + target_lower + " " + post_window
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
