"""Bilingual (EN/FR) ConText/NegEx assertion classifier, TIMEX extractor, and proximity relation linker."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

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
        "possible": (
            "suspected ",
            "possible ",
            "probable ",
            "likely ",
            "question of ",
            "rule out ",
            "suspect ",
            "uncertain whether ",
            "uncertain",
            "unclear",
            "borderline ",
            "differential",
            "not established",
            "pending",
            "not yet done",
            "not completed",
        ),
        "hypothetical": (
            "if ",
            "return if ",
            "should ",
            "in case of ",
            "watch for ",
            "return precautions",
            "seek urgent",
        ),
        "family": (
            "family history of ",
            "family medical history notable for ",
            "family history:",
            "family:",
            "family history reveals ",
            "paternal history includes ",
            "hereditary risk: ",
            "sibling diagnosed with ",
            "mother ",
            "mother—",
            "mother-",
            "father ",
            "father—",
            "father-",
            "brother ",
            "brother—",
            "brother-",
            "sister ",
            "sister—",
            "sister-",
            "maternal ",
            "paternal ",
            "mother had ",
            "mother has ",
            "mother with ",
            "her mother ",
            "his mother ",
            "father had ",
            "father has ",
            "her father ",
            "his father ",
            "parents have ",
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
            "childhood",
            "inactive",
            "discontinued",
            "old prescription",
            "old note",
            "old chart",
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
            "ne montre pas ",
            "ne retrouve pas ",
            "n'objective pas ",
            "ne met pas en évidence ",
            "ne révèle pas ",
            "n'a pas montré ",
            "ne présente pas ",
            "ne montre aucun",
            "ne retrouve aucun",
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
            "antécédent familial ",
            "histoire familiale marquée par ",
            "prédisposition familiale à ",
            "du côté paternel, on note ",
            "enquête familiale retrouvant ",
            "sa mère ",
            "son père ",
            "mère ",
            "père ",
        ),
        "historical": (
            "antécédent ",
            "antécédents ",
            "antécédent de ",
            "antécédent d'",
            "antécédents de ",
            "ancien ",
            "précédent ",
            "précédente ",
            "antérieur ",
            "antérieure ",
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


@dataclass(frozen=True)
class AssertionAttributes:
    """Orthogonal 3-axis clinical assertion contract (NLP-002)."""

    certainty: Literal["confirmed", "possible", "absent"] = "confirmed"
    temporality: Literal["current", "historical"] = "current"
    experiencer: Literal["patient", "family"] = "patient"
    legacy_assertion: str = "present"


def derive_assertion_attributes_from_legacy(assertion: str) -> AssertionAttributes:
    """Backward-compatible projection from legacy single-axis assertion enum without silent loss."""
    low = str(assertion or "present").strip().lower()
    if low == "absent":
        return AssertionAttributes(certainty="absent", temporality="current", experiencer="patient", legacy_assertion="absent")
    if low in {"possible", "uncertain", "hypothetical"}:
        return AssertionAttributes(certainty="possible", temporality="current", experiencer="patient", legacy_assertion=low)
    if low == "historical":
        return AssertionAttributes(certainty="confirmed", temporality="historical", experiencer="patient", legacy_assertion="historical")
    if low in {"family", "family_history"}:
        return AssertionAttributes(certainty="confirmed", temporality="current", experiencer="family", legacy_assertion="family")
    return AssertionAttributes(certainty="confirmed", temporality="current", experiencer="patient", legacy_assertion="present")


def classify_assertion_attributes(text: str, start: int, end: int, language: str = "en") -> AssertionAttributes:
    """Orthogonal 3-axis ConText/NegEx classifier (certainty × temporality × experiencer)."""
    lang = language if language in CONTEXT_CUES else "en"
    compiled = _COMPILED_CUES[lang]
    term_patterns = _COMPILED_TERMINATORS[lang]

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

    is_pseudo_neg = _matches_any(compiled["pseudo_neg"], full_clause)
    is_family = _matches_any(compiled["family"], pre_window)
    is_hypothetical = _matches_any(compiled["hypothetical"], pre_window) or _matches_any(
        compiled["hypothetical"], post_window
    )
    is_negated = (not is_pseudo_neg) and (
        _matches_any(compiled["neg_pre"], pre_window) or _matches_any(compiled["neg_post"], post_window)
    )
    has_trailing_qmark = bool(re.match(r"^\s*['’\"”)]*\?", text[end : min(len(text), end + 6)]))
    is_possible = (not is_pseudo_neg) and (
        has_trailing_qmark
        or _matches_any(compiled["possible"], pre_window)
        or _matches_any(compiled["possible"], post_window)
    )
    is_historical = _matches_any(compiled["historical"], pre_window) or _matches_any(
        compiled["historical"], post_window
    )

    certainty: Literal["confirmed", "possible", "absent"]
    if is_negated:
        certainty = "absent"
    elif is_possible or is_hypothetical:
        certainty = "possible"
    else:
        certainty = "confirmed"

    temporality: Literal["current", "historical"] = "historical" if is_historical else "current"
    experiencer: Literal["patient", "family"] = "family" if is_family else "patient"

    if is_pseudo_neg:
        legacy = "present"
    elif is_family:
        legacy = "family"
    elif is_hypothetical:
        legacy = "hypothetical"
    elif is_negated:
        legacy = "absent"
    elif is_possible:
        legacy = "possible"
    elif is_historical:
        legacy = "historical"
    else:
        legacy = "present"

    return AssertionAttributes(
        certainty=certainty,
        temporality=temporality,
        experiencer=experiencer,
        legacy_assertion=legacy,
    )


def classify_assertion_context(text: str, start: int, end: int, language: str = "en") -> str:
    """Backward-compatible single-axis ConText/NegEx assertion classifier."""
    return classify_assertion_attributes(text, start, end, language=language).legacy_assertion


def apply_context_and_relations(
    doc: AnnotatedDocument,
    entities: list[PredictedEntity],
) -> tuple[list[PredictedEntity], list[PredictedRelation], list[PredictedTimex]]:
    """Apply 3-axis ConText assertion, extract proximity relations, and normalize simple TIMEX expressions."""
    enriched: list[PredictedEntity] = []
    for e in entities:
        attrs = classify_assertion_attributes(doc.text, e.start, e.end, language=doc.language)
        enriched.append(
            PredictedEntity(
                type=e.type,
                start=e.start,
                end=e.end,
                text=e.text,
                assertion=attrs.legacy_assertion,
                confidence=e.confidence,
                needs_review=e.needs_review,
                confidence_variance=e.confidence_variance,
                high_uncertainty=e.high_uncertainty,
                certainty=attrs.certainty,
                temporality=attrs.temporality,
                experiencer=attrs.experiencer,
            )
        )

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
            elif head.type == "LAB_TEST" and tail.type == "LAB_VALUE" and head.end <= tail.start and tail.start - head.end <= 16:
                rtype = "HAS_VALUE"
            elif head.type in ("LAB_VALUE", "LAB_TEST") and tail.type == "LAB_UNIT" and head.end <= tail.start and tail.start - head.end <= 16:
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
