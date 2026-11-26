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
        "pseudo_neg": ("no increase", "no change", "not ruled out", "not excluded", "not excludcd", "cannot exclude"),
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
        "neg_post": (
            " ruled out",
            " excluded",
            " negative",
            " absent",
            " was not observed",
            " not observed",
            " not administered",
            " not performed",
            " cancelled",
            " canceled",
        ),
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
            "family history reveals ",
            "paternal history includes ",
            "hereditary risk: ",
            "sibling diagnosed with ",
            "mother ",
            "father ",
            "brother ",
            "sister ",
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
            "stopped ",
            "has resolved",
        ),
        "terminators": (".", ";", "\n", " but ", " however ", " except ", " although ", " whereas "),
    },
    "fr": {
        "pseudo_neg": ("sans changement", "pas d'augmentation", "non éliminé", "non exclu", "sans rendez-vous"),
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
            " peu vraisemblable",
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
            "on discute ",
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
            "frère ",
            "sœur ",
            "soeur ",
            "antécédent maternel",
            "antécédent paternel",
            "maternel",
            "paternel",
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


_SECTION_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,4}\s*|\d+\.\s*|[-•*]\s*)?"
    r"(?P<hdr>"
    r"chief\s+complaint|presenting\s+complaint|history\s+of\s+present\s+illness|hpi|motif\s+de\s+consultation|"
    r"past\s+medical\s+history|past\s+history|pmhx?|ant[ée]c[ée]dents(?:\s+m[ée]dicaux)?|"
    r"family\s+history|ant[ée]c[ée]dents\s+familiaux|"
    r"medications?(?:\s+list|\s+reconciliation)?|traitements?(?:\s+de\s+sortie)?|"
    r"allergies?|"
    r"vital\s+signs?|vitals|constantes|"
    r"physical\s+examination|examination|examen\s+clinique|"
    r"investigations?|laboratory|blood\s+tests?|biologie|imagerie|imaging|"
    r"assessment(?:\s+fragments)?|impression|avis\s+et\s+diagnostic|"
    r"plan(?:\s*/\s*disposition|\s*/\s*follow-up)?|conduite\s+[àa]\s+tenir|"
    r"return\s+precautions?"
    r")\b",
    re.IGNORECASE,
)

_SECTION_CANON_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("family", "familiaux"), "family_history"),
    (("past", "pmh", "antécédent", "antecedent"), "pmh"),
    (("medication", "traitement"), "medications"),
    (("allerg",), "allergies"),
    (("vital", "constante"), "vitals"),
    (("exam",), "exam"),
    (("lab", "blood", "biologie", "investigation"), "labs"),
    (("imag",), "imaging"),
    (("return precaution",), "return_precautions"),
    (("assessment", "impression", "diagnostic"), "assessment"),
    (("plan", "conduite"), "plan"),
    (("chief", "presenting", "hpi", "history of present", "motif"), "hpi"),
)


def detect_section_at_offset(text: str, offset: int) -> str:
    """Return the canonical clinical section governing character `offset` in `text`."""
    prefix = text[: max(0, offset)]
    last_hdr: str | None = None
    for m in _SECTION_HEADER_RE.finditer(prefix):
        last_hdr = m.group("hdr").lower()
    if not last_hdr:
        return "general"
    for needles, canon in _SECTION_CANON_MAP:
        if any(n in last_hdr for n in needles):
            return canon
    return "general"


_POST_CLAUSE_BREAK_RE = re.compile(
    r",\s*(?:though\s+|although\s+|negative\s+for|positive\s+for|no\s+|denies\s+|without\s+|pas\s+de\s+|sans\s+)",
    re.IGNORECASE,
)
_UNRESOLVED_RE = re.compile(
    r"\b(?:not\s+excluded|not\s+excludcd|cannot\s+(?:be\s+)?exclud(?:e|ed)|not\s+ruled\s+out|non\s+exclu|non\s+[ée]limin[ée]|not\s+assessed)\b",
    re.IGNORECASE,
)
_UNPERFORMED_SCAN_RE = re.compile(
    r"\b(?:study\s+not\s+performed|not\s+performed|cancelled|canceled|not\s+done|not\s+completed|non\s+r[ée]alis[ée])\b",
    re.IGNORECASE,
)
_FUTURE_PLANNED_RE = re.compile(
    r"\b(?:follow[- ]up\s+in|suivi\s+dans|in\s+\d+\s+(?:days?|weeks?|months?)|"
    r"in\s+(?:one|two|three|four|six)\s+(?:days?|weeks?|months?)|"
    r"dans\s+(?:un|deux|trois|quatre|six|\d+)\s+(?:jours?|semaines?|mois)|"
    r"scheduled\s+for|planned\s+for|appointment\s+pending|tomorrow|demain)\b",
    re.IGNORECASE,
)
_DISCONTINUED_RE = re.compile(
    r"\b(?:stopped|discontinued|arr[êe]t[ée]?|interrompu)\b",
    re.IGNORECASE,
)
_RESOLVED_RE = re.compile(r"\b(?:resolved|r[ée]solu|r[ée]solue|gu[ée]ri)\b", re.IGNORECASE)
_RECURRENT_RE = re.compile(r"\b(?:recurrent|r[ée]cidivant|r[ée]current)\b", re.IGNORECASE)
_SUSPECTED_RE = re.compile(r"\b(?:suspect(?:ed)?|suspicion|rule\s+out|[ée]voquant)\b", re.IGNORECASE)
_POST_SEEK_URGENT_RE = re.compile(
    r"\b(?:seek\s+urgent|reassess\s+asap|return\s+(?:immediately|to\s+(?:the\s+)?(?:ed|er))|go\s+to\s+(?:the\s+)?(?:ed|er)|call\s+911|consulter\s+en\s+urgence|retourner\s+aux\s+urgences)\b",
    re.IGNORECASE,
)

_REL_DETAIL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:mother|maternal|m[èe]re|maternel(?:le)?)\b", re.IGNORECASE), "mother"),
    (re.compile(r"\b(?:father|paternal|p[èe]re|paternel(?:le)?)\b", re.IGNORECASE), "father"),
    (re.compile(r"\b(?:brother|fr[èe]re)\b", re.IGNORECASE), "brother"),
    (re.compile(r"\b(?:sister|s[œo]ur)\b", re.IGNORECASE), "sister"),
    (re.compile(r"\b(?:sibling|family|familial(?:e|ux)?|grandmother|grandfather|parent)\b", re.IGNORECASE), "other_family"),
)


def _infer_experiencer_detail(pre_window: str, is_family: bool) -> str:
    if not is_family:
        return "patient"
    best_pos = -1
    best_label = "other_family"
    for pat, label in _REL_DETAIL_PATTERNS:
        for m in pat.finditer(pre_window):
            if m.start() >= best_pos:
                best_pos = m.start()
                best_label = label
    return best_label


@dataclass(frozen=True)
class AssertionAttributes:
    """Orthogonal 3-axis + fine-grained P1 clinical assertion contract."""

    certainty: Literal["confirmed", "possible", "absent"] = "confirmed"
    temporality: Literal["current", "historical"] = "current"
    experiencer: Literal["patient", "family"] = "patient"
    legacy_assertion: str = "present"
    experiencer_detail: str = "patient"
    assertion_detail: str = "present"
    temporality_detail: str = "current"
    section: str = "general"


def derive_assertion_attributes_from_legacy(assertion: str) -> AssertionAttributes:
    """Backward-compatible projection from legacy single-axis assertion enum without silent loss."""
    low = str(assertion or "present").strip().lower()
    if low == "absent":
        return AssertionAttributes(
            certainty="absent",
            temporality="current",
            experiencer="patient",
            legacy_assertion="absent",
            assertion_detail="absent",
        )
    if low in {"possible", "uncertain", "hypothetical"}:
        return AssertionAttributes(
            certainty="possible",
            temporality="current",
            experiencer="patient",
            legacy_assertion=low,
            assertion_detail="conditional" if low == "hypothetical" else low,
        )
    if low == "historical":
        return AssertionAttributes(
            certainty="confirmed",
            temporality="historical",
            experiencer="patient",
            legacy_assertion="historical",
            temporality_detail="historical",
            assertion_detail="present",
        )
    if low in {"family", "family_history"}:
        return AssertionAttributes(
            certainty="confirmed",
            temporality="current",
            experiencer="family",
            legacy_assertion="family",
            experiencer_detail="other_family",
            assertion_detail="present",
        )
    return AssertionAttributes(certainty="confirmed", temporality="current", experiencer="patient", legacy_assertion="present")


def classify_assertion_attributes(text: str, start: int, end: int, language: str = "en") -> AssertionAttributes:
    """Orthogonal 3-axis + section-aware fine-grained ConText/NegEx classifier."""
    lang = language if language in CONTEXT_CUES else "en"
    compiled = _COMPILED_CUES[lang]
    term_patterns = _COMPILED_TERMINATORS[lang]

    section = detect_section_at_offset(text, start)
    pre_window = text[max(0, start - 90) : start].lower()
    post_window = text[end : min(len(text), end + 45)].lower()

    for term_pat in term_patterns:
        matches_pre = list(term_pat.finditer(pre_window))
        if matches_pre:
            pre_window = pre_window[matches_pre[-1].end() :]
        m_post = term_pat.search(post_window)
        if m_post:
            post_window = post_window[: m_post.start()]

    m_post_comma = _POST_CLAUSE_BREAK_RE.search(post_window)
    if m_post_comma:
        post_window = post_window[: m_post_comma.start()]

    target_lower = text[start:end].lower()
    pre_window = _apply_general_clause_scope(pre_window, target_lower, post_window, lang)
    full_clause = pre_window + " " + target_lower + " " + post_window

    is_unresolved = bool(_UNRESOLVED_RE.search(full_clause))
    is_pseudo_neg = _matches_any(compiled["pseudo_neg"], full_clause)
    is_family = _matches_any(compiled["family"], pre_window) or (section == "family_history")
    # Extended same-sentence/same-line lookahead (up to 120 chars) for return-precaution urgent action clauses
    line_post = text[end : min(len(text), end + 120)].lower()
    for stop_ch in ("\n", ".", ";"):
        s_idx = line_post.find(stop_ch)
        if s_idx != -1:
            line_post = line_post[:s_idx]

    is_hypothetical = (
        _matches_any(compiled["hypothetical"], pre_window)
        or bool(_POST_SEEK_URGENT_RE.search(line_post))
        or (section == "return_precautions")
    )
    is_negated = (not is_pseudo_neg) and (not is_unresolved) and (
        _matches_any(compiled["neg_pre"], pre_window) or _matches_any(compiled["neg_post"], post_window)
    )
    # Safety check: never treat a negative scan finding as confirmed absent if the same paragraph states study was not performed
    if is_negated:
        para_start = text.rfind("\n\n", 0, start)
        para_end = text.find("\n\n", end)
        para_txt = text[(para_start + 2 if para_start != -1 else 0) : (para_end if para_end != -1 else len(text))]
        if _UNPERFORMED_SCAN_RE.search(para_txt):
            is_negated = False
            is_unresolved = True

    has_trailing_qmark = bool(re.match(r"^\s*['’\"”)]*\?", text[end : min(len(text), end + 6)]))
    is_possible = (not is_pseudo_neg) and (
        has_trailing_qmark
        or _matches_any(compiled["possible"], pre_window)
        or _matches_any(compiled["possible"], post_window)
    )
    is_future = bool(_FUTURE_PLANNED_RE.search(full_clause))
    is_discontinued = bool(_DISCONTINUED_RE.search(full_clause))
    is_resolved = bool(_RESOLVED_RE.search(full_clause))
    is_recurrent = bool(_RECURRENT_RE.search(full_clause))
    is_historical = (
        _matches_any(compiled["historical"], pre_window)
        or _matches_any(compiled["historical"], post_window)
        or is_discontinued
    )

    certainty: Literal["confirmed", "possible", "absent"]
    if is_negated:
        certainty = "absent"
    elif is_unresolved or is_possible or is_hypothetical:
        certainty = "possible"
    else:
        certainty = "confirmed"

    temporality: Literal["current", "historical"] = "historical" if is_historical else "current"
    experiencer: Literal["patient", "family"] = "family" if is_family else "patient"
    experiencer_detail = _infer_experiencer_detail(pre_window, is_family)

    if is_unresolved:
        assertion_detail = "unresolved"
    elif is_hypothetical:
        assertion_detail = "conditional"
    elif is_discontinued:
        assertion_detail = "discontinued"
    elif is_negated:
        assertion_detail = "absent"
    elif is_future:
        assertion_detail = "planned"
    elif is_possible:
        assertion_detail = "suspected" if _SUSPECTED_RE.search(full_clause) else "possible"
    else:
        assertion_detail = "present"

    if is_future or is_hypothetical:
        temporality_detail = "future_planned"
    elif is_resolved:
        temporality_detail = "resolved"
    elif is_recurrent:
        temporality_detail = "recurrent"
    elif is_historical:
        temporality_detail = "historical"
    else:
        temporality_detail = "current"

    if is_unresolved:
        legacy = "possible"
    elif is_pseudo_neg:
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
        experiencer_detail=experiencer_detail,
        assertion_detail=assertion_detail,
        temporality_detail=temporality_detail,
        section=section,
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
