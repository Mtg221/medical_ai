"""Shared EN/FR Person (Clinician/Patient) and Age expression detector for pre-NER masking (W9) and P10 redaction (W10)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class PersonSpan:
    pii_type: str  # "PATIENT_NAME" or "CLINICIAN_NAME"
    start: int
    end: int
    text: str


_CLINICIAN_TITLE_RE = re.compile(
    r"\b((?:Dr\.?|Docteur|Pr\.?|Prof\.?)\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ\-]{1,20}(?:\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ\-]{1,20})?)\b",
    re.UNICODE,
)

_PATIENT_LABELLED_RE = re.compile(
    r"(?:Patient(?:\s+Name)?|Nom(?:\s+du\s+patient)?|Prénom)\s*:\s*([A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ\-]*(?:\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ\-]*)*)",
    re.UNICODE,
)

_PATIENT_HONORIFIC_RE = re.compile(
    r"\b((?:Mme|Mlle|M\.|Mr\.?|Mrs\.?|Ms\.?|Miss|Monsieur|Madame)\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ\-]*(?:\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ\-]+)*)(?=\W|$)",
    re.UNICODE,
)

_UNTITLED_NAME_PAIR_RE = re.compile(
    r"(?<![\-/])\b([A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ\-]{1,20}(?:\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ\-]{1,20})+)\b(?![\-/])",
    re.UNICODE,
)

_AGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{1,3}\s*(?:ans?|years?(?:\s+old)?|y/?o|yrs?|months?\s+old|mois\s+d'[aâ]ge)\b", re.IGNORECASE),
    re.compile(r"\b(?:aged|age|âgée?|âgé)\s*(?:de\s+)?\d{1,3}(?:\s*(?:ans?|years?(?:\s+old)?|mois|months?))?\b", re.IGNORECASE),
)

_NON_NAME_WORDS: frozenset[str] = frozenset(
    {
        "patient",
        "patiente",
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "the",
        "this",
        "these",
        "those",
        "d-dimères",
        "d-dimers",
        "d-dimer",
        "cardiac",
        "brain",
        "liver",
        "consultation",
        "examen",
        "compte",
        "rendu",
        "diagnostic",
        "traitement",
        "antécédents",
        "antecedents",
        "allergies",
        "biologie",
        "imagerie",
        "tension",
        "douleur",
        "histoire",
        "clinique",
        "service",
        "hôpital",
        "hopital",
        "hospital",
        "centre",
        "center",
        "clinic",
        "clinique",
        "university",
        "université",
        "institut",
        "institute",
        "chest",
        "blood",
        "heart",
        "lung",
        "renal",
        "hepatic",
        "pulmonary",
        "clinical",
        "note",
        "summary",
        "discharge",
        "follow",
        "history",
        "physical",
        "exam",
        "impression",
        "plan",
        "assessment",
        "medication",
        "medications",
        "left",
        "right",
        "upper",
        "lower",
        "acute",
        "chronic",
        "type",
        "stage",
        "grade",
        "normal",
        "negative",
        "positive",
        "no",
        "aucun",
        "aucune",
        "pas",
        "sans",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "janvier",
        "février",
        "fevrier",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "aout",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
        "decembre",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "lundi",
        "mardi",
        "mercredi",
        "jeudi",
        "vendredi",
        "samedi",
        "dimanche",
        "dakar",
        "paris",
        "lyon",
        "marseille",
        "london",
        "new",
        "york",
        "saint",
        "sainte",
        "rue",
        "avenue",
        "boulevard",
        "street",
        "road",
        "lane",
        "drive",
    }
)


@lru_cache(maxsize=1)
def _medical_lexicon_words() -> tuple[frozenset[str], frozenset[str]]:
    from medresearch_platform.data.lexicon_v3 import (
        ANATOMY_EN,
        ANATOMY_FR,
        DIAGNOSES_EN,
        DIAGNOSES_FR,
        LABS_EN,
        LABS_FR,
        MEDICATIONS_EN,
        MEDICATIONS_FR,
        PROCEDURES_EN,
        PROCEDURES_FR,
        SYMPTOMS_EN,
        SYMPTOMS_FR,
    )
    from medresearch_platform.data.synthetic_v2 import (
        ANATOMIES,
        DIAGNOSES,
        LAB_TESTS,
        MEDICATIONS,
        PROCEDURES,
        SYMPTOMS,
    )

    phrases: set[str] = set()
    words: set[str] = set(_NON_NAME_WORDS)
    _AMBIGUOUS_SURNAME_COLORS = {"white", "black", "brown", "green", "gray", "grey", "young", "king", "hill", "wood", "stone"}
    for bucket in (
        ANATOMY_EN,
        ANATOMY_FR,
        DIAGNOSES_EN,
        DIAGNOSES_FR,
        LABS_EN,
        LABS_FR,
        MEDICATIONS_EN,
        MEDICATIONS_FR,
        PROCEDURES_EN,
        PROCEDURES_FR,
        SYMPTOMS_EN,
        SYMPTOMS_FR,
    ):
        for item in bucket:
            variants = getattr(item, "variants", (str(item),))
            for term in variants:
                low = str(term).strip().lower()
                if low:
                    phrases.add(low)
                    for w in re.findall(r"[a-zà-öø-ÿ\-]+", low):
                        if len(w) >= 3 and w not in _AMBIGUOUS_SURNAME_COLORS:
                            words.add(w)

    for v2_bucket in (ANATOMIES, DIAGNOSES, LAB_TESTS, MEDICATIONS, PROCEDURES, SYMPTOMS):
        for v2_item in v2_bucket:
            for attr in ("en", "fr"):
                low = str(getattr(v2_item, attr, "")).strip().lower()
                if low:
                    phrases.add(low)
                    for w in re.findall(r"[a-zà-öø-ÿ\-]+", low):
                        if len(w) >= 3 and w not in _AMBIGUOUS_SURNAME_COLORS:
                            words.add(w)
    return frozenset(phrases), frozenset(words)


def detect_person_spans(text: str) -> list[PersonSpan]:
    """Detect clinician names (Dr/Pr) and patient names (titled, labelled, or untitled First Last pairs)."""
    spans: list[PersonSpan] = []

    for m in _CLINICIAN_TITLE_RE.finditer(text):
        spans.append(PersonSpan("CLINICIAN_NAME", m.start(1), m.end(1), m.group(1)))

    for m in _PATIENT_LABELLED_RE.finditer(text):
        spans.append(PersonSpan("PATIENT_NAME", m.start(1), m.end(1), m.group(1)))

    for m in _PATIENT_HONORIFIC_RE.finditer(text):
        spans.append(PersonSpan("PATIENT_NAME", m.start(1), m.end(1), m.group(1)))

    med_phrases, med_words = _medical_lexicon_words()
    for m in _UNTITLED_NAME_PAIR_RE.finditer(text):
        word_matches = list(re.finditer(r"[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ\-]{1,20}", m.group(1)))
        while word_matches and word_matches[0].group(0).lower() in med_words:
            word_matches.pop(0)
        while word_matches and word_matches[-1].group(0).lower() in med_words:
            word_matches.pop()
        if len(word_matches) < 2 or len(word_matches) > 3:
            continue
        s = m.start(1) + word_matches[0].start()
        e = m.start(1) + word_matches[-1].end()
        if any(not (e <= sp.start or s >= sp.end) for sp in spans):
            continue
        cand = text[s:e]
        low = cand.lower()
        if low in med_phrases:
            continue
        toks = [wm.group(0).lower() for wm in word_matches]
        if any(t in med_words for t in toks):
            continue
        spans.append(PersonSpan("PATIENT_NAME", s, e, cand))

    # Deduplicate / keep longest non-overlapping
    spans.sort(key=lambda sp: (sp.start, -(sp.end - sp.start)))
    out: list[PersonSpan] = []
    last_end = -1
    for sp in spans:
        if sp.start >= last_end:
            out.append(sp)
            last_end = sp.end
    return out


def detect_age_spans(text: str) -> list[tuple[int, int]]:
    """Detect EN/FR age expression character spans `(start, end)`."""
    raw: list[tuple[int, int]] = []
    for pat in _AGE_PATTERNS:
        for m in pat.finditer(text):
            raw.append((m.start(), m.end()))
    raw.sort()
    return raw


def span_overlaps(start: int, end: int, blocked: list[tuple[int, int]]) -> bool:
    """Return True if `[start, end)` overlaps any span in `blocked`."""
    return any(not (end <= bs or start >= be) for bs, be in blocked)
