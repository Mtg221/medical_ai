"""Stage 4: Temporal Analysis — CRF Temporal Tagger + EN/FR ISO Normalizer."""

from __future__ import annotations

import re
from datetime import date, timedelta

import sklearn_crfsuite

from medresearch_platform.approaches.classical_ml import _token_window_features
from medresearch_platform.approaches.tokenizer import (
    Token,
    tokenize_with_offsets,
)
from medresearch_platform.data.schema import AnnotatedDocument, TimexSpan
from medresearch_platform.eval_harness.metrics import PredictedTimex


def _align_timex_to_bio(tokens: list[Token], timex: list[TimexSpan]) -> list[str]:
    tags = ["O"] * len(tokens)
    for tx in sorted(timex, key=lambda t: (t.start, -(t.end - t.start))):
        matched = [i for i, tok in enumerate(tokens) if max(tok.start, tx.start) < min(tok.end, tx.end)]
        if not matched or any(tags[i] != "O" for i in matched):
            continue
        tags[matched[0]] = f"B-{tx.type}"
        for i in matched[1:]:
            tags[i] = f"I-{tx.type}"
    return tags


_MONTHS: dict[str, int] = {
    "january": 1, "janvier": 1, "jan": 1,
    "february": 2, "fevrier": 2, "février": 2, "feb": 2, "fev": 2, "fév": 2,
    "march": 3, "mars": 3, "mar": 3,
    "april": 4, "avril": 4, "apr": 4, "avr": 4,
    "may": 5, "mai": 5,
    "june": 6, "juin": 6, "jun": 6,
    "july": 7, "juillet": 7, "jul": 7, "juil": 7,
    "august": 8, "aout": 8, "août": 8, "aug": 8,
    "september": 9, "septembre": 9, "sep": 9, "sept": 9,
    "october": 10, "octobre": 10, "oct": 10,
    "november": 11, "novembre": 11, "nov": 11,
    "december": 12, "decembre": 12, "décembre": 12, "dec": 12, "déc": 12,
}


def normalize_timex(text: str, ttype: str, document_date: str) -> str | None:
    """Normalize EN/FR date, relative, duration, or frequency expression against document_date."""
    low = re.sub(r"\s+", " ", text.lower().strip())
    try:
        anchor = date.fromisoformat(document_date)
    except ValueError:
        anchor = None

    if ttype == "DATE":
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", low):
            return low
        m_dmy = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", low)
        if m_dmy:
            day, month, year = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"
        # Month DD, YYYY or Month DDth
        m_en = re.fullmatch(
            r"([a-zàâäéèêëîïôöùûü]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?", low
        )
        if m_en and m_en.group(1) in _MONTHS:
            month = _MONTHS[m_en.group(1)]
            day = int(m_en.group(2))
            year_opt: int | None = (
                int(m_en.group(3)) if m_en.group(3) else (anchor.year if anchor else None)
            )
            if year_opt is not None and 1 <= day <= 31:
                return f"{year_opt:04d}-{month:02d}-{day:02d}"
        # DD mois YYYY or DD mois
        m_fr = re.fullmatch(r"(\d{1,2})\s+([a-zàâäéèêëîïôöùûü]+)(?:\s+(\d{4}))?", low)
        if m_fr and m_fr.group(2) in _MONTHS:
            day = int(m_fr.group(1))
            month = _MONTHS[m_fr.group(2)]
            year_opt = int(m_fr.group(3)) if m_fr.group(3) else (anchor.year if anchor else None)
            if year_opt is not None and 1 <= day <= 31:
                return f"{year_opt:04d}-{month:02d}-{day:02d}"
        # Month YYYY / mois YYYY
        m_my = re.fullmatch(r"([a-zàâäéèêëîïôöùûü]+)\s+(\d{4})", low)
        if m_my and m_my.group(1) in _MONTHS:
            return f"{int(m_my.group(2)):04d}-{_MONTHS[m_my.group(1)]:02d}"
        return None

    if ttype == "RELATIVE":
        if anchor is None:
            return None
        if low in {"yesterday", "la veille", "hier"}:
            return (anchor - timedelta(days=1)).isoformat()
        if low in {"today", "aujourd'hui"} or re.fullmatch(r"(?:pod\s*\d+|j\d+\s*post-?op)", low):
            return anchor.isoformat()
        m_fut = re.search(r"(?:in|dans)\s+(\d+)\s+(days?|jours?|weeks?|semaines?|months?|mois)", low)
        if m_fut:
            n = int(m_fut.group(1))
            unit = m_fut.group(2)
            mult = 7 if unit.startswith(("week", "semaine")) else (30 if unit.startswith(("month", "mois")) else 1)
            return (anchor + timedelta(days=n * mult)).isoformat()
        m_past = re.search(
            r"(?:(?:il y a|depuis)\s+)?(\d+)\s+(days?|jours?|weeks?|semaines?|months?|mois)(?:\s+ago)?",
            low,
        )
        if m_past and ("ago" in low or "il y a" in low or "depuis" in low):
            n = int(m_past.group(1))
            unit = m_past.group(2)
            mult = 7 if unit.startswith(("week", "semaine")) else (30 if unit.startswith(("month", "mois")) else 1)
            return (anchor - timedelta(days=n * mult)).isoformat()
        return None

    if ttype == "DURATION":
        m_dur = re.search(r"(\d+)\s*(days?|jours?|weeks?|semaines?|months?|mois|years?|ans?)", low)
        if m_dur:
            n = int(m_dur.group(1))
            unit = m_dur.group(2)
            if unit.startswith(("week", "semaine")):
                return f"P{n}W"
            if unit.startswith(("month", "mois")):
                return f"P{n}M"
            if unit.startswith(("year", "an")):
                return f"P{n}Y"
            return f"P{n}D"
        return None

    if ttype == "FREQUENCY":
        if any(p in low for p in ("once daily", "1 fois par jour", "qd")):
            return "QD"
        if any(p in low for p in ("twice daily", "b.i.d", "bid", "matin et soir", "1 cp x 2/j")):
            return "BID"
        if any(p in low for p in ("three times daily", "t.i.d", "tid", "3 fois par jour", "toutes les 8 heures")):
            return "TID"
        if any(p in low for p in ("as needed", "si besoin", "prn")):
            return "PRN"
        return None

    return None


class CRFTemporalTagger:
    """sklearn-crfsuite CRF trained on train_docs TIMEX BIO tags + EN/FR normalizer."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=60,
            all_possible_transitions=True,
        )
        self._fitted = False

    def fit(self, train_docs: list[AnnotatedDocument]) -> CRFTemporalTagger:
        x_tr = []
        y_tr = []
        for d in train_docs:
            toks = tokenize_with_offsets(d.text)
            if not toks:
                continue
            x_tr.append([_token_window_features(toks, i) for i in range(len(toks))])
            y_tr.append(_align_timex_to_bio(toks, d.timex))
        self.crf.fit(x_tr, y_tr)
        self._fitted = True
        return self

    def predict(self, text: str, document_date: str) -> list[PredictedTimex]:
        if not self._fitted:
            raise RuntimeError("CRFTemporalTagger must be fitted before predict()")
        toks = tokenize_with_offsets(text)
        if not toks:
            return []
        tags = list(self.crf.predict_single([_token_window_features(toks, i) for i in range(len(toks))]))

        timex_list: list[PredictedTimex] = []
        cur_type: str | None = None
        cur_start: int | None = None
        cur_end: int | None = None

        def _flush() -> None:
            nonlocal cur_type, cur_start, cur_end
            if cur_type is not None and cur_start is not None and cur_end is not None:
                span_txt = text[cur_start:cur_end]
                timex_list.append(
                    PredictedTimex(
                        type=cur_type,
                        start=cur_start,
                        end=cur_end,
                        text=span_txt,
                        normalized=normalize_timex(span_txt, cur_type, document_date),
                    )
                )
            cur_type = None
            cur_start = None
            cur_end = None

        for tok, tag in zip(toks, tags, strict=True):
            if tag.startswith("B-"):
                _flush()
                cur_type = tag[2:]
                cur_start = tok.start
                cur_end = tok.end
            elif tag.startswith("I-") and cur_type == tag[2:]:
                cur_end = tok.end
            else:
                _flush()
        _flush()
        return timex_list
