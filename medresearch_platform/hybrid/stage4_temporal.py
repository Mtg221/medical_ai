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


def _parse_header_date_token(raw_val: str, *, is_english_header: bool = False) -> str | None:
    cleaned = raw_val.strip()
    m_iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", cleaned)
    if m_iso:
        y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        if 1 <= m <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{m:02d}-{d:02d}"

    m_slash = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", cleaned)
    if m_slash:
        a, b, y = int(m_slash.group(1)), int(m_slash.group(2)), int(m_slash.group(3))
        if a > 12 and 1 <= b <= 12:
            d, m = a, b
        elif b > 12 and 1 <= a <= 12:
            m, d = a, b
        elif is_english_header:
            m, d = a, b
        else:
            d, m = a, b
        if 1 <= m <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{m:02d}-{d:02d}"

    m_fr = re.search(r"\b(\d{1,2})\s+([a-zA-Zàâäéèêëîïôöùûü]+)\s+(\d{4})\b", cleaned)
    if m_fr and m_fr.group(2).lower() in _MONTHS:
        d = int(m_fr.group(1))
        m = _MONTHS[m_fr.group(2).lower()]
        y = int(m_fr.group(3))
        if 1 <= d <= 31:
            return f"{y:04d}-{m:02d}-{d:02d}"

    m_en = re.search(r"\b([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})\b", cleaned)
    if m_en and m_en.group(1).lower() in _MONTHS:
        m = _MONTHS[m_en.group(1).lower()]
        d = int(m_en.group(2))
        y = int(m_en.group(3))
        if 1 <= d <= 31:
            return f"{y:04d}-{m:02d}-{d:02d}"

    return None


def resolve_document_date(text: str, fallback_date: str | None = None) -> tuple[str, str]:
    """Extract document anchor date from text headers first, falling back to form parameter or today."""
    header_window = text[:2500]
    header_patterns: tuple[tuple[re.Pattern[str], bool], ...] = (
        (re.compile(r"(?:discharge\s+date)\s*:\s*([^\n\r]+)", re.IGNORECASE), True),
        (re.compile(r"(?:date\s+de\s+sortie)\s*:\s*([^\n\r]+)", re.IGNORECASE), False),
        (re.compile(r"(?:admission\s+date|date\s+of\s+service|visit\s+date)\s*:\s*([^\n\r]+)", re.IGNORECASE), True),
        (re.compile(r"(?:date\s+d['’]entr[ée]e|date\s+de\s+consultation|date\s+de\s+l['’]examen)\s*:\s*([^\n\r]+)", re.IGNORECASE), False),
        (re.compile(r"(?:^|\n)\s*date\s*:\s*([^\n\r]+)", re.IGNORECASE), False),
        (re.compile(r"(?:^|\n)\s*le\s+(\d{1,2}(?:/\d{1,2}/\d{4}|\s+[a-zA-Zàâäéèêëîïôöùûü]+\s+\d{4}))", re.IGNORECASE), False),
    )
    for pat, is_en_hdr in header_patterns:
        m = pat.search(header_window)
        if m:
            is_en = is_en_hdr or ("date:" in m.group(0).lower() and "date :" not in m.group(0).lower())
            parsed = _parse_header_date_token(m.group(1), is_english_header=is_en)
            if parsed is not None:
                return parsed, "extracted_from_text"

    if fallback_date and fallback_date.strip():
        try:
            date.fromisoformat(fallback_date.strip())
            return fallback_date.strip(), "form_parameter"
        except ValueError:
            parsed_fb = _parse_header_date_token(fallback_date.strip())
            if parsed_fb:
                return parsed_fb, "form_parameter"

    return date.today().isoformat(), "system_today"


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

        regex_timex_rules: tuple[tuple[str, re.Pattern[str]], ...] = (
            ("RELATIVE", re.compile(r"\b(?:depuis|dans|il y a|in)\s+\d+\s+(?:jours?|semaines?|mois|days?|weeks?|months?)\b", re.IGNORECASE)),
            ("RELATIVE", re.compile(r"\b\d+\s+(?:days?|weeks?|months?)\s+ago\b", re.IGNORECASE)),
            ("RELATIVE", re.compile(r"\b(?:hier|aujourd'hui|la veille|yesterday|today)\b", re.IGNORECASE)),
            ("DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
            ("DATE", re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")),
            ("DURATION", re.compile(r"\b(?:for|pendant|durant)\s+\d+\s+(?:jours?|semaines?|mois|ans?|days?|weeks?|months?|years?)\b", re.IGNORECASE)),
        )
        for rx_type, rx_pat in regex_timex_rules:
            for m in rx_pat.finditer(text):
                s, e = m.start(), m.end()
                if any(max(s, tx.start) < min(e, tx.end) for tx in timex_list):
                    continue
                span_txt = text[s:e]
                timex_list.append(
                    PredictedTimex(
                        type=rx_type,
                        start=s,
                        end=e,
                        text=span_txt,
                        normalized=normalize_timex(span_txt, rx_type, document_date),
                    )
                )
        timex_list.sort(key=lambda t: t.start)
        return timex_list
