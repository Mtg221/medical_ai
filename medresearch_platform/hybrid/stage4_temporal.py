"""Stage 4: Temporal Analysis — CRF Temporal Tagger + EN/FR ISO Normalizer."""

from __future__ import annotations

import calendar
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


def _valid_iso_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _parse_header_date_token(raw_val: str, *, is_english_header: bool = False) -> str | None:
    cleaned = raw_val.strip()
    m_iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", cleaned)
    if m_iso:
        y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        valid = _valid_iso_date(y, m, d)
        if valid is not None:
            return valid

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
        valid = _valid_iso_date(y, m, d)
        if valid is not None:
            return valid

    m_fr = re.search(r"\b(\d{1,2})\s+([a-zA-Zàâäéèêëîïôöùûü]+)\s+(\d{4})\b", cleaned)
    if m_fr and m_fr.group(2).lower() in _MONTHS:
        d = int(m_fr.group(1))
        m = _MONTHS[m_fr.group(2).lower()]
        y = int(m_fr.group(3))
        valid = _valid_iso_date(y, m, d)
        if valid is not None:
            return valid

    m_en = re.search(r"\b([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})\b", cleaned)
    if m_en and m_en.group(1).lower() in _MONTHS:
        m = _MONTHS[m_en.group(1).lower()]
        d = int(m_en.group(2))
        y = int(m_en.group(3))
        valid = _valid_iso_date(y, m, d)
        if valid is not None:
            return valid

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


_NUM_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "eighteen": 18,
    "twenty": 20, "twenty-four": 24, "thirty": 30,
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11, "douze": 12,
    "quatorze": 14, "quinze": 15, "vingt": 20, "trente": 30,
}

_WEEKDAYS: dict[str, int] = {
    "monday": 0, "mon": 0, "lundi": 0,
    "tuesday": 1, "tue": 1, "tues": 1, "mardi": 1,
    "wednesday": 2, "wed": 2, "mercredi": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "jeudi": 3,
    "friday": 4, "fri": 4, "vendredi": 4,
    "saturday": 5, "sat": 5, "samedi": 5,
    "sunday": 6, "sun": 6, "dimanche": 6,
}

_NUM_TOKEN_PATTERN = r"(?:\d+|" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) + r")"


def _parse_number_token(tok: str) -> int | None:
    cleaned = tok.strip().lower()
    if cleaned.isdigit():
        return int(cleaned)
    return _NUM_WORDS.get(cleaned)


def _expand_year(year_str: str) -> int:
    cleaned = year_str.strip().lstrip("'’")
    val = int(cleaned)
    if len(cleaned) == 2:
        return 2000 + val if val <= 49 else 1900 + val
    return val


def _add_calendar_months(anchor: date, delta_months: int) -> date:
    total = (anchor.year * 12 + (anchor.month - 1)) + delta_months
    year, zero_month = divmod(total, 12)
    month = zero_month + 1
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(anchor.day, max_day))


def _normalize_relative_expr(low: str, anchor: date) -> str | None:
    """Resolve general EN/FR relative temporal expressions against anchor date."""
    if re.fullmatch(r"(?:yesterday|hier|la veille)(?:\s+(?:morning|afternoon|evening|night|matin|soir))?", low):
        return (anchor - timedelta(days=1)).isoformat()
    if low in {"avant-hier", "the day before yesterday", "day before yesterday"}:
        return (anchor - timedelta(days=2)).isoformat()
    if re.fullmatch(r"(?:tomorrow|demain)(?:\s+(?:morning|afternoon|evening|night|matin|soir))?", low):
        return (anchor + timedelta(days=1)).isoformat()
    # In daily surgical progress notes dated on POD N / J{N} post-op, the expression refers to the note's anchor date.
    if re.fullmatch(r"(?:today|aujourd['’]hui)(?:\s+(?:morning|afternoon|evening|night|matin|soir))?", low) or re.fullmatch(
        r"(?:pod\s*#?\s*\d+|j\s*\+?\s*\d+\s*post-?op)", low
    ):
        return anchor.isoformat()

    if re.fullmatch(r"(?:last week|la semaine derni[eè]re)", low):
        return (anchor - timedelta(days=7)).isoformat()
    if re.fullmatch(r"(?:next week|la semaine prochaine)", low):
        return (anchor + timedelta(days=7)).isoformat()
    if re.fullmatch(r"(?:last month|le mois dernier)", low):
        return _add_calendar_months(anchor, -1).isoformat()
    if re.fullmatch(r"(?:next month|le mois prochain)", low):
        return _add_calendar_months(anchor, 1).isoformat()

    # Weekday expressions: next Tuesday / last Monday / mardi prochain / lundi dernier
    m_wd_en = re.fullmatch(r"(next|last)\s+([a-zàâäéèêëîïôöùûü]+)", low)
    if m_wd_en and m_wd_en.group(2) in _WEEKDAYS:
        target_wd = _WEEKDAYS[m_wd_en.group(2)]
        if m_wd_en.group(1) == "next":
            delta = (target_wd - anchor.weekday()) % 7 or 7
            return (anchor + timedelta(days=delta)).isoformat()
        delta = (anchor.weekday() - target_wd) % 7 or 7
        return (anchor - timedelta(days=delta)).isoformat()

    m_wd_fr = re.fullmatch(r"([a-zàâäéèêëîïôöùûü]+)\s+(prochain|dernier)", low)
    if m_wd_fr and m_wd_fr.group(1) in _WEEKDAYS:
        target_wd = _WEEKDAYS[m_wd_fr.group(1)]
        if m_wd_fr.group(2) == "prochain":
            delta = (target_wd - anchor.weekday()) % 7 or 7
            return (anchor + timedelta(days=delta)).isoformat()
        delta = (anchor.weekday() - target_wd) % 7 or 7
        return (anchor - timedelta(days=delta)).isoformat()

    # French "quinze jours" idiomatic fortnight (14 days)
    if re.search(r"\b(?:il y a|depuis)\s+quinze\s+jours\b", low):
        return (anchor - timedelta(days=14)).isoformat()
    if re.search(r"\bdans\s+quinze\s+jours\b", low):
        return (anchor + timedelta(days=14)).isoformat()

    unit_pat = r"(days?|d|jours?|j|weeks?|wks?|semaines?|sem|months?|mos?|mois|years?|yrs?|ans?|ann[eé]es?)"

    m_fut = re.search(rf"\b(?:in|dans)\s+({_NUM_TOKEN_PATTERN})\s+{unit_pat}\b", low)
    if m_fut:
        n = _parse_number_token(m_fut.group(1))
        if n is not None:
            unit = m_fut.group(2)
            if unit.startswith(("week", "wk", "sem")):
                return (anchor + timedelta(days=n * 7)).isoformat()
            if unit.startswith(("month", "mo")):
                return (anchor + timedelta(days=n * 30)).isoformat()
            if unit.startswith(("year", "yr", "an")):
                return (anchor + timedelta(days=n * 365)).isoformat()
            return (anchor + timedelta(days=n)).isoformat()

    m_past = re.search(
        rf"(?:\b(?:il y a|depuis)\s+)?({_NUM_TOKEN_PATTERN})\s+{unit_pat}(?:\s+(?:ago|prior|earlier|before))?\b",
        low,
    )
    if m_past and any(cue in low for cue in ("ago", "prior", "earlier", "before", "il y a", "depuis")):
        n = _parse_number_token(m_past.group(1))
        if n is not None:
            unit = m_past.group(2)
            if unit.startswith(("week", "wk", "sem")):
                return (anchor - timedelta(days=n * 7)).isoformat()
            if unit.startswith(("month", "mo")):
                return (anchor - timedelta(days=n * 30)).isoformat()
            if unit.startswith(("year", "yr", "an")):
                return (anchor - timedelta(days=n * 365)).isoformat()
            return (anchor - timedelta(days=n)).isoformat()

    return None


def normalize_timex(text: str, ttype: str, document_date: str) -> str | None:
    """Normalize EN/FR date, relative, duration, or frequency expression against document_date."""
    low = re.sub(r"\s+", " ", text.lower().strip()).replace("’", "'")
    try:
        anchor = date.fromisoformat(document_date)
    except ValueError:
        anchor = None

    if ttype == "DATE":
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", low):
            try:
                return date.fromisoformat(low).isoformat()
            except ValueError:
                return None

        # Numeric DD/MM/YYYY, MM/DD/YYYY, DD/MM/YY, MM/DD/YY (with / or -)
        m_num = re.fullmatch(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2}|\d{4})", low)
        if m_num:
            a, b = int(m_num.group(1)), int(m_num.group(2))
            year = _expand_year(m_num.group(3))
            if a > 12 and 1 <= b <= 12:
                day, month = a, b
            elif b > 12 and 1 <= a <= 12:
                month, day = a, b
            else:
                day, month = a, b
            if 1 <= month <= 12 and 1 <= day <= calendar.monthrange(year, month)[1]:
                return f"{year:04d}-{month:02d}-{day:02d}"

        # DD-Mon-YYYY or DD-Mon-YY (e.g. 15-Mar-25)
        m_dmon = re.fullmatch(r"(\d{1,2})-([a-zàâäéèêëîïôöùûü]+)-('?\d{2}|\d{4})", low)
        if m_dmon and m_dmon.group(2) in _MONTHS:
            day = int(m_dmon.group(1))
            month = _MONTHS[m_dmon.group(2)]
            year = _expand_year(m_dmon.group(3))
            if 1 <= day <= calendar.monthrange(year, month)[1]:
                return f"{year:04d}-{month:02d}-{day:02d}"

        # Coarse month qualifiers: early/late, mid-/mi-, debut/début, fin
        m_coarse = re.fullmatch(
            r"(early|late|mid|mi|d[eé]but|fin)(?:[\s\-]+(?:de\s+|d')?)([a-zàâäéèêëîïôöùûü]+)(?:[\s,]+('?\d{2}|\d{4}))?",
            low,
        )
        if m_coarse and m_coarse.group(2) in _MONTHS:
            qual = m_coarse.group(1)
            month = _MONTHS[m_coarse.group(2)]
            year_opt = _expand_year(m_coarse.group(3)) if m_coarse.group(3) else (anchor.year if anchor else None)
            if year_opt is not None:
                if qual in {"early", "late"}:
                    return f"{year_opt:04d}-{month:02d}"
                if qual in {"mid", "mi"}:
                    return f"{year_opt:04d}-{month:02d}-15"
                if qual in {"debut", "début"}:
                    return f"{year_opt:04d}-{month:02d}-01"
                if qual == "fin":
                    last_day = calendar.monthrange(year_opt, month)[1]
                    return f"{year_opt:04d}-{month:02d}-{last_day:02d}"

        # Month YYYY / mois YYYY (check before Month DD so 4-digit years never split into DD+YY)
        m_my = re.fullmatch(r"([a-zàâäéèêëîïôöùûü]+)\.?\s+(\d{4}|'\d{2})", low)
        if m_my and m_my.group(1) in _MONTHS:
            return f"{_expand_year(m_my.group(2)):04d}-{_MONTHS[m_my.group(1)]:02d}"

        # English: Month DD[st|nd|rd|th][, 'YY | YYYY]
        m_en = re.fullmatch(
            r"([a-zàâäéèêëîïôöùûü]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:(?:,\s*|\s+)('?\d{2}|\d{4}))?",
            low,
        )
        if m_en and m_en.group(1) in _MONTHS:
            month = _MONTHS[m_en.group(1)]
            day = int(m_en.group(2))
            year_opt = _expand_year(m_en.group(3)) if m_en.group(3) else (anchor.year if anchor else None)
            if year_opt is not None and 1 <= day <= calendar.monthrange(year_opt, month)[1]:
                return f"{year_opt:04d}-{month:02d}-{day:02d}"

        # French: [le] DD[er|e] mois ['YY | YYYY]
        m_fr = re.fullmatch(
            r"(?:le\s+)?(\d{1,2})(?:er|e)?\s+([a-zàâäéèêëîïôöùûü]+)\.?(?:\s+('?\d{2}|\d{4}))?",
            low,
        )
        if m_fr and m_fr.group(2) in _MONTHS:
            day = int(m_fr.group(1))
            month = _MONTHS[m_fr.group(2)]
            year_opt = _expand_year(m_fr.group(3)) if m_fr.group(3) else (anchor.year if anchor else None)
            if year_opt is not None and 1 <= day <= calendar.monthrange(year_opt, month)[1]:
                return f"{year_opt:04d}-{month:02d}-{day:02d}"

        # Fallback: relative expressions tagged as DATE in clinical text
        if anchor is not None:
            rel = _normalize_relative_expr(low, anchor)
            if rel is not None:
                return rel
        return None

    if ttype == "RELATIVE":
        if anchor is None:
            return None
        return _normalize_relative_expr(low, anchor)

    if ttype == "DURATION":
        m_dur = re.search(
            rf"\b({_NUM_TOKEN_PATTERN})[\s\-]*"
            r"(hours?|hrs?|heures?|h|days?|jours?|d|j|weeks?|wks?|semaines?|sem|months?|mos?|mois|years?|yrs?|ans?|ann[eé]es?)\b",
            low,
        )
        if m_dur:
            n = _parse_number_token(m_dur.group(1))
            if n is not None:
                unit = m_dur.group(2)
                if unit in {"h", "hr", "hrs", "hour", "hours", "heure", "heures"}:
                    return f"PT{n}H"
                if unit.startswith(("week", "wk", "sem")):
                    return f"P{n}W"
                if unit.startswith(("month", "mo")):
                    return f"P{n}M"
                if unit.startswith(("year", "yr", "an")):
                    return f"P{n}Y"
                return f"P{n}D"
        return None

    if ttype == "FREQUENCY":
        m_qh = re.search(r"\bq\s*(\d+)\s*h\b", low)
        if m_qh:
            return f"PT{int(m_qh.group(1))}H"
        if re.search(r"\b(?:qhs|q\.h\.s\.?|au coucher|at bedtime)\b", low):
            return "QHS"
        if re.search(r"\b(?:three times daily|t\.i\.d\.?|tid|3\s*fois\s*(?:par|/)\s*jour|3\s*[x×]\s*/\s*j|3\s*cp\s*/\s*j|toutes les 8 heures)\b", low):
            return "TID"
        if re.search(r"\b(?:four times daily|q\.i\.d\.?|qid|4\s*fois\s*(?:par|/)\s*jour|4\s*[x×]\s*/\s*j)\b", low):
            return "QID"
        if re.search(r"\b(?:twice daily|b\.i\.d\.?|bid|matin et soir|2\s*fois\s*(?:par|/)\s*jour|2\s*[x×]\s*/\s*j|1\s*cp\s*[x×]\s*2\s*/\s*j)\b", low):
            return "BID"
        if re.search(r"\b(?:once daily|1\s*fois\s*(?:par|/)\s*jour|q\.d\.?|qd|1\s*[x×]\s*/\s*j|1\s*cp\s*/\s*j)\b", low):
            return "QD"
        if re.search(r"\b(?:once weekly|weekly|1\s*[x×]\s*/\s*sem(?:aine)?|1\s*fois\s*(?:par|/)\s*semaine|hebdomadaire)\b", low):
            return "P1W"
        if re.search(r"\b(?:as needed|si besoin|prn|p\.r\.n\.?)\b", low):
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
