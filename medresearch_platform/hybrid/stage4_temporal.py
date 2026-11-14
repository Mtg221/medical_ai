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


def normalize_timex(text: str, ttype: str, document_date: str) -> str | None:
    """Normalize EN/FR date, relative, or duration expression against document_date."""
    low = text.lower().strip()
    if ttype == "DATE" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", low):
        return low
    m_rel = re.search(r"(?:il y a\s+)?(\d+)\s*(?:days?\s+ago|jours?)", low)
    if ttype == "RELATIVE" and m_rel:
        days = int(m_rel.group(1))
        try:
            anchor = date.fromisoformat(document_date)
            return (anchor - timedelta(days=days)).isoformat()
        except ValueError:
            return None
    m_dur = re.search(r"(\d+)\s*(?:days?|jours?)", low)
    if ttype == "DURATION" and m_dur:
        return f"P{int(m_dur.group(1))}D"
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
        for d in train_docs[:250]:
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
