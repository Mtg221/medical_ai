"""Train-only tokenizer, vocabulary, BIO tag alignment, and span decoding."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from medresearch_platform.data.schema import ENTITY_TYPES, AnnotatedDocument, EntitySpan
from medresearch_platform.eval_harness.metrics import PredictedEntity

_TOKEN_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|PAT-[A-Z]{2}-\d+|\d+(?:[.,]\d+)?|\w+(?:-\w+)*|[^\w\s]",
    re.UNICODE,
)

BIO_TAGS: tuple[str, ...] = ("O",) + tuple(
    tag for etype in ENTITY_TYPES for tag in (f"B-{etype}", f"I-{etype}")
)
TAG_TO_ID: dict[str, int] = {t: i for i, t in enumerate(BIO_TAGS)}
ID_TO_TAG: dict[int, str] = {i: t for i, t in enumerate(BIO_TAGS)}


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize_with_offsets(text: str) -> list[Token]:
    """Tokenize text while preserving exact character offsets."""
    return [Token(text=m.group(0), start=m.start(), end=m.end()) for m in _TOKEN_RE.finditer(text)]


class TrainOnlyVocabulary:
    """Word vocabulary built strictly from the training split."""

    def __init__(self, max_size: int = 8000, min_freq: int = 1) -> None:
        self.max_size = max_size
        self.min_freq = min_freq
        self.word_to_id: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, docs: list[AnnotatedDocument]) -> TrainOnlyVocabulary:
        counts: Counter[str] = Counter()
        for d in docs:
            for tok in tokenize_with_offsets(d.text):
                counts[tok.text.lower()] += 1
        for word, freq in counts.most_common(self.max_size - 2):
            if freq >= self.min_freq and word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)
        return self

    def encode_token(self, word: str) -> int:
        return self.word_to_id.get(word.lower(), 1)

    def __len__(self) -> int:
        return len(self.word_to_id)


def align_entities_to_bio(tokens: list[Token], entities: list[EntitySpan]) -> list[str]:
    """Convert character-offset EntitySpans into BIO tags aligned to tokens."""
    tags = ["O"] * len(tokens)
    sorted_ents = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
    for ent in sorted_ents:
        matched_idxs = [
            i for i, tok in enumerate(tokens) if max(tok.start, ent.start) < min(tok.end, ent.end)
        ]
        if not matched_idxs:
            continue
        if any(tags[i] != "O" for i in matched_idxs):
            continue
        tags[matched_idxs[0]] = f"B-{ent.type}"
        for i in matched_idxs[1:]:
            tags[i] = f"I-{ent.type}"
    return tags


def decode_bio_to_entities(
    text: str,
    tokens: list[Token],
    tags: list[str],
    confidences: list[float] | None = None,
) -> list[PredictedEntity]:
    """Decode a BIO tag sequence back into character-offset PredictedEntity spans."""
    entities: list[PredictedEntity] = []
    cur_type: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None
    cur_confs: list[float] = []

    def _flush() -> None:
        nonlocal cur_type, cur_start, cur_end, cur_confs
        if cur_type is not None and cur_start is not None and cur_end is not None:
            avg_conf = sum(cur_confs) / len(cur_confs) if cur_confs else 0.75
            entities.append(
                PredictedEntity(
                    type=cur_type,
                    start=cur_start,
                    end=cur_end,
                    text=text[cur_start:cur_end],
                    assertion="present",
                    confidence=round(float(avg_conf), 4),
                )
            )
        cur_type = None
        cur_start = None
        cur_end = None
        cur_confs = []

    for idx, (tok, tag) in enumerate(zip(tokens, tags, strict=True)):
        conf = confidences[idx] if confidences and idx < len(confidences) else 0.75
        if tag.startswith("B-"):
            _flush()
            cur_type = tag[2:]
            cur_start = tok.start
            cur_end = tok.end
            cur_confs = [conf]
        elif tag.startswith("I-"):
            etype = tag[2:]
            if cur_type == etype and cur_end is not None:
                cur_end = tok.end
                cur_confs.append(conf)
            else:
                _flush()
                cur_type = etype
                cur_start = tok.start
                cur_end = tok.end
                cur_confs = [conf]
        else:
            _flush()
    _flush()
    return entities
