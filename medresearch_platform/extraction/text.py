"""Shared tokenization, BIO tagging, vocabulary, and the shared deterministic assertion module.

The assertion module (ConText-style cue lexicons) is deliberately NOT one of the compared approaches:
all six approaches are compared on entity-span extraction only, and every approach reuses this one
module to fill the negation / assertion / temporal fields of the common output contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from medresearch_platform.ai.nlp_model import (
    _FAMILY_CUES,
    _HISTORICAL_CUES,
    _HYPOTHETICAL_CUES,
    _NEGATION_CUES,
    _UNCERTAINTY_CUES,
)

ENTITY_TYPES: tuple[str, ...] = (
    "SYMPTOM",
    "DIAGNOSIS",
    "MEDICATION",
    "DOSAGE",
    "FREQUENCY",
    "ROUTE",
    "LAB_TEST",
    "LAB_VALUE",
    "LAB_UNIT",
)
TAGS: tuple[str, ...] = ("O", *[f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")])
TAG_TO_ID: dict[str, int] = {t: i for i, t in enumerate(TAGS)}

_TOKEN_RE = re.compile(r"\w+(?:[./'’-]\w+)*|[^\w\s]", re.UNICODE)
_DIGIT_RE = re.compile(r"\d")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def normalize_token(token: str) -> str:
    """Lower-case and map every digit to '0' so numbers share one shape."""
    return _DIGIT_RE.sub("0", token.lower())


def spans_to_bio(tokens: list[Token], spans: list[tuple[int, int, str]]) -> list[str]:
    tags = ["O"] * len(tokens)
    for start, end, etype in spans:
        first = True
        for i, tok in enumerate(tokens):
            if tok.start < end and tok.end > start:
                tags[i] = f"{'B' if first else 'I'}-{etype}"
                first = False
    return tags


def bio_to_spans(tokens: list[Token], tags: list[str]) -> list[tuple[int, int, str, int, int]]:
    """Decode BIO tags to (char_start, char_end, type, first_token_idx, last_token_idx); repairs stray I- tags."""
    spans: list[tuple[int, int, str, int, int]] = []
    cur: list[int] | None = None
    cur_type = ""
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur is not None:
                spans.append((tokens[cur[0]].start, tokens[cur[-1]].end, cur_type, cur[0], cur[-1]))
                cur = None
            continue
        prefix, etype = tag.split("-", 1)
        if prefix == "I" and cur is not None and etype == cur_type:
            cur.append(i)
            continue
        if cur is not None:
            spans.append((tokens[cur[0]].start, tokens[cur[-1]].end, cur_type, cur[0], cur[-1]))
        cur, cur_type = [i], etype
    if cur is not None:
        spans.append((tokens[cur[0]].start, tokens[cur[-1]].end, cur_type, cur[0], cur[-1]))
    return spans


class Vocabulary:
    """Custom vocabulary built from training tokens only; id 0 is the unknown token."""

    UNK = 0

    def __init__(self, words: list[str]) -> None:
        self.itos = ["<unk>", *words]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists: list[list[str]]) -> Vocabulary:
        seen = sorted({normalize_token(t) for toks in token_lists for t in toks})
        return cls(seen)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(normalize_token(t), self.UNK) for t in tokens]

    def __len__(self) -> int:
        return len(self.itos)


def assess_assertion(sentence_text: str, start: int, end: int) -> tuple[str, str, str]:
    """Return (assertion_status, negation_status, temporal_status) for the span using cue lexicons."""
    left = sentence_text[:start].lower()
    right = sentence_text[end : end + 25].lower()
    window = f"{left} {sentence_text[start:end].lower()} {right}"
    if any(c in left for c in _FAMILY_CUES):
        return "FAMILY_HISTORY", "AFFIRMED", "HISTORICAL"
    temporal = "HISTORICAL" if any(c in window for c in _HISTORICAL_CUES) else "CURRENT"
    if any(c in left for c in _HYPOTHETICAL_CUES):
        return "HYPOTHETICAL", "UNKNOWN", temporal
    if any(c in left or c in right for c in _UNCERTAINTY_CUES):
        return "UNCERTAIN", "UNKNOWN", temporal
    if any(c in left[-30:] or c in right for c in _NEGATION_CUES):
        return "NEGATED", "NEGATED", temporal
    return "AFFIRMED", "AFFIRMED", temporal
