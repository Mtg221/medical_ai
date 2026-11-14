"""Regression tests for synthetic corpus generator integrity (Phase 11 Step 1)."""

from __future__ import annotations

from medresearch_platform.approaches.tokenizer import tokenize_with_offsets
from medresearch_platform.data.synthetic_v2 import generate_synthetic_corpus
from scripts.audit_corpus import _find_adjacent_duplicate_ngrams, _find_doubled_spans


def test_generated_corpus_has_no_doubled_text_or_offset_errors() -> None:
    docs = generate_synthetic_corpus(en_count=200, fr_count=200, seed=42)
    assert len(docs) == 400

    for doc in docs:
        # 1. No doubled spans or adjacent duplicate n-grams
        assert _find_doubled_spans(doc) == [], f"Doubled spans in {doc.doc_id}"
        assert _find_adjacent_duplicate_ngrams(doc.text) == [], f"Duplicate n-gram in {doc.doc_id}"

        # 2. 100% offset integrity
        ent_ids = {e.id for e in doc.entities}
        for ent in doc.entities:
            assert doc.text[ent.start : ent.end] == ent.text
        for tmx in doc.timex:
            assert doc.text[tmx.start : tmx.end] == tmx.text
            if tmx.type in ("DATE", "RELATIVE", "DURATION"):
                assert tmx.normalized is not None and len(tmx.normalized) > 0
        for rel in doc.relations:
            assert rel.head in ent_ids and rel.tail in ent_ids

        # 3. Every gold entity & TIMEX boundary aligns with token boundaries
        tokens = tokenize_with_offsets(doc.text)
        starts = {t.start for t in tokens}
        ends = {t.end for t in tokens}
        for ent in doc.entities:
            assert ent.start in starts, f"Entity start {ent} not token-aligned in {doc.doc_id}"
            assert ent.end in ends, f"Entity end {ent} not token-aligned in {doc.doc_id}"
        for tmx in doc.timex:
            assert tmx.start in starts, f"TIMEX start {tmx} not token-aligned in {doc.doc_id}"
            assert tmx.end in ends, f"TIMEX end {tmx} not token-aligned in {doc.doc_id}"
