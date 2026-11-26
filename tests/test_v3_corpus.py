"""Tests for SYNTH-MED-V3 lexicon scale, template scale, integrity, and split isolation."""

from __future__ import annotations

from collections import defaultdict

from medresearch_platform.approaches.tokenizer import tokenize_with_offsets
from medresearch_platform.data.synthetic_v3 import generate_v3_corpus_splits, summarize_v3_resources
from medresearch_platform.data.templates_v3 import SECTIONS, build_all_templates
from scripts.audit_corpus import (
    _SENT_SPLIT_RE,
    _find_adjacent_duplicate_ngrams,
    _find_doubled_spans,
    _mask_digits,
)


def test_v3_lexicon_and_template_minimums() -> None:
    stats = summarize_v3_resources()
    assert stats["diagnoses_en"] >= 80
    assert stats["diagnoses_fr"] >= 80
    assert stats["symptoms_en"] >= 60
    assert stats["symptoms_fr"] >= 60
    assert stats["medications_en"] >= 80
    assert stats["medications_fr"] >= 80
    assert stats["labs_en"] >= 50
    assert stats["labs_fr"] >= 50
    assert stats["procedures_en"] >= 40
    assert stats["procedures_fr"] >= 40
    assert stats["anatomy_en"] >= 30
    assert stats["anatomy_fr"] >= 30
    assert stats["total_templates"] >= 600

    tpls = build_all_templates()
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for t in tpls:
        counts[(t.language, t.section)] += 1
    for lang in ("en", "fr"):
        for sec in SECTIONS:
            assert counts[(lang, sec)] >= 40, f"{lang}:{sec} has {counts[(lang, sec)]} < 40"


def test_v3_splits_integrity_and_generalization_isolation() -> None:
    splits = generate_v3_corpus_splits(
        split_counts_per_lang={
            "train": 60,
            "val": 20,
            "test_iid": 20,
            "test_unseen_terms": 20,
            "test_unseen_templates": 20,
        },
        seed=42,
    )
    assert set(splits.keys()) == {"train", "val", "test_iid", "test_unseen_terms", "test_unseen_templates"}

    lexicon_types = {"DIAGNOSIS", "SYMPTOM", "MEDICATION", "LAB_TEST", "PROCEDURE", "ANATOMY"}
    train_terms_by_type: dict[str, set[str]] = defaultdict(set)
    train_sents: set[str] = set()

    for sp_name, docs in splits.items():
        for doc in docs:
            assert _find_doubled_spans(doc) == [], f"Doubled span in {doc.doc_id}"
            assert _find_adjacent_duplicate_ngrams(doc.text) == [], f"Dup ngram in {doc.doc_id}"
            for e in doc.entities:
                assert doc.text[e.start : e.end] == e.text
            for tx in doc.timex:
                assert doc.text[tx.start : tx.end] == tx.text
                assert tx.normalized is not None and len(tx.normalized) > 0

            toks = tokenize_with_offsets(doc.text)
            starts = {t.start for t in toks}
            ends = {t.end for t in toks}
            for e in doc.entities:
                assert e.start in starts and e.end in ends, f"Unaligned {e.text!r} in {doc.doc_id}"

            if sp_name == "train":
                for e in doc.entities:
                    if e.type in lexicon_types:
                        train_terms_by_type[e.type].add(e.text.lower())
                for s in _SENT_SPLIT_RE.split(doc.text):
                    ms = _mask_digits(s)
                    if ms:
                        train_sents.add(ms)

    # Verify unseen_terms has 0% concept/surface overlap with train on clinical lexicon categories
    for doc in splits["test_unseen_terms"]:
        for e in doc.entities:
            if e.type in lexicon_types:
                assert e.text.lower() not in train_terms_by_type[e.type], (
                    f"Leaked term {e.text!r} ({e.type}) into test_unseen_terms"
                )

    # Verify unseen_templates has very low masked-sentence overlap with train (< 15%)
    unseen_tpl_sents = [
        _mask_digits(s)
        for doc in splits["test_unseen_templates"]
        for s in _SENT_SPLIT_RE.split(doc.text)
        if _mask_digits(s)
    ]
    overlap = sum(1 for s in unseen_tpl_sents if s in train_sents) / max(1, len(unseen_tpl_sents))
    assert overlap < 0.15, f"Unseen templates masked sentence overlap too high: {overlap:.3f}"
