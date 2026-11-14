"""Regression tests for 11B Step 1: hidden doc_type invariance (G2) and doc_type learnability (G1)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.runner import APPROACH_REGISTRY, _load_jsonl

ALL_APPROACHES: tuple[str, ...] = (
    "dummy",
    "rule_based",
    "classical_ml",
    "classical_crf",
    "bilstm",
    "bilstm_crf",
    "transformer",
    "hybrid",
)


def _hide_doc_type(docs: list[AnnotatedDocument]) -> list[AnnotatedDocument]:
    return [replace(d, doc_type="__hidden__") for d in docs]


@pytest.mark.parametrize("approach_name", ALL_APPROACHES)
def test_no_approach_reads_gold_doc_type(approach_name: str) -> None:
    """Blinding eval doc_type to '__hidden__' must not change predictions or leak '__hidden__'."""
    v3_train = _load_jsonl(Path("data/processed/v3/train.jsonl"))[:24]
    v3_val = _load_jsonl(Path("data/processed/v3/val.jsonl"))[:6]
    hidden_val = _hide_doc_type(v3_val)

    cfg = {"epochs": 2, "aux_epochs": 2, "max_train_docs": 24, "max_iterations": 15}
    predictor = APPROACH_REGISTRY[approach_name]

    preds_gold = predictor(v3_train, v3_val, 1, cfg)
    preds_hidden = predictor(v3_train, hidden_val, 1, cfg)

    assert len(preds_gold) == len(preds_hidden) == len(v3_val)
    for pg, ph in zip(preds_gold, preds_hidden, strict=True):
        assert ph.predicted_doc_type != "__hidden__", (
            f"{approach_name} leaked hidden gold doc_type into predicted_doc_type!"
        )
        assert pg.predicted_doc_type == ph.predicted_doc_type, (
            f"{approach_name} changed predicted_doc_type when gold doc_type was hidden "
            f"({pg.predicted_doc_type} vs {ph.predicted_doc_type})"
        )
        assert [(e.type, e.start, e.end, e.assertion) for e in pg.entities] == [
            (e.type, e.start, e.end, e.assertion) for e in ph.entities
        ]


def test_v3_1_doc_type_is_learnable_above_80_pct() -> None:
    """TF-IDF + LinearSVC trained on V3.1 train must achieve > 0.80 doc-type accuracy on val."""
    v3_1_dir = Path("data/processed/v3_1")
    assert v3_1_dir.exists(), "SYNTH-MED-V3.1 corpus (data/processed/v3_1) has not been built yet"

    train_docs = _load_jsonl(v3_1_dir / "train.jsonl")
    val_docs = _load_jsonl(v3_1_dir / "val.jsonl")

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=4000, sublinear_tf=True)
    x_tr = vec.fit_transform([d.text for d in train_docs])
    x_va = vec.transform([d.text for d in val_docs])

    clf = LinearSVC(random_state=42, class_weight="balanced")
    clf.fit(x_tr, [d.doc_type for d in train_docs])
    preds = clf.predict(x_va)
    acc = sum(int(p == d.doc_type) for p, d in zip(preds, val_docs, strict=True)) / len(val_docs)
    assert acc > 0.80, f"Expected TF-IDF+LinearSVC doc_type val accuracy > 0.80, got {acc:.4f}"
