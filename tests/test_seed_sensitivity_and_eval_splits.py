"""Tests for Phase 11 Step 3: seed sensitivity (F3), char morphology, and v3 eval splits."""

from __future__ import annotations

import numpy as np

from medresearch_platform.approaches.neural_seq import CHAR_FEAT_DIM, _ortho_features, predict_bilstm_crf
from medresearch_platform.approaches.tokenizer import tokenize_with_offsets
from medresearch_platform.data.synthetic_v3 import generate_v3_corpus_splits
from medresearch_platform.eval_harness.metrics import Evaluator
from medresearch_platform.eval_harness.runner import VALID_EVAL_SPLITS


def test_runner_supports_v3_eval_splits() -> None:
    for sp in ("val", "test", "test_iid", "test_unseen_terms", "test_unseen_templates"):
        assert sp in VALID_EVAL_SPLITS


def test_char_ortho_features_capture_medical_morphology() -> None:
    assert CHAR_FEAT_DIM >= 32
    t1 = tokenize_with_offsets("atorvastatin")
    t2 = tokenize_with_offsets("simvastatin")
    t3 = tokenize_with_offsets("120")
    f1 = _ortho_features(t1)
    f2 = _ortho_features(t2)
    f3 = _ortho_features(t3)
    assert f1.shape == (1, CHAR_FEAT_DIM)
    # Shared -statin suffix produces high cosine similarity vs numeric token
    sim_statin = float(np.dot(f1[0], f2[0]) / (np.linalg.norm(f1[0]) * np.linalg.norm(f2[0]) + 1e-12))
    sim_num = float(np.dot(f1[0], f3[0]) / (np.linalg.norm(f1[0]) * np.linalg.norm(f3[0]) + 1e-12))
    assert sim_statin > sim_num


def test_bilstm_crf_seeds_produce_distinct_predictions_on_v3() -> None:
    splits = generate_v3_corpus_splits(
        split_counts_per_lang={"train": 16, "val": 8, "test_iid": 4, "test_unseen_terms": 4, "test_unseen_templates": 4},
        seed=42,
    )
    cfg = {"epochs": 4, "lr": 0.018, "hidden_dim": 16, "max_train_docs": 32, "patience": 2, "dropout": 0.15}
    p1 = predict_bilstm_crf(splits["train"], splits["val"], seed=1, cfg=cfg)
    p2 = predict_bilstm_crf(splits["train"], splits["val"], seed=2, cfg=cfg)

    ev = Evaluator()
    f1_1 = ev.evaluate(splits["val"], p1)["strict_span"]["f1"]
    f1_2 = ev.evaluate(splits["val"], p2)["strict_span"]["f1"]
    # Even if F1 is close, confidences / predicted spans must differ across seeds
    confs_1 = [e.confidence for d in p1 for e in d.entities]
    confs_2 = [e.confidence for d in p2 for e in d.entities]
    assert confs_1 != confs_2 or f1_1 != f1_2
