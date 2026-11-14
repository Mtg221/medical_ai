"""Finite-difference gradient checks (float64, rel_err < 1e-4) on all 6 evaluated neural models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from medresearch_platform.approaches.neural_seq import (
    CHAR_FEAT_DIM,
    NUM_TAGS,
    NeuralBiLSTMCRFTagger,
    NeuralBiLSTMTagger,
    NeuralTransformerTagger,
)
from medresearch_platform.eval_harness.metrics import PredictedEntity
from medresearch_platform.hybrid.stage2_ner import CandidateEntity
from medresearch_platform.hybrid.stage3_context_relations import (
    BiLSTMAssertionClassifier,
    PiecewiseCNNRelationClassifier,
)
from medresearch_platform.hybrid.stage5_integration import TransformerCandidateReranker


def _check_params_finite_diff(
    params: dict[str, np.ndarray],
    loss_fn: Callable[[], float],
    grads: dict[str, np.ndarray],
    *,
    eps: float = 1e-6,
    max_rel_err: float = 1e-4,
    samples_per_param: int = 3,
    seed: int = 123,
) -> None:
    rng = np.random.default_rng(seed)
    for name, arr in params.items():
        g_arr = grads[name]
        assert g_arr.shape == arr.shape, f"Gradient shape mismatch for {name}"
        flat = arr.reshape(-1)
        g_flat = g_arr.reshape(-1)
        # Prefer checking non-zero analytic gradients as well as random coordinates
        nonzero = np.where(np.abs(g_flat) > 1e-8)[0]
        if len(nonzero) > 0:
            idxs = rng.choice(nonzero, size=min(samples_per_param, len(nonzero)), replace=False)
        else:
            idxs = rng.choice(len(flat), size=min(samples_per_param, len(flat)), replace=False)
        for idx in idxs:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            loss_plus = float(loss_fn())
            flat[idx] = orig - eps
            loss_minus = float(loss_fn())
            flat[idx] = orig
            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            ana_grad = float(g_flat[idx])
            denom = max(1e-7, abs(num_grad) + abs(ana_grad))
            rel_err = abs(num_grad - ana_grad) / denom
            assert rel_err < max_rel_err, (
                f"Gradient mismatch in {name}[{idx}]: analytic={ana_grad:.8e}, "
                f"numeric={num_grad:.8e}, rel_err={rel_err:.8e}"
            )


def test_bilstm_tagger_gradients() -> None:
    rng = np.random.default_rng(7)
    model = NeuralBiLSTMTagger(vocab_size=12, emb_dim=6, hidden_dim=5, seed=7)
    w_ids = np.array([2, 5, 3, 8, 4], dtype=np.int64)
    c_feats = rng.normal(0.0, 0.5, size=(5, CHAR_FEAT_DIM))
    y_ids = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    cw = np.ones(NUM_TAGS, dtype=np.float64)
    _, grads = model.loss_and_grads(w_ids, c_feats, y_ids, cw)
    _check_params_finite_diff(
        model.params,
        lambda: model.loss_and_grads(w_ids, c_feats, y_ids, cw)[0],
        grads,
    )


def test_bilstm_crf_tagger_gradients() -> None:
    rng = np.random.default_rng(11)
    model = NeuralBiLSTMCRFTagger(vocab_size=12, emb_dim=6, hidden_dim=5, seed=11)
    w_ids = np.array([2, 5, 3, 8, 4], dtype=np.int64)
    c_feats = rng.normal(0.0, 0.5, size=(5, CHAR_FEAT_DIM))
    y_ids = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    cw = np.ones(NUM_TAGS, dtype=np.float64)
    _, grads = model.loss_and_grads(w_ids, c_feats, y_ids, cw)
    _check_params_finite_diff(
        model.params,
        lambda: model.loss_and_grads(w_ids, c_feats, y_ids, cw)[0],
        grads,
    )


def test_transformer_tagger_gradients() -> None:
    rng = np.random.default_rng(19)
    model = NeuralTransformerTagger(
        vocab_size=12, d_model=8, num_layers=2, num_heads=4, bpe_vocab_size=10, max_len=16, seed=19
    )
    w_ids = np.array([2, 5, 3, 8, 4, 0], dtype=np.int64)
    bpe_ids = [[2, 3], [4], [5, 6, 7], [2, 8], [9], [0]]
    c_feats = rng.normal(0.0, 0.5, size=(6, CHAR_FEAT_DIM))
    y_ids = np.array([0, 1, 2, 0, 3, 0], dtype=np.int64)
    valid_mask = np.array([1, 1, 1, 1, 1, 0], dtype=np.float64)
    cw = np.ones(NUM_TAGS, dtype=np.float64)
    _, grads = model.loss_and_grads(
        w_ids, c_feats, y_ids, cw, bpe_ids=bpe_ids, valid_mask=valid_mask
    )
    _check_params_finite_diff(
        model.params,
        lambda: model.loss_and_grads(
            w_ids, c_feats, y_ids, cw, bpe_ids=bpe_ids, valid_mask=valid_mask
        )[0],
        grads,
    )


def test_assertion_bilstm_gradients() -> None:
    rng = np.random.default_rng(23)
    model = BiLSTMAssertionClassifier(vocab_size=12, emb_dim=6, hidden_dim=5, seed=23)
    w_ids = np.array([2, 4, 6, 3, 5], dtype=np.int64)
    flags = rng.normal(0.0, 0.5, size=(5, 2))
    _, grads = model.loss_and_grads(w_ids, flags, pool_idx=2, target_id=1, weight=1.5)
    _check_params_finite_diff(
        model.params,
        lambda: model.loss_and_grads(w_ids, flags, pool_idx=2, target_id=1, weight=1.5)[0],
        grads,
    )


def test_relation_cnn_gradients() -> None:
    model = PiecewiseCNNRelationClassifier(vocab_size=12, emb_dim=6, pos_dim=3, n_filters=5, seed=29)
    w_ids = np.array([2, 4, 6, 3, 5, 7], dtype=np.int64)
    p1 = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
    p2 = np.array([6, 7, 8, 9, 10, 11], dtype=np.int64)
    _, grads = model.loss_and_grads(w_ids, p1, p2, target_id=2, weight=1.2)
    _check_params_finite_diff(
        model.params,
        lambda: model.loss_and_grads(w_ids, p1, p2, target_id=2, weight=1.2)[0],
        grads,
    )


def test_stage5_transformer_reranker_gradients() -> None:
    model = TransformerCandidateReranker(vocab_size=20, emb_dim=6, d_model=8, seed=31)
    cands = [
        CandidateEntity(PredictedEntity("DIAGNOSIS", 0, 6, "asthma", "present", 0.8), "bilstm_crf"),
        CandidateEntity(PredictedEntity("MEDICATION", 10, 19, "metformin", "present", 0.9), "bilstm_crf+rule_layer"),
        CandidateEntity(PredictedEntity("SYMPTOM", 24, 29, "fever", "present", 0.6), "rule_layer"),
    ]
    labels = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    doc_text = "asthma on metformin and fever"
    _, grads = model.loss_and_grads(cands, labels, doc_text=doc_text)
    _check_params_finite_diff(
        model.params,
        lambda: model.loss_and_grads(cands, labels, doc_text=doc_text)[0],
        grads,
    )
