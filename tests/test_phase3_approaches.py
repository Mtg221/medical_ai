"""Tests for Phase 3 single approaches: tokenizer, ConText/NegEx, overfit-20-sentences, and smoke runs."""

from __future__ import annotations

import numpy as np

from medresearch_platform.approaches.classical_ml import (
    predict_classical_crf,
    predict_classical_ml,
)
from medresearch_platform.approaches.context_negex import classify_assertion_context
from medresearch_platform.approaches.neural_seq import (
    NeuralBiLSTMCRFTagger,
    NeuralBiLSTMTagger,
    NeuralTransformerTagger,
    _AdamW,
    _compute_class_weights,
    _encode_docs,
    predict_bilstm,
    predict_bilstm_crf,
    predict_transformer,
)
from medresearch_platform.approaches.rule_based import predict_rule_based
from medresearch_platform.approaches.tokenizer import (
    TrainOnlyVocabulary,
    align_entities_to_bio,
    decode_bio_to_entities,
    tokenize_with_offsets,
)
from medresearch_platform.data.synthetic_v2 import generate_synthetic_corpus
from medresearch_platform.eval_harness.metrics import Evaluator


def test_tokenizer_and_bio_roundtrip() -> None:
    docs = generate_synthetic_corpus(en_count=4, fr_count=4, seed=42)
    for d in docs:
        toks = tokenize_with_offsets(d.text)
        tags = align_entities_to_bio(toks, d.entities)
        decoded = decode_bio_to_entities(d.text, toks, tags)
        assert len(decoded) >= len(d.entities) - 1
        for ent in decoded:
            assert d.text[ent.start : ent.end] == ent.text


def test_context_negex_en_and_fr() -> None:
    en_txt = "Pulmonary embolism ruled out. Suspected pneumonia. No increase in fever. Family history of asthma."
    assert classify_assertion_context(en_txt, 0, 18, "en") == "absent"
    assert classify_assertion_context(en_txt, 40, 49, "en") == "possible"
    assert classify_assertion_context(en_txt, 66, 71, "en") == "present"  # pseudo-negation
    assert classify_assertion_context(en_txt, 91, 97, "en") == "family"

    fr_txt = "Absence d'embolie pulmonaire. Suspicion de pneumopathie. Consulter si apparition de fièvre."
    assert classify_assertion_context(fr_txt, 10, 28, "fr") == "absent"
    assert classify_assertion_context(fr_txt, 43, 55, "fr") == "possible"
    assert classify_assertion_context(fr_txt, 84, 90, "fr") == "hypothetical"


def test_neural_models_loss_goes_down_and_overfit_20_docs() -> None:
    docs = generate_synthetic_corpus(en_count=10, fr_count=10, seed=7)
    vocab = TrainOnlyVocabulary().fit(docs)
    seqs = _encode_docs(docs, vocab)
    cw = _compute_class_weights(seqs)
    ev = Evaluator()

    for cls_type in (NeuralBiLSTMTagger, NeuralBiLSTMCRFTagger, NeuralTransformerTagger):
        model = cls_type(len(vocab), seed=42)
        opt = _AdamW(model.params, lr=0.012)
        first_loss = float(np.mean([model.step_sequence(w, c, y, cw, opt) for w, c, y in seqs]))
        for _ in range(18):
            last_loss = float(np.mean([model.step_sequence(w, c, y, cw, opt) for w, c, y in seqs]))
        assert last_loss < first_loss

        # Verify overfit F1 on the 20 training sentences is substantially > 0
        preds = []
        for d in docs:
            toks = tokenize_with_offsets(d.text)
            w_ids = np.array([vocab.encode_token(t.text) for t in toks], dtype=np.int64)
            from medresearch_platform.approaches.neural_seq import _ortho_features

            tags, confs = model.predict(w_ids, _ortho_features(toks))
            ents = decode_bio_to_entities(d.text, toks, tags, confs)
            from medresearch_platform.eval_harness.metrics import DocumentPrediction

            preds.append(DocumentPrediction(d.doc_id, d.language, d.doc_type, entities=ents))
        f1 = ev.evaluate(docs, preds)["strict_span"]["f1"]
        assert f1 >= 0.45, f"{cls_type.__name__} failed to learn on 20 docs (F1={f1})"


def test_all_five_approaches_smoke_eval() -> None:
    train_docs = generate_synthetic_corpus(en_count=18, fr_count=18, seed=11)
    val_docs = generate_synthetic_corpus(en_count=6, fr_count=6, seed=99)
    ev = Evaluator()
    cfg = {"epochs": 6, "lr": 0.02, "hidden_dim": 24, "max_train_docs": 36, "max_iterations": 30}

    for fn in (
        predict_rule_based,
        predict_classical_ml,
        predict_classical_crf,
        predict_bilstm,
        predict_bilstm_crf,
        predict_transformer,
    ):
        preds = fn(train_docs, val_docs, 42, cfg)
        res = ev.evaluate(val_docs, preds)
        assert res["strict_span"]["f1"] > 0.15, f"{fn.__name__} F1 too low: {res['strict_span']}"
        assert res["per_language"]["en"]["f1"] > 0.15
        assert res["per_language"]["fr"]["f1"] > 0.15
