"""Tests for Hybrid v2 (CRF backoff + ConText/NegEx assertion + zero-overhead ablations)."""

from __future__ import annotations

from medresearch_platform.data.synthetic_v3 import generate_v3_flat_list
from medresearch_platform.hybrid.pipeline import HybridPipeline


def test_hybrid_v2_crf_ensemble_and_context_negex() -> None:
    docs = generate_v3_flat_list(split_counts_per_lang={"train": 12}, seed=7, version="v3_1")
    train_docs = docs[:18]
    eval_docs = docs[18:]

    pipe = HybridPipeline(
        seed=1,
        cfg={
            "hidden_dim": 16,
            "epochs": 3,
            "aux_epochs": 2,
            "use_crf_ensemble": True,
            "use_assertion_bilstm": False,
            "use_transformer_fusion": False,
        },
    ).fit(train_docs)

    assert pipe.stage2.crf_model is not None
    assert pipe.assertion_model is None

    pred, contract = pipe.extract_document(
        eval_docs[0],
        use_assertion_bilstm=False,
        use_transformer_fusion=False,
    )
    assert pred.doc_id == eval_docs[0].doc_id
    assert isinstance(contract["entities"], list)
    assert len(pred.entities) > 0
