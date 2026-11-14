"""Tests for Phase 11 Step 4: Stage 3 (F4) and Stage 5 text-reading Transformer fusion (F5)."""

from __future__ import annotations

from medresearch_platform.approaches.tokenizer import TrainOnlyVocabulary
from medresearch_platform.data.synthetic_v3 import generate_v3_corpus_splits
from medresearch_platform.eval_harness.metrics import PredictedEntity
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.hybrid.stage2_ner import CandidateEntity
from medresearch_platform.hybrid.stage3_context_relations import _REL_ALLOWED_PAIRS
from medresearch_platform.hybrid.stage5_integration import TransformerCandidateReranker


def test_stage3_allowed_pairs_cover_v3_relations() -> None:
    expected = {
        ("MEDICATION", "DOSAGE"),
        ("MEDICATION", "FREQUENCY"),
        ("MEDICATION", "ROUTE"),
        ("MEDICATION", "DURATION"),
        ("LAB_TEST", "LAB_VALUE"),
        ("LAB_TEST", "LAB_UNIT"),
        ("DIAGNOSIS", "PROCEDURE"),
        ("DIAGNOSIS", "ANATOMY"),
    }
    assert expected.issubset(_REL_ALLOWED_PAIRS)


def test_stage5_reranker_reads_text_and_context() -> None:
    splits = generate_v3_corpus_splits(
        split_counts_per_lang={"train": 10, "val": 4, "test_iid": 2, "test_unseen_terms": 2, "test_unseen_templates": 2},
        seed=42,
    )
    vocab = TrainOnlyVocabulary(max_size=2000, min_freq=1).fit(splits["train"])
    reranker = TransformerCandidateReranker(vocab_size=len(vocab), d_model=24, seed=42)
    assert "E" in reranker.params

    cand_a = [
        CandidateEntity(
            entity=PredictedEntity(type="DIAGNOSIS", start=20, end=29, text="pneumonia", confidence=0.6),
            source_stage="bilstm_crf",
        )
    ]
    text_a = "Diagnosis confirmed: pneumonia treated with antibiotics."
    text_b = "Unrelated random xyz pneumonia placeholder string here."
    scores_a = reranker.score_candidates(cand_a, doc_text=text_a, vocab=vocab)
    scores_b = reranker.score_candidates(cand_a, doc_text=text_b, vocab=vocab)
    assert scores_a[0] != scores_b[0]


def test_hybrid_pipeline_end_to_end_on_v3() -> None:
    splits = generate_v3_corpus_splits(
        split_counts_per_lang={"train": 20, "val": 6, "test_iid": 2, "test_unseen_terms": 2, "test_unseen_templates": 2},
        seed=42,
    )
    pipe = HybridPipeline(seed=42, cfg={"epochs": 3, "aux_epochs": 3, "max_train_docs": 30})
    pipe.fit(splits["train"])
    pred, contract = pipe.extract_document(splits["val"][0])
    assert pred.doc_id == splits["val"][0].doc_id
    assert contract["schema_version"] == "1.0"
