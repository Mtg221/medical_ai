"""Tests for general EN/FR TIMEX normalization and Stage 5 out-of-fold reranker stacking."""

from __future__ import annotations

from pathlib import Path

import pytest

from medresearch_platform.approaches.tokenizer import TrainOnlyVocabulary
from medresearch_platform.data.schema import AnnotatedDocument, EntitySpan
from medresearch_platform.data.synthetic_v3 import generate_v3_corpus_splits
from medresearch_platform.eval_harness.metrics import PredictedEntity
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.hybrid.stage2_ner import CandidateEntity
from medresearch_platform.hybrid.stage4_temporal import normalize_timex
from medresearch_platform.hybrid.stage5_integration import TransformerCandidateReranker


@pytest.mark.parametrize(
    ("ttype", "text", "anchor", "expected"),
    [
        # Fresh EN/FR dates not in OUT_OF_FORMAT_TIMEX
        ("DATE", "Nov 21st, '23", "2025-03-15", "2023-11-21"),
        ("DATE", "le 1er avril 2024", "2025-03-15", "2024-04-01"),
        ("DATE", "fin janvier 2024", "2025-03-15", "2024-01-31"),
        ("DATE", "fin février 2024", "2025-03-15", "2024-02-29"),  # leap year
        ("DATE", "début octobre 2024", "2025-03-15", "2024-10-01"),
        ("DATE", "mid-August 2024", "2025-03-15", "2024-08-15"),
        ("DATE", "mardi prochain", "2025-03-15", "2025-03-18"),
        ("DATE", "last Monday", "2025-03-15", "2025-03-10"),
        ("DATE", "demain matin", "2025-03-15", "2025-03-16"),
        ("DATE", "four weeks prior", "2025-03-15", "2025-02-15"),
        ("DATE", "le mois dernier", "2025-03-31", "2025-02-28"),
        ("DATE", "22-Jun-24", "2025-03-15", "2024-06-22"),
        # Fresh EN/FR durations
        ("DURATION", "pendant trois semaines", "2025-03-15", "P3W"),
        ("DURATION", "x 14d", "2025-03-15", "P14D"),
        ("DURATION", "five hrs", "2025-03-15", "PT5H"),
        ("DURATION", "pd 10j", "2025-03-15", "P10D"),
        ("DURATION", "for 2 yrs", "2025-03-15", "P2Y"),
        # Fresh EN/FR frequencies
        ("FREQUENCY", "q6h", "2025-03-15", "PT6H"),
        ("FREQUENCY", "4x/j", "2025-03-15", "QID"),
        ("FREQUENCY", "au coucher", "2025-03-15", "QHS"),
        ("FREQUENCY", "hebdomadaire", "2025-03-15", "P1W"),
        ("FREQUENCY", "1 fois par semaine", "2025-03-15", "P1W"),
    ],
)
def test_general_en_fr_timex_normalization(
    ttype: str, text: str, anchor: str, expected: str
) -> None:
    assert normalize_timex(text, ttype, anchor) == expected


def test_stage5_reranker_learns_to_filter_false_positives_and_calibrates_threshold() -> None:
    docs: list[AnnotatedDocument] = []
    cand_pairs: list[tuple[AnnotatedDocument, list[CandidateEntity]]] = []
    for i in range(12):
        text = "Diagnosis: pneumonia treated with amoxicillin; random noise token xyz."
        doc = AnnotatedDocument(
            doc_id=f"d{i}",
            patient_id=f"p{i}",
            language="en",
            doc_type="consultation",
            document_date="2025-03-15",
            source="unit_test",
            text=text,
            entities=[
                EntitySpan(id="e1", type="DIAGNOSIS", start=11, end=20, text="pneumonia"),
                EntitySpan(id="e2", type="MEDICATION", start=34, end=45, text="amoxicillin"),
            ],
        )
        cands = [
            CandidateEntity(
                entity=PredictedEntity(type="DIAGNOSIS", start=11, end=20, text="pneumonia", confidence=0.88),
                source_stage="bilstm_crf",
            ),
            CandidateEntity(
                entity=PredictedEntity(type="MEDICATION", start=34, end=45, text="amoxicillin", confidence=0.90),
                source_stage="bilstm_crf",
            ),
            # Deliberate false positive from noisy span
            CandidateEntity(
                entity=PredictedEntity(type="SYMPTOM", start=66, end=69, text="xyz", confidence=0.35),
                source_stage="bilstm_crf",
            ),
        ]
        docs.append(doc)
        cand_pairs.append((doc, cands))

    vocab = TrainOnlyVocabulary(max_size=1000, min_freq=1).fit(docs)
    reranker = TransformerCandidateReranker(vocab_size=len(vocab), d_model=24, seed=7)
    reranker.fit_on_candidates(cand_pairs, vocab=vocab, epochs=15, seed=7)

    test_doc, test_cands = cand_pairs[0]
    scores = reranker.score_candidates(test_cands, doc_text=test_doc.text, vocab=vocab)
    assert scores[0] > scores[2]
    assert scores[1] > scores[2]

    kept = reranker.rerank(test_cands, doc_text=test_doc.text, vocab=vocab)
    kept_texts = {c.entity.text for c in kept}
    assert "pneumonia" in kept_texts
    assert "amoxicillin" in kept_texts
    assert "xyz" not in kept_texts


def test_stage5_oof_stacking_and_threshold_persistence(tmp_path: Path) -> None:
    splits = generate_v3_corpus_splits(
        split_counts_per_lang={"train": 14, "val": 4, "test_iid": 2, "test_unseen_terms": 2, "test_unseen_templates": 2},
        seed=11,
    )
    pipe = HybridPipeline(
        seed=11,
        cfg={
            "epochs": 2,
            "aux_epochs": 2,
            "max_train_docs": 24,
            "use_transformer_fusion": True,
            "stage5_oof_folds": 2,
            "stage5_oof_max_docs": 12,
        },
    ).fit(splits["train"])
    saved_dir = pipe.save(tmp_path / "hybrid_oof_model")
    loaded = HybridPipeline.load(saved_dir)
    assert loaded.reranker.threshold == pytest.approx(pipe.reranker.threshold)
    pred, contract = loaded.extract_document(splits["val"][0], use_transformer_fusion=True)
    assert pred.doc_id == splits["val"][0].doc_id
    assert contract["schema_version"] == "1.0"
