"""Tests for the six independent medical extraction approaches and common output contract."""

from __future__ import annotations

import numpy as np

from medresearch_platform.documents.synthetic_generator import SYNTHETIC_CLINICAL_CORPUS
from extras.legacy.extraction.benchmark import resolve_gold_spans, run_six_approach_benchmark
from extras.legacy.extraction.bilstm import BiLSTMExtractor
from extras.legacy.extraction.bilstm_crf import BiLSTMCRFExtractor, build_bio_transition_mask
from extras.legacy.extraction.classical_ml import ClassicalMLExtractor
from extras.legacy.extraction.hybrid_multimodal import HybridMultimodalExtractor
from extras.legacy.extraction.rule_based import RuleBasedMedicalExtractor
from medresearch_platform.extraction.text import TAGS
from extras.legacy.extraction.transformer import TransformerExtractor

REQUIRED_CONTRACT_KEYS = {
    "source_record_id",
    "document_id",
    "original_text",
    "extracted_value",
    "entity_type",
    "normalized_concept",
    "char_start",
    "char_end",
    "page_number",
    "region_coordinates",
    "language",
    "negation_status",
    "assertion_status",
    "uncertainty",
    "temporal_information",
    "relationships",
    "confidence_score",
    "model_name",
    "model_version",
    "provenance",
    "review_status",
}


def test_neural_finite_difference_gradient_checks() -> None:
    """Verify hand-derived backprop for BiLSTM, BiLSTM-CRF, and Transformer against central finite differences."""
    bilstm_err = BiLSTMExtractor(seed=7).check_gradients()
    assert bilstm_err, "BiLSTM gradient check returned empty"
    for name, err in bilstm_err.items():
        assert err < 1e-4, f"BiLSTM gradient mismatch for {name}: {err}"

    crf_err = BiLSTMCRFExtractor(seed=7).check_gradients()
    assert crf_err, "BiLSTM-CRF gradient check returned empty"
    for name, err in crf_err.items():
        assert err < 1e-4, f"BiLSTM-CRF gradient mismatch for {name}: {err}"

    tr_err = TransformerExtractor(seed=7).check_gradients()
    assert tr_err, "Transformer gradient check returned empty"
    for name, err in tr_err.items():
        assert err < 1e-4, f"Transformer gradient mismatch for {name}: {err}"


def test_bio_transition_constraints_and_training_convergence() -> None:
    """Verify BIO transition mask and that all trainable extractors reduce loss from random init."""
    trans_mask, start_mask = build_bio_transition_mask(TAGS)
    o_idx = TAGS.index("O")
    b_sym = TAGS.index("B-SYMPTOM")
    i_sym = TAGS.index("I-SYMPTOM")
    i_diag = TAGS.index("I-DIAGNOSIS")
    assert start_mask[i_sym] == 0.0
    assert trans_mask[o_idx, i_sym] == 0.0
    assert trans_mask[b_sym, i_diag] == 0.0
    assert trans_mask[b_sym, i_sym] == 1.0

    train_corpus = [
        (s.raw_text, resolve_gold_spans(s))
        for s in SYNTHETIC_CLINICAL_CORPUS
        if s.split_assignment in {"train", "val"}
    ]

    for cls in (BiLSTMExtractor, BiLSTMCRFExtractor, TransformerExtractor):
        model = cls(epochs=15, seed=42)
        summary = model.fit(train_corpus)
        assert summary["pretrained_weights_used"] is False
        assert isinstance(summary["weights_sha256"], str) and len(summary["weights_sha256"]) == 64
        assert summary["final_loss"] < summary["initial_loss"]


def test_common_output_contract_across_all_six_approaches() -> None:
    """Verify all six approaches emit valid common-contract ExtractedRecord payloads."""
    train_corpus = [
        (s.raw_text, resolve_gold_spans(s))
        for s in SYNTHETIC_CLINICAL_CORPUS
        if s.split_assignment in {"train", "val"}
    ]
    sample_text = (
        "Patient presents with chest pain and fever. "
        "Prescribed metformin 500 mg twice daily orally. Pulmonary embolism ruled out."
    )

    rule_ext = RuleBasedMedicalExtractor()
    classical_ext = ClassicalMLExtractor(random_state=42)
    classical_ext.fit(train_corpus)
    bilstm_ext = BiLSTMExtractor(epochs=12, seed=42)
    bilstm_ext.fit(train_corpus)
    crf_ext = BiLSTMCRFExtractor(epochs=12, seed=42)
    crf_ext.fit(train_corpus)
    tr_ext = TransformerExtractor(epochs=12, seed=42)
    tr_ext.fit(train_corpus)
    hybrid_ext = HybridMultimodalExtractor(
        rule_extractor=rule_ext,
        classical_extractor=classical_ext,
        neural_extractor=crf_ext,
    )

    results = [
        rule_ext.extract(sample_text, source_record_id="rec-1", document_id="doc-1", language="en"),
        classical_ext.extract(sample_text, source_record_id="rec-1", document_id="doc-1", language="en"),
        bilstm_ext.extract(sample_text, source_record_id="rec-1", document_id="doc-1", language="en"),
        crf_ext.extract(sample_text, source_record_id="rec-1", document_id="doc-1", language="en"),
        tr_ext.extract(sample_text, source_record_id="rec-1", document_id="doc-1", language="en"),
        hybrid_ext.extract(text=sample_text, source_record_id="rec-1", document_id="doc-1", language="en"),
    ]

    assert len(results[0].records) >= 3  # Rule-based must catch chest pain, fever, metformin, etc.
    for res in results:
        payload = res.to_dict()
        assert payload["source_record_id"] == "rec-1"
        assert payload["document_id"] == "doc-1"
        for rec in payload["records"]:
            assert REQUIRED_CONTRACT_KEYS.issubset(rec.keys())
            assert rec["provenance"]["pretrained_weights_used"] is False


def test_hybrid_multimodal_rejects_unverified_image_pair() -> None:
    """Ensure Approach 6 works with partial modalities and never creates false image-text pairs."""
    hybrid = HybridMultimodalExtractor(rule_extractor=RuleBasedMedicalExtractor())
    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)

    unverified = hybrid.extract(
        text="Patient has fever.",
        image_rgb=dummy_img,
        image_record_id="rec-other",
        explicit_image_pair_verified=False,
        source_record_id="rec-1",
    )
    assert "unverified_image_pair_not_linked" in unverified.warnings
    assert unverified.modality_status["image"] == "unavailable"
    assert len(unverified.records) >= 1

    ocr_only = hybrid.extract(
        text=None,
        ocr_text="Patient takes metformin 500 mg orally.",
        source_record_id="rec-2",
    )
    assert "using_ocr_fallback_without_native_text" in ocr_only.warnings
    assert ocr_only.modality_status["text"] == "absent"
    assert ocr_only.modality_status["ocr"] == "available"
    assert len(ocr_only.records) >= 1


def test_six_approach_benchmark_runs_cleanly() -> None:
    """Run leak-free comparative benchmark across all 6 approaches."""
    report = run_six_approach_benchmark(epochs=10, seed=42)
    assert report["leakage_verified"] is True
    assert report["pretrained_weights_used"] is False
    assert len(report["approaches"]) == 6
    ids = [row["approach_id"] for row in report["approaches"]]
    assert ids == [
        "approach_1_rule_based",
        "approach_2_classical_ml",
        "approach_3_bilstm",
        "approach_4_bilstm_crf",
        "approach_5_transformer",
        "approach_6_hybrid_multimodal",
    ]
