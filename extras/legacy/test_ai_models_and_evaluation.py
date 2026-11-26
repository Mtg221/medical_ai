"""Unit and scientific validation tests for custom random-initialized OCR, NLP, Vision, Multimodal, and Federated models."""

from __future__ import annotations

from medresearch_platform.ai.evaluation import (
    compute_calibration_and_selective_risk,
    compute_span_f1_metrics,
    simulate_federated_aggregation_round,
)
from medresearch_platform.ai.nlp_model import CustomMedicalNLPNet, _build_nlp_weights
from medresearch_platform.ai.ocr_model import (
    CustomOCRConvNet,
    _build_ocr_weights,
    compute_cer_and_wer,
    render_scanned_document_image,
)
from medresearch_platform.ai.random_init import verify_random_init_provenance
from medresearch_platform.ai.terminology import BilingualTerminologyRegistry
from medresearch_platform.documents.synthetic_generator import generate_nlp_training_samples
from medresearch_platform.workflows.review import compute_inter_annotator_agreement


def test_random_init_provenance_verification() -> None:
    ocr = CustomOCRConvNet(seed=20261112)
    nlp = CustomMedicalNLPNet(seed=20261112)
    assert ocr.init_manifest.pretrained_weights_used is False
    assert nlp.init_manifest.pretrained_weights_used is False
    assert verify_random_init_provenance(ocr.init_manifest, _build_ocr_weights) is True
    assert verify_random_init_provenance(nlp.init_manifest, _build_nlp_weights) is True


def test_custom_ocr_convnet_training_and_cer_wer() -> None:
    ocr = CustomOCRConvNet(seed=20261112)
    stats = ocr.train_on_synthetic_glyphs(epochs=140, learning_rate=0.012)
    assert stats["final_loss"] < stats["initial_loss"]
    assert stats["train_accuracy"] >= 0.98

    lines = ["HBA1C 7.4 %", "AMOXICILLIN 1000 MG"]
    img, meta = render_scanned_document_image(lines, noise_std=0.0, seed=42)
    blocks = ocr.recognize_image_blocks(img, meta)
    preds = [b.recognized_text for b in blocks]
    metrics = compute_cer_and_wer(preds, lines)
    assert metrics["cer"] == 0.0
    assert metrics["wer"] == 0.0
    assert all(0.0 <= b.aleatoric_uncertainty <= 1.0 for b in blocks)


def test_custom_medical_nlp_critical_clinical_distinctions_bilingual() -> None:
    nlp = CustomMedicalNLPNet(seed=20261112)
    nlp.train_on_labeled_spans(generate_nlp_training_samples(), epochs=80, learning_rate=0.12)

    clinical_note = (
        "History of essential hypertension. "
        "Family history of atrial fibrillation. "
        "Pulmonary embolism ruled out. "
        "Suspected bacterial pneumonia. "
        "Return if fever develops. "
        "Patient takes metformin 500 mg twice daily orally. "
        "Amoxicillin discontinued. "
        "Sans douleur thoracique. Suspicion de pneumopathie infectieuse."
    )
    entities, relations = nlp.extract_from_text(clinical_note)
    by_text = {(e.source_text.lower(), e.entity_type): e for e in entities}

    # 1. Historical affirmed diagnosis
    htn = by_text[("essential hypertension", "DIAGNOSIS")]
    assert htn.assertion_status == "AFFIRMED"
    assert htn.temporal_status == "HISTORICAL"

    # 2. Family history vs patient history
    afib = by_text[("atrial fibrillation", "DIAGNOSIS")]
    assert afib.assertion_status == "FAMILY_HISTORY"

    # 3. Ruled-out is NEGATED, not confirmed
    pe = by_text[("pulmonary embolism", "DIAGNOSIS")]
    assert pe.assertion_status == "NEGATED"

    # 4. Suspected is UNCERTAIN
    pna = by_text[("bacterial pneumonia", "DIAGNOSIS")]
    assert pna.assertion_status == "UNCERTAIN"

    # 5. Hypothetical statement ("Return if fever develops") is HYPOTHETICAL
    fev = by_text[("fever", "SYMPTOM")]
    assert fev.assertion_status == "HYPOTHETICAL"

    # 6. Medication taken vs discontinued
    met = by_text[("metformin", "MEDICATION")]
    assert met.medication_status == "TAKEN"
    amox = by_text[("amoxicillin", "MEDICATION")]
    assert amox.medication_status == "DISCONTINUED"

    # 7. French negation ("Sans douleur thoracique") & uncertainty ("Suspicion de pneumopathie infectieuse")
    fr_pain = by_text[("douleur thoracique", "SYMPTOM")]
    assert fr_pain.assertion_status == "NEGATED"
    fr_pna = by_text[("pneumopathie infectieuse", "DIAGNOSIS")]
    assert fr_pna.assertion_status == "UNCERTAIN"

    # 8. Relations extracted (metformin -> 500 mg, twice daily, orally)
    rel_types = {r.relation_type for r in relations}
    assert "HAS_DOSAGE" in rel_types
    assert "HAS_FREQUENCY" in rel_types
    assert "HAS_ROUTE" in rel_types


def test_bilingual_terminology_mapping_preservation_of_unmapped() -> None:
    reg = BilingualTerminologyRegistry()
    exact_en = reg.map_mention("essential hypertension", category_hint="DIAGNOSIS")
    assert exact_en.match_type == "EXACT"
    assert exact_en.code == "I10"

    syn_fr = reg.map_mention("hta", category_hint="DIAGNOSIS")
    assert syn_fr.match_type == "SYNONYM"
    assert syn_fr.code == "I10"

    unmapped = reg.map_mention("rare unindexed syndrome xyz", category_hint="DIAGNOSIS")
    assert unmapped.match_type == "UNMAPPED"
    assert unmapped.concept_id is None


def test_inter_annotator_agreement_and_federated_quarantine() -> None:
    iaa = compute_inter_annotator_agreement(
        ["AFFIRMED", "NEGATED", "UNCERTAIN", "AFFIRMED", "HYPOTHETICAL"],
        ["AFFIRMED", "NEGATED", "AFFIRMED", "AFFIRMED", "HYPOTHETICAL"],
    )
    assert iaa["observed_agreement"] == 0.8
    assert iaa["cohens_kappa"] > 0.7
    assert iaa["unresolved_disagreement_count"] == 1

    nlp_a = CustomMedicalNLPNet(seed=1)
    nlp_b = CustomMedicalNLPNet(seed=2)
    malicious = {k: v * 50.0 for k, v in nlp_a.params.items()}
    _, fed_info = simulate_federated_aggregation_round(
        local_site_weights={"site_a": nlp_a.params, "site_b": nlp_b.params, "bad_site": malicious},
        site_sample_counts={"site_a": 20, "site_b": 20, "bad_site": 5},
    )
    assert "bad_site" in fed_info["quarantined_sites"]
    assert "site_a" in fed_info["participating_sites"]

    cal = compute_calibration_and_selective_risk([0.95, 0.9, 0.85, 0.4], [1, 1, 1, 0])
    assert cal["brier_score"] < 0.15
    span_m = compute_span_f1_metrics([("d1", 0, 5, "SYMPTOM")], [("d1", 0, 5, "SYMPTOM")])
    assert span_m["exact_f1"] == 1.0
