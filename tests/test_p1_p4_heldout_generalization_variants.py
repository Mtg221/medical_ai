"""Held-out synthetic generalization tests (C5 anti-overfitting: >=5 new variants per open item) & Section 6 mandatory cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from medresearch_platform.approaches.context_negex import classify_assertion_attributes
from medresearch_platform.hybrid.stage4_temporal import normalize_timex
from medresearch_platform.serving.extractor import ExtractionService, detect_clinical_contradictions


@pytest.fixture(scope="module")
def extractor() -> ExtractionService:
    if not Path("models/hybrid_v2_seed1/manifest.json").exists():
        pytest.skip("Saved hybrid_v2_seed1 checkpoint not present in fresh-clone environment")
    return ExtractionService()


# --- 5 Held-out Variants for n27_07 category (distant return-precaution urgent action >45 chars, garbled/missing header) ---


@pytest.mark.parametrize(
    ("text", "target", "lang"),
    [
        ("Discharge note: severe abdominal pain, persistent vomiting, high fever or confusion — return immediately to the ED.", "severe abdominal pain", "en"),
        ("Note [OCR cut]: chest tightness, jaw discomfort, diaphoresis or near-syncope -> go to the ER right away.", "chest tightness", "en"),
        ("Instructions: sudden unilateral weakness, facial droop, slurred speech or visual field loss -- call 911.", "sudden unilateral weakness", "en"),
        ("Suivi: aggravation de la dyspnée, douleur thoracique, malaise ou fièvre élevée — consulter en urgence.", "douleur thoracique", "fr"),
        ("Fragment: worsening leg swelling, calf redness, acute shortness of breath or hemoptysis - seek urgent care.", "worsening leg swelling", "en"),
    ],
)
def test_heldout_n27_07_distant_return_precautions(text: str, target: str, lang: str) -> None:
    idx = text.index(target)
    attrs = classify_assertion_attributes(text, idx, idx + len(target), lang)
    assert attrs.certainty == "possible"
    assert attrs.assertion_detail == "conditional"


# --- 5 Held-out Variants for n27_08 category (BP / vital ratios split by CRF tokenizer) ---


@pytest.mark.parametrize(
    ("text", "expected_val"),
    [
        ("Vitals:\nBP 118/74 mmHg", "118/74"),
        ("Constantes:\nTension artérielle: 135/85 mmHg", "135/85"),
        ("| Parameter | Value |\n| BP | 176/104 |", "176/104"),
        ("Triage vitals:\nBP: 92/58 mmHg", "92/58"),
        ("Follow-up check:\nBP repeat: 128/82 mmHg", "128/82"),
    ],
)
def test_heldout_n27_08_ratio_preservation(extractor: ExtractionService, text: str, expected_val: str) -> None:
    res = extractor.extract(text, language="en", document_id="ho_n27_08", approach="hybrid_v2")
    bp_recs = [r for r in res.records if r.entity_type == "LAB_TEST" and any(k in r.original_text.lower() for k in ("bp", "tension"))]
    assert bp_recs, f"No BP LAB_TEST found in {text!r}"
    assert any(
        rel.relation_type == "HAS_VALUE" and rel.target_value == expected_val
        for r in bp_recs
        for rel in r.relationships
    ), f"Expected unified ratio {expected_val!r} in {[r.relationships for r in bp_recs]}"


# --- 5 Held-out Variants for n38_13 category (OCR-corrupted ratios & values MUST stay uncertain/possible, NEVER confirmed) ---


@pytest.mark.parametrize(
    ("text", "expected_raw"),
    [
        ("Vitals:\nBP 16B/9? mmHg", "16B/9?"),
        ("Triage:\nBP 12?/7B mmHg", "12?/7B"),
        ("Scan:\nBP: 1O4/6? mmHg", "1O4/6?"),
        ("| BP repeat | 18?/10A | OCR |", "18?/10A"),
        ("Vitals:\nSpO2 9A% RA", "9A"),
    ],
)
def test_heldout_n38_13_ocr_corrupted_values_never_confirmed(
    extractor: ExtractionService, text: str, expected_raw: str
) -> None:
    res = extractor.extract(text, language="en", document_id="ho_n38_13", approach="hybrid_v2")
    vals = [r for r in res.records if r.entity_type == "LAB_VALUE" and r.original_text == expected_raw]
    assert vals, f"Expected unified OCR LAB_VALUE {expected_raw!r} in {[(r.entity_type, r.original_text) for r in res.records]}"
    assert vals[0].certainty == "possible"
    assert vals[0].provenance.get("ocr_corrupted") is True
    assert vals[0].review_status == "needs_review"


# --- 5 Held-out Variants for n38_15 category (pain/score ratios MUST NEVER be tagged as DATE) ---


@pytest.mark.parametrize(
    ("text", "score"),
    [
        ("Patient reports pain currently 6/10 in right knee.", "6/10"),
        ("Triage note: CP rated 8/10 radiating to jaw.", "8/10"),
        ("Examen: douleur évaluée à 4/10 à la mobilisation.", "4/10"),
        ("After analgesia, pain improved to 2/10.", "2/10"),
        ("Score EVA 5/10 au repos.", "5/10"),
    ],
)
def test_heldout_n38_15_pain_scores_never_date(extractor: ExtractionService, text: str, score: str) -> None:
    res = extractor.extract(text, language="en", document_id="ho_n38_15", approach="hybrid_v2")
    assert not any(r.entity_type == "DATE" and r.original_text == score for r in res.records)


# --- Section 5 P2 & Section 6 Mandatory Clinical Cases ---


def test_mandatory_four_distinct_pneumonia_interpretations() -> None:
    a_no = classify_assertion_attributes("No evidence of pneumonia.", 15, 24, "en")
    a_cannot = classify_assertion_attributes("Cannot exclude pneumonia.", 15, 24, "en")
    a_hist = classify_assertion_attributes("History of pneumonia.", 11, 20, "en")
    a_ro = classify_assertion_attributes("Rule out pneumonia.", 9, 18, "en")

    sigs = {
        (a_no.certainty, a_no.assertion_detail, a_no.temporality),
        (a_cannot.certainty, a_cannot.assertion_detail, a_cannot.temporality),
        (a_hist.certainty, a_hist.assertion_detail, a_hist.temporality),
        (a_ro.certainty, a_ro.assertion_detail, a_ro.temporality),
    }
    assert len(sigs) == 4, f"Expected 4 distinct interpretations, got {sigs}"
    assert a_no.certainty == "absent"
    assert a_cannot.assertion_detail == "unresolved"
    assert a_hist.temporality == "historical"
    assert a_ro.assertion_detail == "suspected"


def test_mandatory_relative_date_grounding() -> None:
    assert normalize_timex("three days ago", "RELATIVE", "2026-11-26") == "2026-11-23"
    assert normalize_timex("last Friday", "RELATIVE", "2026-11-26") == "2026-11-20"


def test_mandatory_contradictions_and_order_vs_admin(extractor: ExtractionService) -> None:
    # 1. Unperformed CTPA vs negative finding
    t_ctpa = "No pulmonary embolism. CTPA not performed because of elevated creatinine."
    r_ctpa = extractor.extract(t_ctpa, language="en", document_id="m_ctpa", approach="hybrid_v2")
    c_ctpa = detect_clinical_contradictions(r_ctpa.records, t_ctpa)
    assert any(c["topic"] == "imaging_unperformed_vs_negative" for c in c_ctpa)

    # 2. Ordered vs not administered aspirin
    t_asp = "Plan:\n- Aspirin 300 mg loading dose ordered. MAR: not administered."
    r_asp = extractor.extract(t_asp, language="en", document_id="m_asp", approach="hybrid_v2")
    c_asp = detect_clinical_contradictions(r_asp.records, t_asp)
    assert any(c["topic"] == "medication_order_vs_administration:aspirin" for c in c_asp)

    # 3. Hypertension vs No history of hypertension
    t_htn = "Problem list: Hypertension. Discharge summary: No history of hypertension."
    r_htn = extractor.extract(t_htn, language="en", document_id="m_htn", approach="hybrid_v2")
    c_htn = detect_clinical_contradictions(r_htn.records, t_htn)
    assert any(c["topic"] == "polarity:hypertension" for c in c_htn)
