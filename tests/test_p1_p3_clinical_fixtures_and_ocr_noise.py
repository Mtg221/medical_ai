"""Comprehensive P1-P3 clinical fixture table (18 cases), 8 OCR noise variants, edge cases & provisional NOISE-0027/0038 checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medresearch_platform.approaches.context_negex import (
    classify_assertion_attributes,
    detect_section_at_offset,
)
from medresearch_platform.serving.extractor import (
    ExtractionService,
    _normalize_ocr_clinical_token,
    detect_clinical_contradictions,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def extractor() -> ExtractionService:
    return ExtractionService()


# --- 18 Clinical Fixture Table Cases (P1 Context, P2 Meds/Labs/Vitals, P3 Contradictions) ---


def test_p1_18_clinical_fixtures_context_and_structure(extractor: ExtractionService) -> None:
    # 1. Simple negation
    t1 = "Patient denies fever and chills."
    a1 = classify_assertion_attributes(t1, t1.index("fever"), t1.index("fever") + 5, "en")
    assert a1.certainty == "absent" and a1.assertion_detail == "absent"

    # 2. Pseudo-negation / unresolved ("PE not excluded.")
    t2 = "PE not excluded."
    a2 = classify_assertion_attributes(t2, 0, 2, "en")
    assert a2.certainty == "possible" and a2.assertion_detail == "unresolved"

    # 3. Positive followed by comma + negative ("positive for blurred vision, negative for dysuria")
    t3 = "Review of systems positive for blurred vision, negative for dysuria."
    a3_pos = classify_assertion_attributes(t3, t3.index("blurred vision"), t3.index("blurred vision") + 14, "en")
    a3_neg = classify_assertion_attributes(t3, t3.index("dysuria"), t3.index("dysuria") + 7, "en")
    assert a3_pos.certainty == "confirmed"
    assert a3_neg.certainty == "absent"

    # 4. Family history: mother
    t4 = "Family history: mother has migraine."
    a4 = classify_assertion_attributes(t4, t4.index("migraine"), t4.index("migraine") + 8, "en")
    assert a4.experiencer == "family" and a4.experiencer_detail == "mother"

    # 5. Family history: father
    t5 = "Father died after heart attack at age 58."
    a5 = classify_assertion_attributes(t5, t5.index("heart attack"), t5.index("heart attack") + 12, "en")
    assert a5.experiencer == "family" and a5.experiencer_detail == "father"

    # 6. Family history: brother
    t6 = "Brother has atrial fibrillation."
    a6 = classify_assertion_attributes(t6, t6.index("atrial fibrillation"), t6.index("atrial fibrillation") + 19, "en")
    assert a6.experiencer == "family" and a6.experiencer_detail == "brother"

    # 7. Return precautions / conditional
    t7 = "Return precautions:\nWorsening headache or weakness — seek urgent evaluation."
    a7 = classify_assertion_attributes(t7, t7.index("headache"), t7.index("headache") + 8, "en")
    assert a7.certainty == "possible" and a7.assertion_detail == "conditional"
    assert detect_section_at_offset(t7, t7.index("headache")) == "return_precautions"

    # 8. Unperformed scan safety guard
    t8 = "CTPA report says No evidence of PE. Addendum: study not performed due to elevated creatinine."
    a8 = classify_assertion_attributes(t8, t8.index("PE"), t8.index("PE") + 2, "en")
    assert a8.certainty == "possible" and a8.assertion_detail == "unresolved"

    # 9. Discontinued medication temporality
    t9 = "Patient stopped atorvastatin 40 mg two months ago due to myalgia."
    a9 = classify_assertion_attributes(t9, t9.index("atorvastatin"), t9.index("atorvastatin") + 12, "en")
    assert a9.temporality == "historical" and a9.assertion_detail == "discontinued"

    # 10. Future planned follow-up
    t10 = "Plan: Cardiology follow-up in 2 weeks for echocardiogram."
    a10 = classify_assertion_attributes(t10, t10.index("echocardiogram"), t10.index("echocardiogram") + 14, "en")
    assert a10.assertion_detail == "planned" and a10.temporality_detail == "future_planned"

    # 11. French negation + French pseudo-negation ("sans rendez-vous")
    t11_neg = "Absence de fièvre et pas de dyspnée."
    a11_neg = classify_assertion_attributes(t11_neg, t11_neg.index("fièvre"), t11_neg.index("fièvre") + 6, "fr")
    assert a11_neg.certainty == "absent"
    t11_pseudo = "Consultation sans rendez-vous pour céphalées."
    a11_pseudo = classify_assertion_attributes(t11_pseudo, t11_pseudo.index("céphalées"), t11_pseudo.index("céphalées") + 9, "fr")
    assert a11_pseudo.certainty == "confirmed"

    # 12. French family history
    t12 = "Antécédents familiaux : mère diabétique, frère avec AVC."
    a12 = classify_assertion_attributes(t12, t12.index("AVC"), t12.index("AVC") + 3, "fr")
    assert a12.experiencer == "family" and a12.experiencer_detail == "brother"

    # 13. P2: Troponin with reference range -> within_range
    r13 = extractor.extract("Troponin I: 18 ng/L (reference <19 ng/L)", language="en", document_id="f13", approach="hybrid_v2")
    trop = [r for r in r13.records if r.entity_type == "LAB_TEST" and "troponin" in r.original_text.lower()]
    assert trop and trop[0].provenance.get("reference_range") == "<19 ng/L"
    assert trop[0].provenance.get("interpretation") == "within_range"

    # 14. P2: SpO2 on room air -> oxygen_condition == "room_air" & no bogus MEDICATION "air"
    r14 = extractor.extract("SpO2 94% on room air", language="en", document_id="f14", approach="hybrid_v2")
    spo2 = [r for r in r14.records if r.entity_type == "LAB_TEST" and r.original_text.lower() == "spo2"]
    assert spo2 and spo2[0].provenance.get("oxygen_condition") == "room_air"
    assert not any(r.entity_type == "MEDICATION" and r.original_text.lower() == "air" for r in r14.records)

    # 15. P2: Aspirin ordered but not administered -> NOT confirmed present administration
    r15 = extractor.extract("Aspirin 300 mg ordered but not administered.", language="en", document_id="f15", approach="hybrid_v2")
    asp = [r for r in r15.records if r.entity_type == "MEDICATION" and "aspirin" in r.original_text.lower()]
    assert asp and asp[0].provenance.get("medication_status") == "ordered_not_administered"
    assert asp[0].assertion_status == "uncertain"

    # 16. P2: Ibuprofen PRN last taken yesterday -> prn_active + historical last admin
    r16 = extractor.extract("Ibuprofen 400 mg PRN — last taken yesterday.", language="en", document_id="f16", approach="hybrid_v2")
    ibu = [r for r in r16.records if r.entity_type == "MEDICATION" and "ibuprofen" in r.original_text.lower()]
    assert ibu and ibu[0].provenance.get("medication_status") == "prn_active"
    assert ibu[0].provenance.get("last_administered_temporality") == "historical"

    # 17. P2: Omeprazole 20 mg OD -> OD is FREQUENCY ("once daily"), NOT ROUTE
    r17 = extractor.extract("Omeprazole 20 mg OD", language="en", document_id="f17", approach="hybrid_v2")
    assert not any(r.entity_type == "ROUTE" and r.original_text.upper() == "OD" for r in r17.records)

    # 18. P3: Contradictions preserved (NKDA vs latex + Amlodipine 10 mg vs 5 mg)
    t18 = "Allergies: NKDA. Separate page: latex - itching.\n- Amlodipine 10 mg daily; patient says 5 mg."
    r18 = extractor.extract(t18, language="en", document_id="f18", approach="hybrid_v2")
    contras = detect_clinical_contradictions(r18.records, t18)
    topics = {c["topic"] for c in contras}
    assert "allergy_status" in topics
    assert "medication_dose:amlodipine" in topics


# --- 8 OCR Noise Variants ---


@pytest.mark.parametrize(
    ("raw", "etype", "expect_corrupted", "expect_cand"),
    [
        ("TnI2O", "LAB_VALUE", True, "TnI20"),
        ("CARDI0LOGY", "DIAGNOSIS", True, "CARDIOLOGY"),
        ("chest p@in", "SYMPTOM", True, "chest pain"),
        ("15B/9?", "LAB_VALUE", True, None),
        ("9A", "LAB_VALUE", True, "94"),
        ("M3tformin", "MEDICATION", True, "Metformin"),
        ("elevat1on", "DIAGNOSIS", True, "elevation"),
    ],
)
def test_p3_ocr_noise_variants_normalization(
    raw: str,
    etype: str,
    expect_corrupted: bool,
    expect_cand: str | None,
) -> None:
    cand, corrupted = _normalize_ocr_clinical_token(raw, etype)
    assert corrupted is expect_corrupted
    assert cand == expect_cand


def test_p3_ocr_noise_8th_variant_pe_not_excludcd() -> None:
    txt = "CTPA limited: PE not excludcd."
    attrs = classify_assertion_attributes(txt, txt.index("PE"), txt.index("PE") + 2, "en")
    assert attrs.certainty == "possible"
    assert attrs.assertion_detail == "unresolved"


# --- Edge Cases & Determinism ---


def test_edge_cases_empty_whitespace_and_deterministic_cpu(extractor: ExtractionService) -> None:
    empty_res = extractor.extract("", language="en", document_id="empty_doc", approach="hybrid_v2")
    assert len(empty_res.records) == 0

    ws_res = extractor.extract("   \n\t  ", language="en", document_id="ws_doc", approach="hybrid_v2")
    assert len(ws_res.records) == 0

    sample = "BP 142/86 mmHg. Troponin I: 18 ng/L (reference <19 ng/L). Denies fever."
    run1 = extractor.extract(sample, language="en", document_id="det_1", approach="hybrid_v2")
    run2 = extractor.extract(sample, language="en", document_id="det_2", approach="hybrid_v2")
    sig1 = [(r.entity_type, r.char_start, r.char_end, r.assertion_status, r.confidence_score) for r in run1.records]
    sig2 = [(r.entity_type, r.char_start, r.char_end, r.assertion_status, r.confidence_score) for r in run2.records]
    assert sig1 == sig2


# --- Provisional Reference Checks on NOISE-0027 and NOISE-0038 ---


def test_provisional_draft_references_exist_and_marked_unreviewed() -> None:
    for fname in ("noise_0027_ref_draft.json", "noise_0038_ref_draft.json"):
        payload = json.loads((FIXTURES_DIR / fname).read_text(encoding="utf-8"))
        assert payload["status"] == "draft_unreviewed"
        assert payload["reviewer_signoff"] is None
        assert len(payload["expected_clinical_checks"]) >= 15
