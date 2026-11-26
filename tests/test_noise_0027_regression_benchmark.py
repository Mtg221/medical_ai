"""Full-pipeline regression benchmark for mandatory reference case NOISE-0027 and unseen noisy EN/FR notes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from medresearch_platform.approaches.context_negex import classify_assertion_attributes
from medresearch_platform.eval_harness.metrics import PredictedEntity
from medresearch_platform.serving.extractor import (
    ExtractionService,
    _filter_non_clinical_artifacts,
)

NOISE_0027_PATH = Path("tests/fixtures/noise_0027.txt")
NOISE_0027_EXPECTED_SHA256 = "0e2739681e773a6eb9fb4eaf1d48d63febc5007a12157bd22fe80477f19f7778"


def test_noise_0027_fixture_preserved_verbatim() -> None:
    assert NOISE_0027_PATH.exists(), "NOISE-0027 fixture missing"
    normalized_bytes = NOISE_0027_PATH.read_bytes().replace(b"\r\n", b"\n")
    actual = hashlib.sha256(normalized_bytes).hexdigest()
    assert actual == NOISE_0027_EXPECTED_SHA256


def test_generalizable_artifact_filter_and_context_cues_on_unseen_noisy_notes() -> None:
    # Unseen noisy EN snippet (not NOISE-0027) testing Room Air, family sister, return precautions, discontinued med, OCR noise
    text = (
        "SpO2 95% RA. Family: sister—epilepsy?? (uncertain). "
        "Lisinopril 10 mg discontinued in 2023. "
        "Return precautions: syncope or chest pain—seek urgent care. "
        "Header: Neuro vs Cardio; Dates 01/02/2025; stamp: ILLEGIBLE; OCR."
    )
    raw_ents = [
        PredictedEntity("DIAGNOSIS", text.index("RA"), text.index("RA") + 2, "RA"),
        PredictedEntity("LAB_TEST", text.index("vs"), text.index("vs") + 2, "vs"),
        PredictedEntity("LAB_TEST", text.index("Dates"), text.index("Dates") + 5, "Dates"),
        PredictedEntity("PROCEDURE", text.index("ILLEGIBLE"), text.index("ILLEGIBLE") + 9, "ILLEGIBLE"),
        PredictedEntity("PROCEDURE", text.index("OCR"), text.index("OCR") + 3, "OCR"),
        PredictedEntity("DOSAGE", text.index("2023"), text.index("2023") + 4, "2023"),
    ]
    assert _filter_non_clinical_artifacts(raw_ents, text) == []

    # Unseen family + uncertain cue
    ep_s = text.index("epilepsy")
    ep_attrs = classify_assertion_attributes(text, ep_s, ep_s + 8, language="en")
    assert ep_attrs.experiencer == "family"
    assert ep_attrs.certainty == "possible"

    # Unseen discontinued medication -> historical
    lis_s = text.index("Lisinopril")
    lis_attrs = classify_assertion_attributes(text, lis_s, lis_s + 10, language="en")
    assert lis_attrs.temporality == "historical"

    # Unseen return precaution -> hypothetical / possible (NOT active confirmed symptom)
    syn_s = text.index("syncope")
    syn_attrs = classify_assertion_attributes(text, syn_s, syn_s + 7, language="en")
    assert syn_attrs.certainty == "possible"


def test_noise_0027_full_pipeline_regression_improvements() -> None:
    pytest.importorskip("torch")
    if not Path("models/hospital_char_cnn_bilstm_crf_seed1/manifest.json").exists():
        pytest.skip("Hospital 18-model checkpoints not present in this environment")

    text = NOISE_0027_PATH.read_text(encoding="utf-8")
    svc = ExtractionService()
    res = svc.extract(
        text,
        language="en",
        document_date="2026-11-26",
        document_id="NOISE-0027",
        approach="hospital_ensemble",
    )

    # 1. Zero non-clinical / OCR / Room-Air false positives that plagued baseline
    forbidden_pairs = {
        ("LAB_TEST", "vs"),
        ("LAB_TEST", "dates"),
        ("PROCEDURE", "ocr"),
        ("PROCEDURE", "illegible"),
        ("PROCEDURE", "reflexes"),
        ("SYMPTOM", "pharmacy record"),
        ("SYMPTOM", "alert"),
        ("SYMPTOM", "oriented"),
        ("SYMPTOM", "oriented x3"),
        ("DIAGNOSIS", "ra"),
        ("DOSAGE", "2021"),
        ("FREQUENCY", "old"),
        ("DATE", "noise-0027"),
    }
    found_pairs = {(r.entity_type, r.original_text.strip().lower()) for r in res.records}
    assert forbidden_pairs.isdisjoint(found_pairs), f"Regressed artifacts: {forbidden_pairs & found_pairs}"

    # 2. Experiencer, temporality, and certainty fixes
    strokes = [r for r in res.records if r.original_text.lower() == "stroke"]
    assert strokes and strokes[0].experiencer == "family" and strokes[0].certainty == "possible"

    asthmas = [r for r in res.records if r.original_text.lower() == "asthma"]
    assert asthmas and asthmas[0].temporality == "historical"

    salb = [r for r in res.records if r.original_text.lower() == "salbutamol"]
    assert salb and salb[0].temporality == "historical"

    weak = [r for r in res.records if r.original_text.lower() == "weakness"]
    assert weak and weak[0].certainty == "possible"

    mris = [r for r in res.records if r.original_text.upper() == "MRI"]
    assert mris and any(m.certainty == "possible" for m in mris)

    # 3. Lab-to-value relation linking restored (baseline had 0 lab values linked)
    linked_labs = [
        r for r in res.records
        if r.entity_type == "LAB_TEST"
        and any(rel.get("type") == "HAS_VALUE" for rel in r.provenance.get("relations", []))
    ]
    assert len(linked_labs) >= 3, f"Expected >= 3 labs linked with HAS_VALUE, got {len(linked_labs)}"

    # 4. Bogus ensemble uncertainty spam reduced substantially from baseline (68 -> < 45)
    needs_review_count = sum(1 for r in res.records if r.review_status == "needs_review")
    assert needs_review_count < 45, f"Too many needs_review flags: {needs_review_count} (baseline was 68)"
