"""Prompt 18 — Step 6: X8 free-text probe regression suite (tracked as xfail before data improvements)."""

from __future__ import annotations

from pathlib import Path

import pytest

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.hybrid.pipeline import HybridPipeline


def _run_doc(pipe: HybridPipeline, doc_id: str, lang: str, text: str):
    doc = AnnotatedDocument(
        doc_id=doc_id,
        patient_id="p_x8",
        document_date="2026-11-18",
        source="x8_probe",
        language=lang,
        doc_type="progress_note",
        text=text,
        entities=[],
        relations=[],
    )
    pred, _ = pipe.extract_document(doc)
    return pred.entities


@pytest.mark.xfail(reason="X8: 'céphalée' mislabelled as ANATOMY instead of SYMPTOM without data coverage")
def test_x8_probe1_cephalee_is_symptom(isolated_test_models: tuple[Path, Path]) -> None:
    hybrid_dir, _ = isolated_test_models
    pipe = HybridPipeline.load(hybrid_dir)
    ents = _run_doc(pipe, "x8_1", "fr", "Pr Ndiaye examine Mlle Camara, âgée de 34 ans, pour une céphalée.")
    hits = [e for e in ents if "céphalée" in e.text.lower() or "cephalee" in e.text.lower()]
    assert hits and hits[0].type == "SYMPTOM", f"Expected SYMPTOM for céphalée, got {ents}"


@pytest.mark.xfail(reason="X8: pronoun 'her' mislabelled as DIAGNOSIS and 'tid' mislabelled as ROUTE")
def test_x8_probe2_pronoun_relative_and_tid(isolated_test_models: tuple[Path, Path]) -> None:
    hybrid_dir, _ = isolated_test_models
    pipe = HybridPipeline.load(hybrid_dir)
    ents = _run_doc(pipe, "x8_2", "en", "Seen with her son Ibrahima Fall; amoxicillin 1 g tid for 7 days.")
    assert not any(e.text.lower() == "her" for e in ents), f"Pronoun 'her' should not be an entity: {ents}"
    tid_hits = [e for e in ents if e.text.lower() == "tid"]
    assert tid_hits and tid_hits[0].type == "FREQUENCY", f"Expected FREQUENCY for 'tid', got {ents}"


@pytest.mark.xfail(reason="X8: excluding test ('angioscanner' / 'chest X-ray') wrongly marked absent instead of present")
def test_x8_probe3_excluding_test_stays_present(isolated_test_models: tuple[Path, Path]) -> None:
    hybrid_dir, _ = isolated_test_models
    pipe = HybridPipeline.load(hybrid_dir)
    ents_fr = _run_doc(pipe, "x8_3fr", "fr", "Embolie pulmonaire écartée par angioscanner.")
    pe_fr = [e for e in ents_fr if "embolie" in e.text.lower()]
    angio = [e for e in ents_fr if "angioscanner" in e.text.lower()]
    assert pe_fr and pe_fr[0].assertion == "absent", f"Expected absent PE, got {ents_fr}"
    assert angio and angio[0].assertion == "present", f"Expected present angioscanner, got {ents_fr}"

    ents_en = _run_doc(pipe, "x8_3en", "en", "Pneumonia was excluded on chest X-ray.")
    pneu = [e for e in ents_en if "pneumonia" in e.text.lower()]
    cxr = [e for e in ents_en if "x-ray" in e.text.lower()]
    assert pneu and pneu[0].assertion == "absent", f"Expected absent pneumonia, got {ents_en}"
    assert cxr and cxr[0].assertion == "present", f"Expected present chest X-ray, got {ents_en}"


@pytest.mark.xfail(reason="X8: 'methimazole' mislabelled as SYMPTOM instead of MEDICATION")
def test_x8_probe4_methimazole_is_medication(isolated_test_models: tuple[Path, Path]) -> None:
    hybrid_dir, _ = isolated_test_models
    pipe = HybridPipeline.load(hybrid_dir)
    ents = _run_doc(pipe, "x8_4", "en", "Graves disease treated with methimazole.")
    meth = [e for e in ents if "methimazole" in e.text.lower()]
    assert meth and meth[0].type == "MEDICATION", f"Expected MEDICATION for methimazole, got {ents}"


@pytest.mark.xfail(reason="X8: '7-year-old' mislabelled as DATE instead of non-clinical age attribute")
def test_x8_probe5_hyphenated_age_not_date(isolated_test_models: tuple[Path, Path]) -> None:
    hybrid_dir, _ = isolated_test_models
    pipe = HybridPipeline.load(hybrid_dir)
    ents = _run_doc(pipe, "x8_5", "en", "A 7-year-old boy with fever.")
    assert not any(e.type == "DATE" and "7" in e.text for e in ents), f"'7-year-old' should not be DATE: {ents}"


@pytest.mark.xfail(reason="X8: 'hospitalisée' mislabelled as DIAGNOSIS instead of non-entity / encounter")
def test_x8_probe6_hospitalisee_not_diagnosis(isolated_test_models: tuple[Path, Path]) -> None:
    hybrid_dir, _ = isolated_test_models
    pipe = HybridPipeline.load(hybrid_dir)
    ents = _run_doc(pipe, "x8_6", "fr", "Patiente de 80 ans, hospitalisée depuis 2 jours.")
    assert not any("hospitalis" in e.text.lower() and e.type == "DIAGNOSIS" for e in ents), (
        f"'hospitalisée' should not be DIAGNOSIS: {ents}"
    )
