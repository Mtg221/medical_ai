"""Cross-document regression benchmark for NOISE-0038, patient ID isolation, and EN/FR structured vitals/labs."""

from __future__ import annotations

from pathlib import Path

import pytest

from medresearch_platform.config import PlatformConfig
from medresearch_platform.serving.extractor import (
    ExtractionService,
    _extract_structured_vitals_and_labs,
)
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService

NOISE_0038_PATH = Path("tests/fixtures/noise_0038.txt")


def test_patient_id_isolation_when_subject_id_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_models: tuple[Path, Path],
) -> None:
    hybrid_dir, crf_dir = isolated_test_models
    monkeypatch.setenv("MEDRESEARCH_DB_PATH", str(tmp_path / "iso.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_SQLITE_PATH", str(tmp_path / "iso.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MEDRESEARCH_MODEL_DIR", str(hybrid_dir))
    monkeypatch.setenv("MEDRESEARCH_CLASSICAL_CRF_DIR", str(crf_dir))

    svc = MedicalResearchPlatformService(PlatformConfig.from_env(workspace_dir=tmp_path))
    svc.initialize_platform()
    principal = svc.get_default_admin_principal()

    r1 = svc.ingest_document(
        principal=principal,
        raw_bytes=b"Document A: Patient has fever and cough.",
        filename="doc_a.txt",
        dataset_id="ds_iso",
        subject_raw_id="",
        family_id="fam_a",
    )
    r2 = svc.ingest_document(
        principal=principal,
        raw_bytes=b"Document B: Patient has headache and hypertension.",
        filename="doc_b.txt",
        dataset_id="ds_iso",
        subject_raw_id="",
        family_id="fam_b",
    )
    with svc.db.connect() as conn:
        s1 = conn.execute("SELECT subject_id FROM documents WHERE document_id = ?", (r1["document_id"],)).fetchone()[0]
        s2 = conn.execute("SELECT subject_id FROM documents WHERE document_id = ?", (r2["document_id"],)).fetchone()[0]
    assert s1 != s2, f"Unrelated documents collided on the same subject_id: {s1}"
    assert s1 != "subj_synth_001" and s2 != "subj_synth_001"


def test_bilingual_fr_structured_vitals_and_labs_extraction() -> None:
    fr_text = "Tension artérielle de 140/90 mmHg.\nPoids de 82 kg.\nCréatinine: 92 µmol/L."
    ents, rels = _extract_structured_vitals_and_labs(fr_text, [])
    tests = {e.text.lower() for e in ents if e.type == "LAB_TEST"}
    vals = {e.text for e in ents if e.type == "LAB_VALUE"}
    assert "tension artérielle" in tests
    assert "poids" in tests
    assert "créatinine" in tests
    assert {"140/90", "82", "92"}.issubset(vals)
    assert sum(1 for r in rels if r.type == "HAS_VALUE") == 3
    assert sum(1 for r in rels if r.type == "HAS_UNIT") == 3


def test_noise_0038_full_pipeline_regression_benchmark() -> None:
    pytest.importorskip("torch")
    if not Path("models/hospital_char_cnn_bilstm_crf_seed1/manifest.json").exists():
        pytest.skip("Hospital 18-model checkpoints not present in this environment")

    text = NOISE_0038_PATH.read_text(encoding="utf-8")
    svc = ExtractionService()
    res = svc.extract(
        text,
        language="en",
        document_date="2026-11-26",
        document_id="NOISE-0038",
        approach="hospital_ensemble",
    )

    # 1. Zero invalid entities from the baseline audit
    forbidden = {
        ("MEDICATION", "says"),
        ("DIAGNOSIS", "as"),
        ("SYMPTOM", "taking"),
        ("SYMPTOM", "dry"),
        ("SYMPTOM", "t catch breath"),
        ("SYMPTOM", "diabetes"),
        ("FREQUENCY", "child"),
        ("ROUTE", "prn"),
        ("ROUTE", "od"),
        ("PROCEDURE", "hba1c"),
        ("PROCEDURE", "spo2"),
        ("DATE", "7/10"),
    }
    found = {(r.entity_type, r.original_text.strip().lower()) for r in res.records}
    assert forbidden.isdisjoint(found), f"Regressed NOISE-0038 artifacts: {forbidden & found}"

    # 2. PRN & OD properly typed as FREQUENCY
    assert ("FREQUENCY", "prn") in found
    assert ("FREQUENCY", "od") in found

    # 3. Family history attribution
    afib = [r for r in res.records if r.original_text.lower() == "atrial fibrillation"]
    assert afib and afib[0].experiencer == "family"
    diab = [r for r in res.records if r.original_text.lower() == "diabetes"]
    assert any(d.entity_type == "DIAGNOSIS" and d.experiencer == "family" for d in diab)
    assert any(d.entity_type == "DIAGNOSIS" and d.certainty == "absent" and d.experiencer == "patient" for d in diab)

    # 4. Structured vital & lab extraction + linking (baseline had only 1 value linked!)
    linked_labs = {
        r.original_text.lower()
        for r in res.records
        if r.entity_type == "LAB_TEST"
        and any(rel.get("type") == "HAS_VALUE" for rel in r.provenance.get("relations", []))
    }
    assert len(linked_labs) >= 14, f"Expected >= 14 distinct linked labs/vitals, got {len(linked_labs)}: {sorted(linked_labs)}"
    for expected_lab in ("bp", "hr", "spo2", "hb", "wbc", "na", "k", "creatinine", "crp", "hs-tni"):
        assert expected_lab in linked_labs, f"Missing linked vital/lab: {expected_lab}"
