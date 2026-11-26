"""Step 1 tests (Phase 2 P10 + Phase 3 Q1-Q7, R2): PHI/name redaction in reports & exports, and A.2 extraction probes."""

from __future__ import annotations

from pathlib import Path

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.reports.generator import render_html_report, render_markdown_report
from medresearch_platform.security.privacy import deidentify_clinical_text


def test_p10_deidentify_names_and_redact_in_reports() -> None:
    text = "Mme D, 67 ans, BPCO. Patient Name: Jean Dupont. IPP: 98765432."
    res = deidentify_clinical_text(text)
    assert "Mme D" not in res.scrubbed_text
    assert "Jean Dupont" not in res.scrubbed_text
    assert "98765432" not in res.scrubbed_text
    assert "[PATIENT_NAME_REDACTED]" in res.scrubbed_text

    doc_payload = {
        "document_id": "doc_p10_1",
        "original_filename": "Mme_D_consult.txt",
        "detected_language": "fr",
        "extraction_mode": "hybrid_v2",
    }
    entities = [
        {
            "entity_type": "DIAGNOSIS",
            "source_text": "BPCO (Patient: Jean Dupont)",
            "normalized_text": "BPCO",
            "assertion_status": "AFFIRMED",
            "temporal_status": "CURRENT",
            "confidence": 0.92,
            "char_start": 0,
            "char_end": 4,
        }
    ]
    md = render_markdown_report(doc_payload, entities, [], lang="fr")
    html_out = render_html_report(doc_payload, entities, [], lang="fr")
    assert "Jean Dupont" not in md
    assert "Jean Dupont" not in html_out


def test_q1_to_q7_and_r2_a2_probes_on_hybrid_v2(isolated_test_models: tuple[Path, Path]) -> None:
    hybrid_dir, _ = isolated_test_models
    pipe = HybridPipeline.load(hybrid_dir)

    # Probe 0: FR note with honorific, age, diagnosis, medication + µg dosage
    doc0 = AnnotatedDocument(
        doc_id="probe_0",
        patient_id="p_probe",
        language="fr",
        doc_type="CLINICAL_NOTE",
        document_date="2026-11-15",
        source="probe",
        text="Mme D, 67 ans, BPCO sous tiotropium 18 µg.",
    )
    pred0, _ = pipe.extract_document(doc0)

    # Mme / Mme D never DIAGNOSIS; 67 ans never DATE
    for e in pred0.entities:
        assert not (e.type == "DIAGNOSIS" and "mme" in e.text.lower()), f"Unexpected DIAGNOSIS on honorific: {e}"
        assert not (e.type == "DATE" and "67 ans" in e.text.lower()), f"Unexpected DATE on age: {e}"
        assert not (e.type == "DATE" and "18 µg" in e.text.lower()), f"Unexpected DATE on dosage: {e}"

    meds0 = [e for e in pred0.entities if e.type == "MEDICATION" and "tiotropium" in e.text.lower()]
    doses0 = [e for e in pred0.entities if e.type == "DOSAGE" and "18" in e.text]
    assert len(meds0) == 1, f"Expected MEDICATION tiotropium, got {pred0.entities}"
    assert len(doses0) == 1, f"Expected DOSAGE 18 µg, got {pred0.entities}"
    assert any(r.type == "HAS_DOSAGE" for r in pred0.relations), f"Expected HAS_DOSAGE relation, got {pred0.relations}"

    # Low-confidence entities (e.g. BPCO < 0.65) must be flagged needs_review=True
    for e in pred0.entities:
        if e.confidence < 0.65:
            assert e.needs_review is True

    # Probe 1: EN negated medication
    doc1 = AnnotatedDocument(
        doc_id="probe_1",
        patient_id="p_probe",
        language="en",
        doc_type="CLINICAL_NOTE",
        document_date="2026-11-15",
        source="probe",
        text="Patient is not taking aspirin.",
    )
    pred1, _ = pipe.extract_document(doc1)
    meds1 = [e for e in pred1.entities if e.type == "MEDICATION" and "aspirin" in e.text.lower()]
    assert len(meds1) == 1, f"Expected MEDICATION aspirin, got {pred1.entities}"
    assert meds1[0].assertion == "absent", f"Expected absent assertion for negated aspirin, got {meds1[0]}"
