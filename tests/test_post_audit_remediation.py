"""Post-audit remediation regression suite covering NLP-001, NLP-002, TEMP-001, and SEC-002."""

from __future__ import annotations

from pathlib import Path

import pytest

from medresearch_platform.approaches.context_negex import classify_assertion_attributes
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.extraction.contract import build_extracted_record
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.hybrid.stage4_temporal import normalize_timex_detailed


def test_nlp_002_orthogonal_three_axis_assertions() -> None:
    # "Possible history of asthma." -> certainty=possible AND temporality=historical AND experiencer=patient
    s1 = "Possible history of asthma."
    i1 = s1.index("asthma")
    a1 = classify_assertion_attributes(s1, i1, i1 + len("asthma"), language="en")
    assert a1.certainty == "possible"
    assert a1.temporality == "historical"
    assert a1.experiencer == "patient"

    # "Her mother has diabetes." -> certainty=confirmed, temporality=current, experiencer=family
    s2 = "Her mother has diabetes."
    i2 = s2.index("diabetes")
    a2 = classify_assertion_attributes(s2, i2, i2 + len("diabetes"), language="en")
    assert a2.certainty == "confirmed"
    assert a2.temporality == "current"
    assert a2.experiencer == "family"
    assert a2.legacy_assertion == "family"

    # "No evidence of pneumonia." -> certainty=absent, temporality=current, experiencer=patient
    s3 = "No evidence of pneumonia."
    i3 = s3.index("pneumonia")
    a3 = classify_assertion_attributes(s3, i3, i3 + len("pneumonia"), language="en")
    assert a3.certainty == "absent"
    assert a3.temporality == "current"
    assert a3.experiencer == "patient"

    # Contract record preserves all 3 axes
    rec = build_extracted_record(
        source_record_id="r1",
        document_id="d1",
        original_text=s1,
        extracted_value="asthma",
        entity_type="DIAGNOSIS",
        normalized_concept="J45.9",
        char_start=i1,
        char_end=i1 + len("asthma"),
        language="en",
        negation_status="uncertain",
        assertion_status="uncertain",
        temporal_information="historical",
        confidence_score=0.92,
        model_name="hybrid_v2",
        model_version="2.0.0",
        provenance={},
        certainty="possible",
        temporality="historical",
        experiencer="patient",
    )
    d = rec.to_dict()
    assert d["certainty"] == "possible"
    assert d["temporality"] == "historical"
    assert d["experiencer"] == "patient"


def test_nlp_001_six_clinical_regression_probes() -> None:
    model_dir = Path("models/hybrid_v2_seed1")
    if not (model_dir / "manifest.json").exists():
        pytest.skip("Local models/hybrid_v2_seed1 not present in clean CI checkout")

    pipe = HybridPipeline.load(model_dir)

    def _run(text: str):
        doc = AnnotatedDocument(
            doc_id="reg",
            patient_id="p0",
            language="en",
            doc_type="clinical_note",
            split="test",
            source="regression",
            document_date="2025-03-15",
            text=text,
        )
        return pipe.extract_document(doc)

    # 1. Negated pneumonia
    p1, _ = _run("No evidence of pneumonia.")
    assert any(e.text.lower() == "pneumonia" and e.certainty == "absent" for e in p1.entities)

    # 2. Orthogonal possible + historical asthma
    p2, c2 = _run("Possible history of asthma.")
    asthma_ents = [e for e in p2.entities if e.text.lower() == "asthma"]
    assert asthma_ents and asthma_ents[0].certainty == "possible" and asthma_ents[0].temporality == "historical"
    assert any(e["certainty"] == "possible" and e["temporality"] == "historical" for e in c2["entities"])

    # 3. Family history diabetes
    p3, _ = _run("Her mother has diabetes.")
    assert any(e.text.lower() == "diabetes" and e.experiencer == "family" for e in p3.entities)

    # 4. Medication regimen: no 'took'->MEDICATION, no 'seven days'->ANATOMY, 'for seven days' -> P7D
    p4, _ = _run("The patient took 5 mg twice daily for seven days.")
    assert not any(e.text.lower() == "took" and e.type == "MEDICATION" for e in p4.entities)
    assert not any("days" in e.text.lower() and e.type == "ANATOMY" for e in p4.entities)
    assert any(t.normalized == "P7D" for t in p4.timex)
    assert any(t.normalized == "BID" for t in p4.timex)

    # 5. Scoped negation with 'but'
    p5, _ = _run("The patient denies chest pain but reports shortness of breath.")
    assert any(e.text.lower() == "chest pain" and e.certainty == "absent" for e in p5.entities)
    assert any(e.text.lower() == "shortness of breath" and e.certainty == "confirmed" for e in p5.entities)

    # 6. Contrastive scan: no 'scan does'->ANATOMY
    p6, _ = _run("The previous scan showed a lesion; the current scan does not.")
    assert not any(e.text.lower() == "scan does" and e.type == "ANATOMY" for e in p6.entities)


def test_temp_001_ranges_ocr_repair_and_ambiguity_flagging() -> None:
    # Date ranges (EN + FR)
    r_en = normalize_timex_detailed("from March 1 to March 5, 2025", "DATE", "2025-03-15", language="en")
    assert r_en["normalized"] == "2025-03-01/2025-03-05"
    assert r_en["ambiguous"] is False

    r_fr = normalize_timex_detailed("du 1 au 5 mars 2025", "DATE", "2025-03-15", language="fr")
    assert r_fr["normalized"] == "2025-03-01/2025-03-05"

    # OCR digit repair
    r_ocr = normalize_timex_detailed("202S-03-15", "DATE", "2025-03-15", language="en")
    assert r_ocr["normalized"] == "2025-03-15"
    assert r_ocr["ocr_repaired"] is True
    assert r_ocr["needs_review"] is True

    # Ambiguous numeric date: language-aware AND explicitly flagged for review
    r_amb_en = normalize_timex_detailed("03/04/2025", "DATE", "2025-03-15", language="en")
    assert r_amb_en["normalized"] == "2025-03-04"
    assert r_amb_en["ambiguous"] is True
    assert r_amb_en["needs_review"] is True

    r_amb_fr = normalize_timex_detailed("03/04/2025", "DATE", "2025-03-15", language="fr")
    assert r_amb_fr["normalized"] == "2025-04-03"
    assert r_amb_fr["ambiguous"] is True
    assert r_amb_fr["needs_review"] is True
