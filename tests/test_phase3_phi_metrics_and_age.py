"""Phase 3 PHI tests: span-level precision/recall/F1, HIPAA >89 age redaction, and over-/under-redaction guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medresearch_platform.security.person_age_detector import detect_age_spans_with_values
from medresearch_platform.security.privacy import deidentify_clinical_text
from scripts.eval_redaction_human import evaluate_redaction_jsonl


def test_age_redaction_hipaa_over_89_and_zero_duration_false_positives() -> None:
    # Ages > 89 MUST be redacted under default HIPAA Safe Harbor policy
    over_89_cases = [
        ("A 92-year-old woman presented with dyspnea.", "92-year-old"),
        ("Patient aged 95 admitted for pneumonia.", "aged 95"),
        ("Patiente âgée de 94 ans hospitalisée pour chute.", "âgée de 94 ans"),
        ("Homme de 91 ans suivi pour insuffisance cardiaque.", "91 ans"),
    ]
    for text, target in over_89_cases:
        res = deidentify_clinical_text(text)
        assert target not in res.scrubbed_text, f"Age > 89 leaked: {res.scrubbed_text!r}"
        assert "[AGE_OVER_89_REDACTED]" in res.scrubbed_text

    # Ages <= 89 preserved by default, redacted when redact_all_ages=True
    under_89_en = "A 64-year-old man with diabetes."
    assert "64-year-old" in deidentify_clinical_text(under_89_en).scrubbed_text
    assert "[AGE_REDACTED]" in deidentify_clinical_text(under_89_en, redact_all_ages=True).scrubbed_text

    under_89_fr = "Patient âgé de 52 ans avec asthme."
    assert "52 ans" in deidentify_clinical_text(under_89_fr).scrubbed_text
    assert "[AGE_REDACTED]" in deidentify_clinical_text(under_89_fr, redact_all_ages=True).scrubbed_text

    # Durations, dates, lab values, dosages, and eponyms MUST NOT trigger age or PHI false positives
    non_phi_clinical = [
        "Took amoxicillin 500 mg twice daily for 14 days.",
        "Traitement par amoxicilline 1 g pendant 7 jours.",
        "Stable hypertension for 5 years; HbA1c 6.4%, BP 120/80 mmHg.",
        "Diabète équilibré depuis 3 ans ; CRP 4 mg/L, créatinine 82 µmol/L.",
        "Follow-up after 2 years and screening every 5 years.",
        "Contrôle après 3 ans et bilan tous les 2 ans.",
        "History of Parkinson disease, Graves disease, and Crohn disease.",
        "Maladie de Crohn et syndrome de Guillain-Barré stables.",
        "CT chest on 2025-03-10 showed no pulmonary embolism.",
    ]
    for text in non_phi_clinical:
        assert detect_age_spans_with_values(text) == [], f"Duration/value misclassified as age in: {text!r}"
        res_default = deidentify_clinical_text(text)
        assert res_default.detected_spans == [], f"Over-redaction in default mode: {text!r} -> {res_default.detected_spans}"
        res_all_ages = deidentify_clinical_text(text, redact_all_ages=True)
        assert res_all_ages.detected_spans == [], f"Over-redaction in redact_all_ages mode: {text!r} -> {res_all_ages.detected_spans}"


def test_eval_redaction_human_computes_true_span_level_precision_and_catches_intra_sentence_fp(tmp_path: Path) -> None:
    # Construct a tiny synthetic test fixture where:
    # - Sentence 1 has 1 gold PHI (DOB) AND an extra unannotated person name -> triggers 1 intra-sentence span FP!
    # - Sentence 2 has 0 gold PHI and 0 redactions -> clean non-PHI sentence.
    # - Sentence 3 (FR) has 2 gold PHI (PERSON + DOB) and 0 FPs.
    rows = [
        {
            "id": "s1",
            "language": "en",
            "text": "Dr. John Smith saw patient born on 1975-06-20.",
            "phi_items": [{"type": "DOB", "text": "1975-06-20"}],
        },
        {
            "id": "s2",
            "language": "en",
            "text": "Metformin 500 mg daily for 14 days; HbA1c 6.8%.",
            "phi_items": [],
        },
        {
            "id": "s3",
            "language": "fr",
            "text": "Mme Claire-Anne Exemple, née le 14 janvier 1982.",
            "phi_items": [
                {"type": "PERSON", "text": "Claire-Anne Exemple"},
                {"type": "DOB", "text": "14 janvier 1982"},
            ],
        },
    ]
    ds = tmp_path / "phi_fixture.jsonl"
    ds.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    rep = evaluate_redaction_jsonl(ds, owner_written=False)
    assert rep["metric_methodology"] == "span_level_exact_and_overlap_separated_from_sentence_fp_rate"
    # 3 gold PHI items all redacted -> tp=3, fn=0, recall=1.0
    assert rep["tp"] == 3
    assert rep["fn"] == 0
    assert rep["recall"] == 1.0
    # Dr. John Smith was redacted but not in gold -> span_fp=1, while fp_on_non_phi_sentences=0!
    assert rep["span_fp"] == 1
    assert rep["fp_on_non_phi_sentences"] == 0
    assert rep["non_phi_sentence_fp_rate"] == 0.0
    # True span precision = 3 / (3 + 1) = 0.75 (NOT 1.0 as the old buggy formula would have reported!)
    assert rep["precision"] == pytest.approx(0.75, abs=1e-4)
    assert rep["f1"] == pytest.approx(2 * 0.75 * 1.0 / 1.75, abs=1e-4)
    assert "en" in rep["by_language"] and "fr" in rep["by_language"]
    assert rep["by_language"]["en"]["precision"] == pytest.approx(0.5, abs=1e-4)
    assert rep["by_language"]["fr"]["precision"] == 1.0
    assert "DOB" in rep["by_type"] and "PERSON" in rep["by_type"]
