"""Phase 3 bilingual (EN & FR) NLP generalization and counterexample regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from medresearch_platform.approaches.context_negex import classify_assertion_attributes
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.hybrid.stage4_temporal import normalize_timex


def test_phase3_bilingual_three_axis_context_cues() -> None:
    # EN: simultaneous absent + historical + family
    s_en = "Mother denies any history of cancer."
    i_en = s_en.index("cancer")
    a_en = classify_assertion_attributes(s_en, i_en, i_en + len("cancer"), language="en")
    assert a_en.certainty == "absent"
    assert a_en.temporality == "historical"
    assert a_en.experiencer == "family"

    # FR: simultaneous possible + historical
    s_fr = "Antécédent probable d'asthme."
    i_fr = s_fr.index("asthme")
    a_fr = classify_assertion_attributes(s_fr, i_fr, i_fr + len("asthme"), language="fr")
    assert a_fr.certainty == "possible"
    assert a_fr.temporality == "historical"
    assert a_fr.experiencer == "patient"

    # FR: family history with possessive pronoun
    s_fr_fam = "Sa mère est diabétique."
    i_fam = s_fr_fam.index("diabétique")
    a_fam = classify_assertion_attributes(s_fr_fam, i_fam, i_fam + len("diabétique"), language="fr")
    assert a_fam.certainty == "confirmed"
    assert a_fam.experiencer == "family"


def test_phase3_french_word_frequency_normalization() -> None:
    assert normalize_timex("une fois par jour", "FREQUENCY", "2025-03-15") == "QD"
    assert normalize_timex("deux fois par jour", "FREQUENCY", "2025-03-15") == "BID"
    assert normalize_timex("trois fois par jour", "FREQUENCY", "2025-03-15") == "TID"
    assert normalize_timex("quatre fois par jour", "FREQUENCY", "2025-03-15") == "QID"


def test_phase3_closed_class_non_entity_and_fr_timex_with_hybrid_v2() -> None:
    model_dir = Path("models/hybrid_v2_seed1")
    if not (model_dir / "manifest.json").exists():
        pytest.skip("Local models/hybrid_v2_seed1 not present in clean CI checkout")

    pipe = HybridPipeline.load(model_dir)

    def _run(lang: str, text: str):
        doc = AnnotatedDocument(
            doc_id="p3",
            patient_id="p0",
            language=lang,
            doc_type="clinical_note",
            split="test",
            source="p3_test",
            document_date="2025-03-15",
            text=text,
        )
        return pipe.extract_document(doc)[0]

    # EN: No pronoun 'His' as DIAGNOSIS
    p_en1 = _run("en", "His father died of myocardial infarction.")
    assert not any(e.text.lower() == "his" for e in p_en1.entities)

    # EN: No 'normal' as DIAGNOSIS
    p_en2 = _run("en", "Prior MRI showed a mass; repeat MRI is normal.")
    assert not any(e.text.lower() == "normal" and e.type == "DIAGNOSIS" for e in p_en2.entities)

    # FR: Detect & normalize both 'deux fois par jour' (BID) and 'pendant dix jours' (P10D)
    p_fr = _run("fr", "Le patient prend 500 mg deux fois par jour pendant dix jours.")
    norms = {t.normalized for t in p_fr.timex}
    assert "BID" in norms
    assert "P10D" in norms


def test_phase3_hospital_pytorch_does_not_tag_french_pronoun_sa_as_date() -> None:
    pytest.importorskip("torch")
    model_dir = Path("models/hospital_char_cnn_bilstm_crf_seed1")
    if not (model_dir / "manifest.json").exists():
        pytest.skip("Local hospital_char_cnn_bilstm_crf_seed1 not present in clean CI checkout")

    from medresearch_platform.approaches.torch_ner import PyTorchCharCNNBiLSTMCRF

    m = PyTorchCharCNNBiLSTMCRF.load(model_dir, device="cpu")
    ents = m.predict_entities("Sa mère est diabétique.")
    assert not any(e.text.lower() == "sa" and e.type == "DATE" for e in ents)
    assert any(e.text.lower() == "diabétique" and e.type == "DIAGNOSIS" for e in ents)
