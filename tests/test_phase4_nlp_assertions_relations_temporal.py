"""Phase 4 NLP regression tests: 7 EN+FR assertion sentences, multi-entity relation disambiguation, and temporal policies."""

from __future__ import annotations

import pytest

from medresearch_platform.approaches.context_negex import classify_assertion_attributes
from medresearch_platform.approaches.tokenizer import TrainOnlyVocabulary
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import PredictedEntity
from medresearch_platform.hybrid.stage3_context_relations import (
    PiecewiseCNNRelationClassifier,
    _has_intervening_same_type_head,
)
from medresearch_platform.hybrid.stage4_temporal import normalize_timex_detailed


@pytest.mark.parametrize(
    "lang,text,target,exp_certainty,exp_temporality,exp_experiencer",
    [
        ("en", "No fever.", "fever", "absent", "current", "patient"),
        ("fr", "Pas de fièvre.", "fièvre", "absent", "current", "patient"),
        ("en", "Possible pneumonia.", "pneumonia", "possible", "current", "patient"),
        ("fr", "Pneumonie possible.", "Pneumonie", "possible", "current", "patient"),
        ("en", "History of asthma.", "asthma", "confirmed", "historical", "patient"),
        ("fr", "Antécédent d'asthme.", "asthme", "confirmed", "historical", "patient"),
        ("en", "Mother has diabetes.", "diabetes", "confirmed", "current", "family"),
        ("fr", "Sa mère a un diabète.", "diabète", "confirmed", "current", "family"),
        ("en", "Previous scan showed a lesion.", "lesion", "confirmed", "historical", "patient"),
        ("fr", "Le scanner précédent montrait une lésion.", "lésion", "confirmed", "historical", "patient"),
        ("en", "Current scan does not show the lesion.", "lesion", "absent", "current", "patient"),
        ("fr", "Le scanner actuel ne montre pas la lésion.", "lésion", "absent", "current", "patient"),
        ("en", "No evidence of recurrent disease.", "recurrent disease", "absent", "current", "patient"),
        ("fr", "Aucun signe de récidive.", "récidive", "absent", "current", "patient"),
    ],
)
def test_seven_clinical_assertions_en_and_fr(
    lang: str,
    text: str,
    target: str,
    exp_certainty: str,
    exp_temporality: str,
    exp_experiencer: str,
) -> None:
    s = text.index(target)
    e = s + len(target)
    attrs = classify_assertion_attributes(text, s, e, language=lang)
    assert attrs.certainty == exp_certainty
    assert attrs.temporality == exp_temporality
    assert attrs.experiencer == exp_experiencer


def test_intervening_same_type_head_and_sentence_boundary_prevent_cross_linking() -> None:
    text = "Started metformin 500 mg BID, and lisinopril 10 mg QD. Next day: aspirin 81 mg."
    met = PredictedEntity("MEDICATION", text.index("metformin"), text.index("metformin") + 9, "metformin")
    d500 = PredictedEntity("DOSAGE", text.index("500 mg"), text.index("500 mg") + 6, "500 mg")
    lis = PredictedEntity("MEDICATION", text.index("lisinopril"), text.index("lisinopril") + 10, "lisinopril")
    d10 = PredictedEntity("DOSAGE", text.index("10 mg"), text.index("10 mg") + 5, "10 mg")
    d81 = PredictedEntity("DOSAGE", text.index("81 mg"), text.index("81 mg") + 5, "81 mg")
    ents = [met, d500, lis, d10, d81]

    # lisinopril intervenes between metformin and 10 mg -> blocked
    assert _has_intervening_same_type_head(ents, met, d10) is True
    # no medication intervenes between metformin and 500 mg -> allowed
    assert _has_intervening_same_type_head(ents, met, d500) is False
    # no medication intervenes between lisinopril and 10 mg -> allowed
    assert _has_intervening_same_type_head(ents, lis, d10) is False

    # Verify PiecewiseCNNRelationClassifier never emits cross-sentence or cross-medication HAS_DOSAGE
    vocab = TrainOnlyVocabulary(min_freq=1)
    vocab.fit(
        [
            AnnotatedDocument(
                doc_id="d1",
                patient_id="p1",
                language="en",
                doc_type="consultation",
                document_date="2025-03-15",
                source="unit_test",
                text=text,
            )
        ]
    )
    clf = PiecewiseCNNRelationClassifier(len(vocab), seed=1)
    # Bias classifier strongly toward HAS_DOSAGE so only structural guards filter invalid pairs
    clf.params["b_cls"][:] = -5.0
    clf.params["b_cls"][2] = 10.0  # HAS_DOSAGE
    rels = clf.predict_relations(text, ents, vocab)
    pairs = {(text[r.head_start : r.head_end], r.type, text[r.tail_start : r.tail_end]) for r in rels}
    assert ("metformin", "HAS_DOSAGE", "500 mg") in pairs
    assert ("lisinopril", "HAS_DOSAGE", "10 mg") in pairs
    assert ("metformin", "HAS_DOSAGE", "10 mg") not in pairs
    assert ("lisinopril", "HAS_DOSAGE", "500 mg") not in pairs
    assert ("lisinopril", "HAS_DOSAGE", "81 mg") not in pairs


def test_temporal_locale_ambiguity_invalid_calendar_and_ocr_repair_flag() -> None:
    en_amb = normalize_timex_detailed("03/04/2025", "DATE", "2025-03-15", language="en")
    fr_amb = normalize_timex_detailed("03/04/2025", "DATE", "2025-03-15", language="fr")
    assert en_amb["ambiguous"] is True and fr_amb["ambiguous"] is True
    assert en_amb["normalized"] == "2025-03-04"
    assert fr_amb["normalized"] == "2025-04-03"

    invalid = normalize_timex_detailed("2025-02-30", "DATE", "2025-03-15", language="en")
    assert invalid["normalized"] is None

    ocr_rep = normalize_timex_detailed("2O25-03-1O", "DATE", "2025-03-15", language="en")
    assert ocr_rep["normalized"] == "2025-03-10"
    assert ocr_rep["ocr_repaired"] is True
