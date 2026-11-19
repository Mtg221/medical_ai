"""Prompt 19 — Step 4: General EN/FR text rules & Y6 probes (eponyms, relatives, ages, durations, cues)."""

from __future__ import annotations

from pathlib import Path

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.security.person_age_detector import detect_age_spans, detect_person_spans
from medresearch_platform.security.privacy import deidentify_clinical_text


def _predict_text(pipe: HybridPipeline, text: str, lang: str) -> list[tuple[str, str, str]]:
    doc = AnnotatedDocument(
        doc_id=f"p19_y6_{lang}",
        patient_id="PAT-TEST",
        language=lang,
        doc_type="consultation",
        document_date="2026-11-18",
        source="p19_y6_probe",
        text=text,
        entities=[],
        relations=[],
        timex=[],
        split="probe",
    )
    pred, _ = pipe.extract_document(doc)
    return [(e.type, e.text.lower(), e.assertion) for e in pred.entities]


def test_p19_step4_eponyms_protected_and_relatives_and_ages_detected() -> None:
    # Eponymous diseases must NOT be redacted or treated as PERSON
    for sent in (
        "Patient has Down syndrome and Graves disease.",
        "Suivi pour maladie de Parkinson et maladie de Crohn.",
        "Hashimoto thyroiditis and Alzheimer disease noted.",
        "Syndrome de Guillain-Barré et maladie de Horton.",
    ):
        assert detect_person_spans(sent) == [], f"False person on eponym: {sent}"
        res = deidentify_clinical_text(sent)
        assert "[PATIENT_NAME]" not in res.scrubbed_text
        assert "[CLINICIAN_NAME]" not in res.scrubbed_text

    # Relative + name must be detected as a single PERSON span
    for sent, expected_sub in (
        ("Accompanied by his wife Claire Martin today.", "his wife Claire Martin"),
        ("Seen with her son Ibrahima Fall.", "her son Ibrahima Fall"),
        ("Accompagnée par sa fille Awa Diop.", "sa fille Awa Diop"),
        ("Présent avec son épouse Marie-Claire Moreau.", "son épouse Marie-Claire Moreau"),
    ):
        spans = detect_person_spans(sent)
        assert any(expected_sub in sp.text for sp in spans), f"Missing relative span in {sent}: {spans}"

    # N-day/week/month/year-old and âgé(e) de N jours/semaines/mois/ans -> AGE spans
    for sent in (
        "5-day-old infant with fever.",
        "6-week-old boy with cough.",
        "14-month-old girl with rash.",
        "Nourrisson âgé de 5 jours avec fièvre.",
        "Enfant âgée de 3 semaines sans toux.",
    ):
        assert len(detect_age_spans(sent)) >= 1, f"Missing age span in {sent}"


def test_p19_step4_20_en_and_20_fr_y6_generalization_sentences(isolated_test_models: tuple[Path, Path]) -> None:
    pipe = HybridPipeline.load(isolated_test_models[0])

    en_sentences = [
        "5-day-old infant with fever for 3 weeks.",
        "6-week-old boy seen with his wife Claire Martin; pneumonia was excluded.",
        "14-month-old girl with cough for 2 days.",
        "A 9-year-old child with asthma; bronchitis was ruled out.",
        "Toddler with dyspnea since 4 days; Down syndrome noted.",
        "Newborn with jaundice for 5 days.",
        "Teenager with headache for 2 weeks; migraine excluded.",
        "Patient accompanied by her husband David Brooks for 3 months.",
        "His daughter Sarah Patel reported chest pain for 6 hours.",
        "Graves disease treated with methimazole for 4 weeks.",
        "Crohn disease flare for 10 days; appendicitis was excluded.",
        "Parkinson disease stable for 2 years.",
        "Hashimoto thyroiditis diagnosed; hypothyroidism ruled out.",
        "Alzheimer disease followed since 3 years.",
        "Boy with fever and cough for 5 days.",
        "Girl with abdominal pain for 3 weeks; appendicitis excluded.",
        "Neonate with tachypnea for 12 hours.",
        "Her son Moussa Ndiaye accompanied the patient for 2 days.",
        "His mother Awa Diop reports fever since 3 days.",
        "Addison disease suspected; Cushing syndrome ruled out.",
    ]
    fr_sentences = [
        "Nourrisson âgé de 5 jours présentant une fièvre depuis 3 semaines.",
        "Enfant âgée de 4 ans vue avec sa fille Awa Diop ; pneumonie écartée.",
        "Garçon de 8 ans avec toux depuis 2 jours.",
        "Fille de 6 ans avec asthme ; bronchite éliminée.",
        "Bébé âgé de 3 semaines avec dyspnée pendant 4 jours.",
        "Adolescent suivi depuis 2 mois pour céphalée ; méningite écartée.",
        "Patient accompagné par son épouse Claire Martin depuis 3 semaines.",
        "Son fils Ibrahima Fall signale une douleur thoracique depuis 6 heures.",
        "Maladie de Parkinson stable depuis 2 ans.",
        "Maladie de Crohn en poussée depuis 10 jours ; appendicite exclue.",
        "Maladie de Basedow traitée depuis 4 semaines.",
        "Syndrome de Down suivi depuis 5 ans.",
        "Nourrisson avec fièvre depuis 2 jours.",
        "Enfant avec douleur abdominale pendant 3 jours ; péritonite éliminée.",
        "Adolescente avec fatigue depuis 3 semaines.",
        "Sa mère Fatou Ndiaye décrit une toux depuis 5 jours.",
        "Son père Jean-Luc Martin signale une fièvre depuis 24 heures.",
        "Thyroïdite de Hashimoto confirmée ; hypothyroïdie écartée.",
        "Maladie d'Alzheimer suivie depuis 4 ans.",
        "Enfant âgé de 14 mois présentant une toux depuis 1 semaine.",
    ]

    age_nouns = {"boy", "girl", "infant", "toddler", "child", "teenager", "newborn", "neonate", "nourrisson", "enfant", "garçon", "fille", "bébé", "adolescent", "adolescente"}
    time_words = {"semaine", "semaines", "week", "weeks", "jour", "jours", "day", "days", "mois", "month", "months", "an", "ans", "year", "years", "heure", "heures", "hour", "hours"}
    cue_words = {"ruled", "excluded", "écartée", "écarté", "éliminée", "éliminé", "exclu", "exclue"}

    for lang, sents in (("en", en_sentences), ("fr", fr_sentences)):
        assert len(sents) == 20
        for sent in sents:
            ents = _predict_text(pipe, sent, lang)
            for etype, etext, _ in ents:
                assert etext not in age_nouns, f"Age noun tagged as {etype} in: {sent}"
                assert not (etype == "MEDICATION" and etext in time_words), f"Time word tagged as MEDICATION in: {sent}"
                assert etext not in cue_words, f"Assertion cue tagged as {etype} in: {sent}"
            if "for " in sent or "depuis " in sent or "pendant " in sent or "since " in sent:
                assert any(etype in {"DURATION", "DATE"} for etype, _, _ in ents), f"Missing DURATION/DATE in: {sent} -> {ents}"
