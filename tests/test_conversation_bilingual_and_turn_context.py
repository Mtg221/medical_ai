"""Bilingual EN/FR conversation-turn, code-switch, question-vs-answer & 3-axis taxonomy regression tests."""

from __future__ import annotations

import pytest

from medresearch_platform.approaches.context_negex import classify_assertion_attributes


def test_question_vs_answer_turns_en_and_fr() -> None:
    # Clinician question must NOT be confirmed present
    q_en = "Doctor: Are you experiencing chest pain?"
    a_q_en = classify_assertion_attributes(q_en, q_en.index("chest pain"), q_en.index("chest pain") + 10, "en")
    assert a_q_en.certainty == "possible" and a_q_en.assertion_detail == "hypothetical"

    q_fr = "Médecin : Avez-vous des douleurs thoraciques ?"
    a_q_fr = classify_assertion_attributes(q_fr, q_fr.index("douleurs thoraciques"), q_fr.index("douleurs thoraciques") + 20, "fr")
    assert a_q_fr.certainty == "possible" and a_q_fr.assertion_detail == "hypothetical"

    # Patient answer -> explicit absence
    ans_en = "Patient: No, I do not have chest pain."
    a_ans_en = classify_assertion_attributes(ans_en, ans_en.index("chest pain"), ans_en.index("chest pain") + 10, "en")
    assert a_ans_en.certainty == "absent" and a_ans_en.experiencer == "patient"

    ans_fr = "Patient : Non, je n'ai pas de douleurs thoraciques."
    a_ans_fr = classify_assertion_attributes(ans_fr, ans_fr.index("douleurs thoraciques"), ans_fr.index("douleurs thoraciques") + 20, "fr")
    assert a_ans_fr.certainty == "absent" and a_ans_fr.experiencer == "patient"


def test_family_history_planned_exam_prescribed_not_taken_and_resolved_en_fr() -> None:
    # Family history
    fam_en = "Patient: My father had heart disease."
    a_fam_en = classify_assertion_attributes(fam_en, fam_en.index("heart disease"), fam_en.index("heart disease") + 13, "en")
    assert a_fam_en.experiencer == "family" and a_fam_en.experiencer_detail == "father"

    fam_fr = "Patient : Ma mère a des migraines."
    a_fam_fr = classify_assertion_attributes(fam_fr, fam_fr.index("migraines"), fam_fr.index("migraines") + 9, "fr")
    assert a_fam_fr.experiencer == "family" and a_fam_fr.experiencer_detail == "mother"

    # Planned exam (not completed)
    plan_en = "Doctor: We will order an MRI tomorrow."
    a_plan_en = classify_assertion_attributes(plan_en, plan_en.index("MRI"), plan_en.index("MRI") + 3, "en")
    assert a_plan_en.assertion_detail == "planned" and a_plan_en.temporality_detail == "future_planned"

    plan_fr = "Médecin : On va prescrire une IRM demain."
    a_plan_fr = classify_assertion_attributes(plan_fr, plan_fr.index("IRM"), plan_fr.index("IRM") + 3, "fr")
    assert a_plan_fr.assertion_detail == "planned" and a_plan_fr.temporality_detail == "future_planned"

    # Prescribed but never taken -> must NOT be confirmed administration
    nt_en = "Patient: I was prescribed aspirin, but I never took it."
    a_nt_en = classify_assertion_attributes(nt_en, nt_en.index("aspirin"), nt_en.index("aspirin") + 7, "en")
    assert a_nt_en.certainty == "absent"

    nt_fr = "Patient : On m'a prescrit de l'aspirine mais je ne l'ai jamais prise."
    a_nt_fr = classify_assertion_attributes(nt_fr, nt_fr.index("aspirine"), nt_fr.index("aspirine") + 8, "fr")
    assert a_nt_fr.certainty == "absent"

    # Resolved symptom
    res_en = "Patient: I had a fever yesterday, but it has disappeared."
    a_res_en = classify_assertion_attributes(res_en, res_en.index("fever"), res_en.index("fever") + 5, "en")
    assert a_res_en.temporality == "historical" and a_res_en.assertion_detail == "resolved"

    res_fr = "Patient : J'avais de la fièvre hier mais elle a disparu."
    a_res_fr = classify_assertion_attributes(res_fr, res_fr.index("fièvre"), res_fr.index("fièvre") + 6, "fr")
    assert a_res_fr.temporality == "historical" and a_res_fr.assertion_detail == "resolved"


@pytest.mark.parametrize(
    ("text", "target", "doc_lang", "expect_cert", "expect_exp"),
    [
        ("Médecin : No fever noted today.", "fever", "fr", "absent", "patient"),
        ("Doctor: Pas de dyspnée au repos.", "dyspnée", "en", "absent", "patient"),
        ("Patient : My mother has diabetes, mais moi non.", "diabetes", "mixed", "confirmed", "family"),
        ("Patient: Ma mère a des migraines sévères.", "migraines", "mixed", "confirmed", "family"),
        ("Médecin : Impossible d'exclure une embolie pulmonaire.", "embolie pulmonaire", "mixed", "possible", "patient"),
    ],
)
def test_bilingual_code_switched_context(
    text: str, target: str, doc_lang: str, expect_cert: str, expect_exp: str
) -> None:
    idx = text.index(target)
    attrs = classify_assertion_attributes(text, idx, idx + len(target), language=doc_lang)
    assert attrs.certainty == expect_cert
    assert attrs.experiencer == expect_exp
