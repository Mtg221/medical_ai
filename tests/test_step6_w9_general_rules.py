"""Step 6 (W9) regression tests: general person, age, and rules-out rules across >=20 EN + >=20 FR sentences."""

from __future__ import annotations

from pathlib import Path

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.hybrid.pipeline import HybridPipeline

EN_SENTENCES: list[tuple[str, list[str], bool]] = [
    ("Mr. Carter, 52 years old, takes lisinopril 10 mg for hypertension.", ["Mr. Carter", "52 years old"], False),
    ("Mrs. Patel, aged 71, has asthma without fever.", ["Mrs. Patel", "aged 71"], False),
    ("Ms. Garcia, 34 years, was prescribed amoxicillin 500 mg.", ["Ms. Garcia", "34 years"], False),
    ("Dr. Thompson evaluated Robert Evans for chest pain.", ["Dr. Thompson", "Robert Evans"], False),
    ("Prof. Wilson noted that ultrasound rules out deep vein thrombosis.", ["Prof. Wilson"], True),
    ("MRI ruled out acute stroke in Linda Brooks, 68 years old.", ["Linda Brooks", "68 years old"], True),
    ("Cardiac troponin excluded acute coronary syndrome in Mr. Adams.", ["Mr. Adams"], True),
    ("Daniel Reed, aged 49, takes atorvastatin 20 mg daily.", ["Daniel Reed", "aged 49"], False),
    ("Mrs. Clarke, 61 years old, denies dyspnea.", ["Mrs. Clarke", "61 years old"], False),
    ("Dr. Nguyen prescribed metformin 850 mg to Sarah Miller.", ["Dr. Nguyen", "Sarah Miller"], False),
    ("X-ray rules out pneumonia in Thomas White, 29 years old.", ["Thomas White", "29 years old"], True),
    ("Ms. Foster, aged 55, has diabetes on insulin 12 UI.", ["Ms. Foster", "aged 55"], False),
    ("Kevin Scott, 44 years old, presented with migraine.", ["Kevin Scott", "44 years old"], False),
    ("CT angiogram excluded pulmonary embolism in Mrs. Bennett.", ["Mrs. Bennett"], True),
    ("Dr. Baker ordered ECG for Hannah Cooper, aged 39.", ["Dr. Baker", "Hannah Cooper", "aged 39"], False),
    ("Mr. Hughes, 77 years old, takes furosemide 40 mg.", ["Mr. Hughes", "77 years old"], False),
    ("Echocardiography rules out heart failure in Olivia Ward.", ["Olivia Ward"], True),
    ("Laura Price, 50 years, takes omeprazole 20 mg.", ["Laura Price", "50 years"], False),
    ("Prof. Morgan confirmed biopsy ruled out lung cancer in Mr. Price.", ["Prof. Morgan", "Mr. Price"], True),
    ("Samuel Perry, aged 64, takes losartan 50 mg daily.", ["Samuel Perry", "aged 64"], False),
]

FR_SENTENCES: list[tuple[str, list[str], bool]] = [
    ("M. Bernard, 62 ans, prend du ramipril 5 mg pour hypertension.", ["M. Bernard", "62 ans"], False),
    ("Mme Laurent, âgée de 54 ans, souffre d'asthme sans fièvre.", ["Mme Laurent", "54 ans"], False),
    ("Mlle Girard, 28 ans, reçoit amoxicilline 1 g.", ["Mlle Girard", "28 ans"], False),
    ("Dr Lefebvre a examiné Claire Moreau pour douleur thoracique.", ["Dr Lefebvre", "Claire Moreau"], False),
    ("Pr Chevalier indique que le scanner élimine une embolie pulmonaire.", ["Pr Chevalier"], True),
    ("L'IRM a écarté un AVC chez Paul Mercier, 73 ans.", ["Paul Mercier", "73 ans"], True),
    ("Les D-dimères ont exclu une thrombose veineuse profonde chez Mme Dupuis.", ["Mme Dupuis"], True),
    ("Nadia Ndiaye, âgée de 41 ans, prend metformine 1000 mg.", ["Nadia Ndiaye", "41 ans"], False),
    ("M. Fontaine, 66 ans, pas de signe en faveur d'une pneumonie.", ["M. Fontaine", "66 ans"], True),
    ("Dr Roux a prescrit aspirine 75 mg à Luc Simon.", ["Dr Roux", "Luc Simon"], False),
    ("La radiographie écarte une pneumonie chez Marc Garnier, 35 ans.", ["Marc Garnier", "35 ans"], True),
    ("Mme Chevalier, 59 ans, diabète sous insuline 10 UI.", ["Mme Chevalier", "59 ans"], False),
    ("Patient de 47 ans, Moussa Diop, présente une toux.", ["47 ans", "Moussa Diop"], False),
    ("L'échographie élimine une insuffisance cardiaque chez M. Blanc.", ["M. Blanc"], True),
    ("Dr Fabre a demandé un bilan pour Sophie Perrin, âgée de 32 ans.", ["Dr Fabre", "Sophie Perrin", "32 ans"], False),
    ("M. Gauthier, 79 ans, sous furosémide 40 mg.", ["M. Gauthier", "79 ans"], False),
    ("Le bilan exclut un diabète de type 2 chez Fatou Sarr.", ["Fatou Sarr"], True),
    ("Patiente de 51 ans, Julie Andre, sous atorvastatine 20 mg.", ["51 ans", "Julie Andre"], False),
    ("Pr Colin note qu'une biopsie a éliminé un mélanome chez Mme Leroy.", ["Pr Colin", "Mme Leroy"], True),
    ("Ousmane Ba, âgé de 60 ans, prend losartan 50 mg.", ["Ousmane Ba", "60 ans"], False),
]


def _run_doc(pipe: HybridPipeline, idx: int, lang: str, text: str) -> list:
    doc = AnnotatedDocument(
        doc_id=f"w9_{lang}_{idx}",
        patient_id="p_w9",
        language=lang,
        doc_type="CLINICAL_NOTE",
        document_date="2026-11-18",
        source="w9_regression",
        text=text,
    )
    pred, _ = pipe.extract_document(doc)
    return pred.entities


def test_w9_20_en_and_20_fr_general_sentences(isolated_test_models: tuple[Path, Path]) -> None:
    hybrid_dir, _ = isolated_test_models
    pipe = HybridPipeline.load(hybrid_dir)

    for idx, (text, forbidden_spans, expect_neg) in enumerate(EN_SENTENCES):
        ents = _run_doc(pipe, idx, "en", text)
        for forb in forbidden_spans:
            f_low = forb.lower()
            for e in ents:
                assert f_low not in e.text.lower() and e.text.lower() not in f_low or len(e.text.strip()) <= 1, (
                    f"Forbidden person/age span {forb!r} leaked as {e} in {text!r}"
                )
        if expect_neg:
            dx_ents = [e for e in ents if e.type == "DIAGNOSIS"]
            assert any(e.assertion == "absent" for e in dx_ents), (
                f"Expected absent assertion in {text!r}, got {dx_ents}"
            )

    for idx, (text, forbidden_spans, expect_neg) in enumerate(FR_SENTENCES):
        ents = _run_doc(pipe, idx, "fr", text)
        for forb in forbidden_spans:
            f_low = forb.lower()
            for e in ents:
                assert f_low not in e.text.lower() and e.text.lower() not in f_low or len(e.text.strip()) <= 1, (
                    f"Forbidden person/age span {forb!r} leaked as {e} in {text!r}"
                )
        if expect_neg:
            dx_ents = [e for e in ents if e.type == "DIAGNOSIS"]
            assert any(e.assertion == "absent" for e in dx_ents), (
                f"Expected absent assertion in {text!r}, got {dx_ents}"
            )


def test_w9_required_four_probes(isolated_test_models: tuple[Path, Path]) -> None:
    hybrid_dir, _ = isolated_test_models
    pipe = HybridPipeline.load(hybrid_dir)

    e1 = _run_doc(pipe, 101, "en", "Mr Smith, 45 years old, on metformin 500 mg.")
    assert not any("smith" in e.text.lower() or "45" in e.text.lower() for e in e1)

    e2 = _run_doc(pipe, 102, "fr", "Patient de 58 ans, Dr Martin, ramipril 5 mg/j.")
    assert not any("martin" in e.text.lower() or "58" in e.text.lower() for e in e2)

    e3 = _run_doc(pipe, 103, "en", "Patient aged 63, Mrs. Jones, warfarin 5mg daily.")
    assert not any("jones" in e.text.lower() or "63" in e.text.lower() for e in e3)

    e4 = _run_doc(pipe, 104, "en", "CT rules out pulmonary embolism.")
    pe = [e for e in e4 if "pulmonary embolism" in e.text.lower()]
    assert pe and pe[0].assertion == "absent", f"Expected absent pulmonary embolism, got {e4}"
