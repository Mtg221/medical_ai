"""Synthetic Bilingual (EN/FR) Clinical Corpus Generator with Leak-Free Subject/Family Splits."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import docx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas

from extras.legacy.ocr_model import render_scanned_document_image


@dataclass(frozen=True)
class SyntheticDocumentSpec:
    doc_key: str
    subject_key: str
    family_key: str
    split_assignment: str
    language: str
    specialty: str
    document_type: str
    encounter_date_iso: str
    file_format: str
    filename: str
    raw_text: str
    ocr_lines: list[str]
    gold_entities: list[dict[str, Any]]


SYNTHETIC_CLINICAL_CORPUS: tuple[SyntheticDocumentSpec, ...] = (
    # Subject 1 (Train Split - English - Longitudinal Encounter 1 & 2)
    SyntheticDocumentSpec(
        doc_key="doc_en_subj1_enc1",
        subject_key="subj_synth_001",
        family_key="fam_synth_001_a",
        split_assignment="train",
        language="en",
        specialty="CARDIOLOGY",
        document_type="ADMISSION_NOTE",
        encounter_date_iso="2026-03-10",
        file_format="TXT",
        filename="subj1_admission_2026_03.txt",
        raw_text=(
            "Patient Name: Jane Doe | MRN: SYNTH-1001 | Date: 2026-03-10\n"
            "CHIEF COMPLAINT:\n"
            "Patient presents with shortness of breath and chest pain for 3 days.\n"
            "PAST MEDICAL HISTORY:\n"
            "History of essential hypertension and type 2 diabetes mellitus. "
            "Family history of atrial fibrillation.\n"
            "MEDICATIONS:\n"
            "Patient takes metformin 500 mg twice daily orally and amlodipine 5 mg once daily.\n"
            "LABORATORY RESULTS:\n"
            "HbA1c is 7.4 % and CRP is 18.2 mg/L.\n"
            "IMAGING:\n"
            "Chest CT performed; pulmonary embolism ruled out.\n"
            "ASSESSMENT AND PLAN:\n"
            "Suspected bacterial pneumonia in lungs. Prescribed amoxicillin 1000 mg three times daily. "
            "Return if fever develops."
        ),
        ocr_lines=["HBA1C 7.4 %", "CRP 18.2 MG/L", "AMOXICILLIN 1000 MG"],
        gold_entities=[
            {"source_text": "shortness of breath", "entity_type": "SYMPTOM", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "chest pain", "entity_type": "SYMPTOM", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "essential hypertension", "entity_type": "DIAGNOSIS", "assertion_status": "AFFIRMED", "temporal_status": "HISTORICAL"},
            {"source_text": "type 2 diabetes mellitus", "entity_type": "DIAGNOSIS", "assertion_status": "AFFIRMED", "temporal_status": "HISTORICAL"},
            {"source_text": "atrial fibrillation", "entity_type": "DIAGNOSIS", "assertion_status": "FAMILY_HISTORY", "temporal_status": "HISTORICAL"},
            {"source_text": "metformin", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "amlodipine", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "pulmonary embolism", "entity_type": "DIAGNOSIS", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
            {"source_text": "bacterial pneumonia", "entity_type": "DIAGNOSIS", "assertion_status": "UNCERTAIN", "temporal_status": "CURRENT"},
            {"source_text": "amoxicillin", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "fever", "entity_type": "SYMPTOM", "assertion_status": "HYPOTHETICAL", "temporal_status": "CURRENT"},
        ],
    ),
    SyntheticDocumentSpec(
        doc_key="doc_en_subj1_enc2",
        subject_key="subj_synth_001",
        family_key="fam_synth_001_b",
        split_assignment="train",
        language="en",
        specialty="PULMONOLOGY",
        document_type="FOLLOWUP_PDF",
        encounter_date_iso="2026-03-24",
        file_format="PDF",
        filename="subj1_followup_2026_03.pdf",
        raw_text=(
            "FOLLOW-UP CLINICAL SUMMARY - 2026-03-24\n"
            "CHIEF COMPLAINT:\n"
            "Follow-up visit. Patient denies chest pain and without shortness of breath.\n"
            "MEDICATIONS:\n"
            "Amoxicillin 1000 mg discontinued after completion. Continues metformin 500 mg twice daily.\n"
            "LABORATORY RESULTS:\n"
            "CRP is 2.1 mg/L and serum creatinine is 0.9 mg/dL.\n"
            "ASSESSMENT AND PLAN:\n"
            "Bacterial pneumonia ruled out on follow-up; essential hypertension remains controlled."
        ),
        ocr_lines=["CRP 2.1 MG/L", "CREATININE 0.9 MG/DL"],
        gold_entities=[
            {"source_text": "chest pain", "entity_type": "SYMPTOM", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
            {"source_text": "shortness of breath", "entity_type": "SYMPTOM", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
            {"source_text": "Amoxicillin", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "metformin", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "Bacterial pneumonia", "entity_type": "DIAGNOSIS", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
            {"source_text": "essential hypertension", "entity_type": "DIAGNOSIS", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        ],
    ),
    # Subject 2 (Validation Split - French DOCX)
    SyntheticDocumentSpec(
        doc_key="doc_fr_subj2_enc1",
        subject_key="subj_synth_002",
        family_key="fam_synth_002_a",
        split_assignment="val",
        language="fr",
        specialty="MEDECINE_INTERNE",
        document_type="COMPTE_RENDU_HOSPITALISATION",
        encounter_date_iso="2026-04-05",
        file_format="DOCX",
        filename="subj2_compte_rendu_fr.docx",
        raw_text=(
            "Nom du patient: Jean Dupont | IPP: DOS-90210 | Date: 2026-04-05\n"
            "MOTIF D'HOSPITALISATION:\n"
            "Le patient présente une dyspnée et une fièvre depuis 4 jours, sans douleur thoracique.\n"
            "ANTÉCÉDENTS:\n"
            "Antécédent de hypertension artérielle et diabète de type 2. "
            "Antécédent familial de embolie pulmonaire.\n"
            "TRAITEMENT:\n"
            "Prend metformine 1000 mg deux fois par jour par voie orale. "
            "Instauration de amoxicilline 1000 mg trois fois par jour.\n"
            "BIOLOGIE:\n"
            "Protéine C-réactive est à 64.5 mg/L et créatininémie est à 88 µmol/L.\n"
            "EXAMENS COMPLÉMENTAIRES:\n"
            "Angioscanner thoracique réalisé : embolie pulmonaire exclue.\n"
            "CONCLUSION:\n"
            "Suspicion de pneumopathie infectieuse des poumons. Consulter en cas de douleur thoracique."
        ),
        ocr_lines=["CRP 64.5 MG/L", "METFORMINE 1000 MG"],
        gold_entities=[
            {"source_text": "dyspnée", "entity_type": "SYMPTOM", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "fièvre", "entity_type": "SYMPTOM", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "douleur thoracique", "entity_type": "SYMPTOM", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
            {"source_text": "hypertension artérielle", "entity_type": "DIAGNOSIS", "assertion_status": "AFFIRMED", "temporal_status": "HISTORICAL"},
            {"source_text": "diabète de type 2", "entity_type": "DIAGNOSIS", "assertion_status": "AFFIRMED", "temporal_status": "HISTORICAL"},
            {"source_text": "embolie pulmonaire", "entity_type": "DIAGNOSIS", "assertion_status": "FAMILY_HISTORY", "temporal_status": "HISTORICAL"},
            {"source_text": "metformine", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "amoxicilline", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "embolie pulmonaire", "entity_type": "DIAGNOSIS", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
            {"source_text": "pneumopathie infectieuse", "entity_type": "DIAGNOSIS", "assertion_status": "UNCERTAIN", "temporal_status": "CURRENT"},
        ],
    ),
    # Subject 3 (Locked Test Split - Mixed EN/FR TXT + Scanned Image)
    SyntheticDocumentSpec(
        doc_key="doc_bilingual_subj3_enc1",
        subject_key="subj_synth_003",
        family_key="fam_synth_003_a",
        split_assignment="locked_test",
        language="mixed_en_fr",
        specialty="CARDIOLOGY",
        document_type="BILINGUAL_CONSULT_NOTE",
        encounter_date_iso="2026-05-12",
        file_format="TXT",
        filename="subj3_bilingual_consult.txt",
        raw_text=(
            "BILINGUAL CLINICAL CONSULTATION - 2026-05-12\n"
            "CHIEF COMPLAINT:\n"
            "Patient presents with dyspnée and cough since 5 days; pas de douleur thoracique.\n"
            "PAST MEDICAL HISTORY:\n"
            "History of atrial fibrillation and hypertension artérielle.\n"
            "MEDICATIONS:\n"
            "Patient takes apixaban 5 mg twice daily and amlodipine 10 mg once daily.\n"
            "LABORATORY RESULTS:\n"
            "HbA1c is 5.8 % and CRP is 12.0 mg/L.\n"
            "ASSESSMENT AND PLAN:\n"
            "Pulmonary embolism ruled out. Possible pneumonia; scheduled chest CT for follow-up."
        ),
        ocr_lines=["APIXABAN 5 MG BID", "AMLODIPINE 10 MG QD", "HBA1C 5.8 %"],
        gold_entities=[
            {"source_text": "dyspnée", "entity_type": "SYMPTOM", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "cough", "entity_type": "SYMPTOM", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "douleur thoracique", "entity_type": "SYMPTOM", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
            {"source_text": "atrial fibrillation", "entity_type": "DIAGNOSIS", "assertion_status": "AFFIRMED", "temporal_status": "HISTORICAL"},
            {"source_text": "hypertension artérielle", "entity_type": "DIAGNOSIS", "assertion_status": "AFFIRMED", "temporal_status": "HISTORICAL"},
            {"source_text": "apixaban", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "amlodipine", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
            {"source_text": "Pulmonary embolism", "entity_type": "DIAGNOSIS", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
            {"source_text": "pneumonia", "entity_type": "DIAGNOSIS", "assertion_status": "UNCERTAIN", "temporal_status": "CURRENT"},
            {"source_text": "chest CT", "entity_type": "PROCEDURE", "assertion_status": "AFFIRMED", "temporal_status": "PLANNED"},
        ],
    ),
)


def build_synthetic_file_bytes(spec: SyntheticDocumentSpec) -> tuple[bytes, list[dict[str, Any]]]:
    """Generate actual binary payload (TXT, PDF, DOCX, or PNG) for a SyntheticDocumentSpec."""
    fmt = spec.file_format.upper()
    if fmt == "TXT":
        return spec.raw_text.encode("utf-8"), []

    if fmt == "PDF":
        buf = io.BytesIO()
        c = pdf_canvas.Canvas(buf, pagesize=letter)
        y = 740
        for line in spec.raw_text.splitlines():
            c.drawString(48, y, line)
            y -= 18
        c.showPage()
        c.save()
        return buf.getvalue(), []

    if fmt == "DOCX":
        doc = docx.Document()
        for line in spec.raw_text.splitlines():
            doc.add_paragraph(line)
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue(), []

    if fmt == "PNG":
        img, line_meta = render_scanned_document_image(spec.ocr_lines, seed=42)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), line_meta

    raise ValueError(f"Unsupported synthetic format: {fmt}")


def generate_nlp_training_samples() -> list[dict[str, Any]]:
    """Produce labeled span + context training examples for training CustomMedicalNLPNet from random init."""
    return [
        # Symptoms
        {"span_text": "shortness of breath", "left_context": "Patient presents with ", "right_context": " and chest pain", "entity_type": "SYMPTOM", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "chest pain", "left_context": "Patient denies ", "right_context": " at rest", "entity_type": "SYMPTOM", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
        {"span_text": "dyspnée", "left_context": "Le patient présente une ", "right_context": " aiguë", "entity_type": "SYMPTOM", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "douleur thoracique", "left_context": "sans ", "right_context": " associée", "entity_type": "SYMPTOM", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
        {"span_text": "fever", "left_context": "Return if ", "right_context": " develops", "entity_type": "SYMPTOM", "assertion_status": "HYPOTHETICAL", "temporal_status": "CURRENT"},
        {"span_text": "fièvre", "left_context": "en cas de ", "right_context": " persistante", "entity_type": "SYMPTOM", "assertion_status": "HYPOTHETICAL", "temporal_status": "CURRENT"},
        # Diagnoses
        {"span_text": "essential hypertension", "left_context": "History of ", "right_context": " since 2019", "entity_type": "DIAGNOSIS", "assertion_status": "AFFIRMED", "temporal_status": "HISTORICAL"},
        {"span_text": "hypertension artérielle", "left_context": "Antécédent de ", "right_context": " traitée", "entity_type": "DIAGNOSIS", "assertion_status": "AFFIRMED", "temporal_status": "HISTORICAL"},
        {"span_text": "type 2 diabetes mellitus", "left_context": "History of ", "right_context": " controlled", "entity_type": "DIAGNOSIS", "assertion_status": "AFFIRMED", "temporal_status": "HISTORICAL"},
        {"span_text": "diabète de type 2", "left_context": "Antécédent de ", "right_context": " équilibré", "entity_type": "DIAGNOSIS", "assertion_status": "AFFIRMED", "temporal_status": "HISTORICAL"},
        {"span_text": "pulmonary embolism", "left_context": "Chest CT performed; ", "right_context": " ruled out.", "entity_type": "DIAGNOSIS", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
        {"span_text": "embolie pulmonaire", "left_context": "Scanner thoracique : ", "right_context": " exclue.", "entity_type": "DIAGNOSIS", "assertion_status": "NEGATED", "temporal_status": "CURRENT"},
        {"span_text": "bacterial pneumonia", "left_context": "Suspected ", "right_context": " in lungs", "entity_type": "DIAGNOSIS", "assertion_status": "UNCERTAIN", "temporal_status": "CURRENT"},
        {"span_text": "pneumopathie infectieuse", "left_context": "Suspicion de ", "right_context": " bilatérale", "entity_type": "DIAGNOSIS", "assertion_status": "UNCERTAIN", "temporal_status": "CURRENT"},
        {"span_text": "atrial fibrillation", "left_context": "Family history of ", "right_context": " in mother", "entity_type": "DIAGNOSIS", "assertion_status": "FAMILY_HISTORY", "temporal_status": "HISTORICAL"},
        {"span_text": "embolie pulmonaire", "left_context": "Antécédent familial de ", "right_context": " chez le père", "entity_type": "DIAGNOSIS", "assertion_status": "FAMILY_HISTORY", "temporal_status": "HISTORICAL"},
        # Medications & Posology
        {"span_text": "metformin", "left_context": "Patient takes ", "right_context": " 500 mg twice daily", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "metformine", "left_context": "Prend ", "right_context": " 1000 mg deux fois par jour", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "amoxicillin", "left_context": "Prescribed ", "right_context": " 1000 mg", "entity_type": "MEDICATION", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "500 mg", "left_context": "metformin ", "right_context": " twice daily", "entity_type": "DOSAGE", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "twice daily", "left_context": "500 mg ", "right_context": " orally", "entity_type": "FREQUENCY", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "orally", "left_context": "twice daily ", "right_context": " with meals", "entity_type": "ROUTE", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        # Labs, Procedures, Anatomy, Temporal
        {"span_text": "HbA1c", "left_context": "Labs: ", "right_context": " is 7.4 %", "entity_type": "LAB_TEST", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "CRP", "left_context": "and ", "right_context": " is 18.2 mg/L", "entity_type": "LAB_TEST", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "7.4", "left_context": "HbA1c is ", "right_context": " %", "entity_type": "LAB_VALUE", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "mg/L", "left_context": "18.2 ", "right_context": ".", "entity_type": "LAB_UNIT", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "chest CT", "left_context": "scheduled ", "right_context": " for follow-up", "entity_type": "PROCEDURE", "assertion_status": "AFFIRMED", "temporal_status": "PLANNED"},
        {"span_text": "lungs", "left_context": "pneumonia in ", "right_context": ".", "entity_type": "ANATOMY", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
        {"span_text": "for 3 days", "left_context": "chest pain ", "right_context": ".", "entity_type": "TEMPORAL_EXPR", "assertion_status": "AFFIRMED", "temporal_status": "CURRENT"},
    ]


def verify_leak_free_splits(specs: tuple[SyntheticDocumentSpec, ...] = SYNTHETIC_CLINICAL_CORPUS) -> dict[str, Any]:
    """Verify zero subject_key or family_key overlap across train, val, and locked_test splits (Section 17)."""
    subjects_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "locked_test": set()}
    families_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "locked_test": set()}

    for spec in specs:
        subjects_by_split.setdefault(spec.split_assignment, set()).add(spec.subject_key)
        families_by_split.setdefault(spec.split_assignment, set()).add(spec.family_key)

    subj_train_val = subjects_by_split["train"] & subjects_by_split["val"]
    subj_train_test = subjects_by_split["train"] & subjects_by_split["locked_test"]
    subj_val_test = subjects_by_split["val"] & subjects_by_split["locked_test"]

    fam_train_val = families_by_split["train"] & families_by_split["val"]
    fam_train_test = families_by_split["train"] & families_by_split["locked_test"]
    fam_val_test = families_by_split["val"] & families_by_split["locked_test"]

    leak_detected = bool(
        subj_train_val or subj_train_test or subj_val_test or fam_train_val or fam_train_test or fam_val_test
    )
    return {
        "leak_detected": leak_detected,
        "subject_counts_by_split": {k: len(v) for k, v in subjects_by_split.items()},
        "family_counts_by_split": {k: len(v) for k, v in families_by_split.items()},
        "subject_overlaps": {
            "train_val": sorted(subj_train_val),
            "train_locked_test": sorted(subj_train_test),
            "val_locked_test": sorted(subj_val_test),
        },
    }
