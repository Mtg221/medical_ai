"""Generate 100 deterministic bilingual (50 EN / 50 FR) synthetic doctor-patient dialogues and evaluate frozen models.

PROVENANCE NOTICE:
All 100 dialogues are 100% synthetic (owner_written=False, synthetic=True) for conversational domain-shift
stress testing. They are NOT real patient transcripts and must NEVER be reported as human clinical gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medresearch_platform.data.schema import (
    AnnotatedDocument,
    EntitySpan,
    RelationSpan,
)
from medresearch_platform.eval_harness.metrics import Evaluator
from medresearch_platform.serving.extractor import ExtractionService

EN_SYMPTOMS = ["fever", "shortness of breath", "chest pain", "cough", "headache", "nausea", "dizziness", "fatigue"]
FR_SYMPTOMS = ["fièvre", "dyspnée", "douleur thoracique", "toux", "céphalée", "nausées", "vertiges", "fatigue"]

EN_DIAGNOSES = ["type 2 diabetes", "hypertension", "pneumonia", "asthma", "migraine", "heart failure", "atrial fibrillation"]
FR_DIAGNOSES = ["diabète de type 2", "hypertension artérielle", "pneumonie", "asthme", "migraine", "insuffisance cardiaque", "fibrillation auriculaire"]

EN_MEDS = [
    ("metformin", "500 mg", "BID", "PO", "for 14 days"),
    ("amoxicillin", "1 g", "TID", "PO", "for 7 days"),
    ("lisinopril", "10 mg", "QD", "PO", "for 30 days"),
    ("atorvastatin", "20 mg", "QD", "PO", "for 30 days"),
    ("azithromycin", "500 mg", "QD", "PO", "for 5 days"),
]
FR_MEDS = [
    ("metformine", "500 mg", "2x/j", "PO", "pendant 14 jours"),
    ("amoxicilline", "1 g", "3x/j", "PO", "pendant 7 jours"),
    ("lisinopril", "10 mg", "1x/j", "PO", "pendant 30 jours"),
    ("atorvastatine", "20 mg", "1x/j", "PO", "pendant 30 jours"),
    ("azithromycine", "500 mg", "1x/j", "PO", "pendant 5 jours"),
]

EN_LABS = [
    ("HbA1c", "7.4", "%"),
    ("creatinine", "1.2", "mg/dL"),
    ("CRP", "18", "mg/L"),
    ("hemoglobin", "13.5", "g/dL"),
    ("blood pressure", "140/90", "mmHg"),
]
FR_LABS = [
    ("HbA1c", "7.4", "%"),
    ("créatinine", "92", "µmol/L"),
    ("CRP", "18", "mg/L"),
    ("hémoglobine", "13.5", "g/dL"),
    ("tension artérielle", "140/90", "mmHg"),
]

EN_ANATOMY = ["right lower lobe", "left lung", "chest", "abdomen", "left knee"]
FR_ANATOMY = ["lobe inférieur droit", "poumon gauche", "thorax", "abdomen", "genou gauche"]

EN_PROCS = ["chest X-ray", "CT scan", "ECG", "echocardiogram"]
FR_PROCS = ["radiographie thoracique", "scanner thoracique", "ECG", "échocardiographie"]


def _build_dialogue(idx: int, lang: str, rng: random.Random) -> AnnotatedDocument:
    """Construct a single doctor-patient conversation with exact character-offset gold entities and relations."""
    parts: list[str] = []
    entities: list[EntitySpan] = []
    relations: list[RelationSpan] = []
    cursor = 0

    def add_text(chunk: str) -> None:
        nonlocal cursor
        parts.append(chunk)
        cursor += len(chunk)

    def add_ent(txt: str, etype: str, assertion: str = "present") -> EntitySpan:
        nonlocal cursor
        start = cursor
        parts.append(txt)
        cursor += len(txt)
        ent = EntitySpan(
            id=f"e{len(entities) + 1}",
            type=etype,
            start=start,
            end=cursor,
            text=txt,
            assertion=assertion,
        )
        entities.append(ent)
        return ent

    def add_rel(rtype: str, head: EntitySpan, tail: EntitySpan) -> None:
        relations.append(RelationSpan(id=f"r{len(relations) + 1}", type=rtype, head=head.id, tail=tail.id))

    if lang == "en":
        sym_pos, sym_neg = rng.sample(EN_SYMPTOMS, 2)
        diag_curr, diag_hist, diag_fam = rng.sample(EN_DIAGNOSES, 3)
        med_name, dose, freq, route, dur = rng.choice(EN_MEDS)
        lab_name, lab_val, lab_unit = rng.choice(EN_LABS)
        anat = rng.choice(EN_ANATOMY)
        proc = rng.choice(EN_PROCS)

        add_text("Doctor: Good morning, what brings you in today?\n")
        add_text("Patient: I have had ")
        e_sym_pos = add_ent(sym_pos, "SYMPTOM", "present")
        add_text(" in my ")
        e_anat = add_ent(anat, "ANATOMY", "present")
        add_rel("LOCATED_IN", e_sym_pos, e_anat)
        add_text(", but no ")
        add_ent(sym_neg, "SYMPTOM", "absent")
        add_text(".\n")

        add_text("Doctor: Any past medical or family history?\n")
        add_text("Patient: I have a history of ")
        add_ent(diag_hist, "DIAGNOSIS", "historical")
        add_text(", and my mother has ")
        add_ent(diag_fam, "DIAGNOSIS", "family")
        add_text(".\n")

        add_text("Doctor: Your ")
        e_lab = add_ent(lab_name, "LAB_TEST", "present")
        add_text(" is ")
        e_val = add_ent(lab_val, "LAB_VALUE", "present")
        add_text(" ")
        e_unit = add_ent(lab_unit, "LAB_UNIT", "present")
        add_rel("HAS_VALUE", e_lab, e_val)
        add_rel("HAS_UNIT", e_lab, e_unit)
        add_text(" and the ")
        e_proc = add_ent(proc, "PROCEDURE", "present")
        add_text(" suggests possible ")
        e_diag = add_ent(diag_curr, "DIAGNOSIS", "possible")
        add_rel("REVEALS", e_proc, e_diag)
        add_text(". Start ")
        e_med = add_ent(med_name, "MEDICATION", "present")
        add_text(" ")
        e_dose = add_ent(dose, "DOSAGE", "present")
        add_text(" ")
        e_freq = add_ent(freq, "FREQUENCY", "present")
        add_text(" ")
        e_route = add_ent(route, "ROUTE", "present")
        add_text(" ")
        add_ent(dur, "DURATION", "present")
        add_rel("HAS_DOSAGE", e_med, e_dose)
        add_rel("HAS_FREQUENCY", e_med, e_freq)
        add_rel("HAS_ROUTE", e_med, e_route)
        add_text(".\nPatient: Thank you, doctor.")
    else:
        sym_pos, sym_neg = rng.sample(FR_SYMPTOMS, 2)
        diag_curr, diag_hist, diag_fam = rng.sample(FR_DIAGNOSES, 3)
        med_name, dose, freq, route, dur = rng.choice(FR_MEDS)
        lab_name, lab_val, lab_unit = rng.choice(FR_LABS)
        anat = rng.choice(FR_ANATOMY)
        proc = rng.choice(FR_PROCS)

        add_text("Médecin : Bonjour, qu'est-ce qui vous amène aujourd'hui ?\n")
        add_text("Patient : J'ai de la ")
        e_sym_pos = add_ent(sym_pos, "SYMPTOM", "present")
        add_text(" au niveau du ")
        e_anat = add_ent(anat, "ANATOMY", "present")
        add_rel("LOCATED_IN", e_sym_pos, e_anat)
        add_text(", mais pas de ")
        add_ent(sym_neg, "SYMPTOM", "absent")
        add_text(".\n")

        add_text("Médecin : Avez-vous des antécédents personnels ou familiaux ?\n")
        add_text("Patient : J'ai un antécédent de ")
        add_ent(diag_hist, "DIAGNOSIS", "historical")
        add_text(" et ma mère a un ")
        add_ent(diag_fam, "DIAGNOSIS", "family")
        add_text(".\n")

        add_text("Médecin : Votre ")
        e_lab = add_ent(lab_name, "LAB_TEST", "present")
        add_text(" est à ")
        e_val = add_ent(lab_val, "LAB_VALUE", "present")
        add_text(" ")
        e_unit = add_ent(lab_unit, "LAB_UNIT", "present")
        add_rel("HAS_VALUE", e_lab, e_val)
        add_rel("HAS_UNIT", e_lab, e_unit)
        add_text(" et le ")
        e_proc = add_ent(proc, "PROCEDURE", "present")
        add_text(" évoque une suspicion de ")
        e_diag = add_ent(diag_curr, "DIAGNOSIS", "possible")
        add_rel("REVEALS", e_proc, e_diag)
        add_text(". Débutez ")
        e_med = add_ent(med_name, "MEDICATION", "present")
        add_text(" ")
        e_dose = add_ent(dose, "DOSAGE", "present")
        add_text(" ")
        e_freq = add_ent(freq, "FREQUENCY", "present")
        add_text(" ")
        e_route = add_ent(route, "ROUTE", "present")
        add_text(" ")
        add_ent(dur, "DURATION", "present")
        add_rel("HAS_DOSAGE", e_med, e_dose)
        add_rel("HAS_FREQUENCY", e_med, e_freq)
        add_rel("HAS_ROUTE", e_med, e_route)
        add_text(".\nPatient : Merci docteur.")

    doc = AnnotatedDocument(
        doc_id=f"synth_dlg_{lang}_{idx:03d}",
        patient_id=f"pat_dlg_{idx:03d}",
        language=lang,
        doc_type="consultation",
        document_date="2025-03-15",
        source="synthetic_doctor_patient_dialogue_stress_test",
        text="".join(parts),
        entities=entities,
        relations=relations,
        timex=[],
    )
    errs = doc.validate_spans()
    if errs:
        raise ValueError(f"Span validation failed on {doc.doc_id}: {errs[:3]}")
    return doc


def generate_100_dialogues(seed: int = 2026) -> list[AnnotatedDocument]:
    rng = random.Random(seed)
    docs: list[AnnotatedDocument] = []
    for i in range(1, 51):
        docs.append(_build_dialogue(i, "en", rng))
    for i in range(1, 51):
        docs.append(_build_dialogue(i, "fr", rng))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 100 synthetic EN/FR doctor-patient dialogues and benchmark models.")
    parser.add_argument("--out-jsonl", type=Path, default=Path("data/synthetic/dialogues_100/eval.jsonl"))
    parser.add_argument("--out-report", type=Path, default=Path("docs/results/raw/dialogue_100_synthetic_stress_eval.json"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    docs = generate_100_dialogues(seed=args.seed)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text(
        "\n".join(json.dumps(d.to_dict(), ensure_ascii=False) for d in docs) + "\n",
        encoding="utf-8",
    )
    dataset_sha256 = hashlib.sha256(args.out_jsonl.read_bytes()).hexdigest()

    svc = ExtractionService()
    hybrid = svc._get_hybrid()
    crf = svc._get_classical_crf()
    ev = Evaluator()

    en_idx = [i for i, d in enumerate(docs) if d.language == "en"]
    fr_idx = [i for i, d in enumerate(docs) if d.language == "fr"]

    results: dict[str, Any] = {}
    for name, predict_fn in [
        ("hybrid_v2", lambda ds: [hybrid.extract_document(d)[0] for d in ds]),
        ("classical_crf", lambda ds: [crf.predict_document(d) for d in ds]),
    ]:
        t0 = time.perf_counter()
        preds = predict_fn(docs)
        elapsed_s = round(time.perf_counter() - t0, 2)
        m_all = ev.evaluate(docs, preds)
        m_en = ev.evaluate([docs[i] for i in en_idx], [preds[i] for i in en_idx])
        m_fr = ev.evaluate([docs[i] for i in fr_idx], [preds[i] for i in fr_idx])
        results[name] = {
            "elapsed_seconds": elapsed_s,
            "ms_per_dialogue": round((elapsed_s * 1000) / len(docs), 1),
            "overall": {
                "strict_span": m_all["strict_span"],
                "lenient_span": m_all["lenient_span"],
                "assertion": m_all["assertion"],
                "relations": m_all["relations"],
            },
            "en": {
                "strict_span": m_en["strict_span"],
                "lenient_span": m_en["lenient_span"],
                "assertion": m_en["assertion"],
                "relations": m_en["relations"],
            },
            "fr": {
                "strict_span": m_fr["strict_span"],
                "lenient_span": m_fr["lenient_span"],
                "assertion": m_fr["assertion"],
                "relations": m_fr["relations"],
            },
        }

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": "synthetic_doctor_patient_dialogue_stress_test",
        "synthetic": True,
        "owner_written": False,
        "seed": args.seed,
        "dataset_path": str(args.out_jsonl),
        "dataset_sha256": dataset_sha256,
        "n_dialogues": len(docs),
        "languages": {"en": 50, "fr": 50},
        "gold_counts": {
            "entities": sum(len(d.entities) for d in docs),
            "relations": sum(len(d.relations) for d in docs),
        },
        "models": results,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for name, r in results.items():
        ov = r["overall"]
        print(
            f"[{name}] {r['ms_per_dialogue']} ms/dlg | "
            f"Strict F1={ov['strict_span']['f1']:.4f} (EN={r['en']['strict_span']['f1']:.4f}, FR={r['fr']['strict_span']['f1']:.4f}) | "
            f"Lenient F1={ov['lenient_span']['f1']:.4f} | "
            f"Assert Acc={ov['assertion']['accuracy']:.4f} | "
            f"Rel F1={ov['relations']['f1']:.4f} (P={ov['relations']['precision']:.4f}, R={ov['relations']['recall']:.4f})"
        )


if __name__ == "__main__":
    main()
