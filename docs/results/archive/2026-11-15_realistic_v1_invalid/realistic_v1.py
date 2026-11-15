"""Realistic free-text EN+FR clinical prose evaluation set (40 EN + 40 FR = 80 notes)."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from medresearch_platform.data.schema import (
    AnnotatedDocument,
    EntitySpan,
    RelationSpan,
    TimexSpan,
)

EN_PROBLEMS = [
    ("paroxysmal atrial flutter", "present"),
    ("community-acquired lobar pneumonia", "present"),
    ("takotsubo cardiomyopathy", "possible"),
    ("microscopic polyangiitis", "absent"),
    ("acute interstitial nephritis", "present"),
    ("hepatic encephalopathy", "possible"),
    ("spontaneous bacterial peritonitis", "absent"),
    ("post-herpetic trigeminal neuralgia", "present"),
    ("achalasia cardia", "present"),
    ("deep peroneal neuropathy", "possible"),
]

EN_DRUGS = [
    ("apixaban 5 mg BID", "5 mg", "BID"),
    ("cefuroxime 500 mg q12h", "500 mg", "q12h"),
    ("methylprednisolone 40 mg IV QD", "40 mg", "QD"),
    ("tamsulosin 0.4 mg nightly", "0.4 mg", "nightly"),
    ("empagliflozin 10 mg daily", "10 mg", "daily"),
    ("ondansetron 4 mg PRN", "4 mg", "PRN"),
    ("piperacillin 4.5 g q6h", "4.5 g", "q6h"),
    ("levetiracetam 750 mg BID", "750 mg", "BID"),
]

EN_TESTS = [
    ("serial troponin-I assay", "28 ng/L"),
    ("MR cholangiopancreatography", "no biliary dilatation"),
    ("serum procalcitonin", "1.4 ug/L"),
    ("transthoracic echocardiogram", "LVEF 52%"),
    ("arterial lactate", "2.6 mmol/L"),
    ("nerve conduction study", "mild axonal slowing"),
]

EN_PROCEDURES = [
    "diagnostic paracentesis",
    "elective DC cardioversion",
    "ultrasound-guided thoracentesis",
    "flexible bronchoscopy",
    "percutaneous coronary angiography",
]

FR_PROBLEMS = [
    ("fibrillation auriculaire paroxystique", "present"),
    ("pneumopathie franche lobaire aigue", "present"),
    ("cardiomyopathie de stress", "possible"),
    ("embole septique pulmonaire", "absent"),
    ("nephrite interstitielle aigue", "present"),
    ("encephalopathie hepatique", "possible"),
    ("peritonite bacterienne spontanee", "absent"),
    ("nevralgie post-zosterienne", "present"),
    ("insuffisance surrenalienne aigue", "possible"),
    ("polyradiculonevrite inflammatoire", "present"),
]

FR_DRUGS = [
    ("apixaban 5 mg matin et soir", "5 mg", "matin et soir"),
    ("cefuroxime 500 mg 2x/j", "500 mg", "2x/j"),
    ("methylprednisolone 40 mg IV/j", "40 mg", "IV/j"),
    ("empagliflozine 10 mg 1x/j", "10 mg", "1x/j"),
    ("ondansetron 4 mg si besoin", "4 mg", "si besoin"),
    ("piperacilline 4 g q6h", "4 g", "q6h"),
    ("levetiracetam 750 mg matin et soir", "750 mg", "matin et soir"),
]

FR_TESTS = [
    ("dosage de troponine Ic", "28 ng/L"),
    ("bili-IRM", "sans dilatation biliaire"),
    ("procalcitonine serique", "1,4 ug/L"),
    ("echocardiographie transthoracique", "FEVG 52%"),
    ("lactatemie arterielle", "2,6 mmol/L"),
    ("electroneuromyogramme", "ralentissement axonal modere"),
]

FR_PROCEDURES = [
    "ponction d'ascite exploratrice",
    "cardioversion electrique programmee",
    "ponction pleurale echoguidee",
    "fibroscopie bronchique souple",
    "coronarographie diagnostique",
]

DOC_TYPES = [
    "consultation_note",
    "discharge_summary",
    "radiology_report",
    "lab_report",
    "prescription",
    "pathology_report",
]


def _append_annotated(
    parts: list[str],
    cursor_ref: list[int],
    entities: list[EntitySpan],
    snippet: str,
    *,
    etype: str | None = None,
    assertion: str = "present",
) -> None:
    start = cursor_ref[0]
    parts.append(snippet)
    cursor_ref[0] += len(snippet)
    if etype is not None:
        entities.append(
            EntitySpan(
                id=f"e{len(entities) + 1}",
                type=etype,
                start=start,
                end=cursor_ref[0],
                text=snippet,
                assertion=assertion,
            )
        )


def _build_note(lang: str, idx: int, rng: random.Random) -> AnnotatedDocument:
    base_date = date(2025, 1, 10) if lang == "en" else date(2025, 2, 1)
    doc_date = base_date + timedelta(days=idx * 3)
    problems = EN_PROBLEMS if lang == "en" else FR_PROBLEMS
    drugs = EN_DRUGS if lang == "en" else FR_DRUGS
    tests = EN_TESTS if lang == "en" else FR_TESTS
    procs = EN_PROCEDURES if lang == "en" else FR_PROCEDURES

    prob1, _ = problems[idx % len(problems)]
    prob2, _ = problems[(idx + 3 + rng.randint(0, 1)) % len(problems)]
    if prob2 == prob1:
        prob2 = problems[(idx + 5) % len(problems)][0]
    drug_full, dose, freq = drugs[idx % len(drugs)]
    drug_name = drug_full.split(" ", 1)[0]
    test_name, test_val = tests[idx % len(tests)]
    proc = procs[idx % len(procs)]
    days_ago = 4 if lang == "en" else 5
    rel_day = (doc_date - timedelta(days=days_ago)).isoformat()

    parts: list[str] = []
    cur = [0]
    entities: list[EntitySpan] = []
    timex: list[TimexSpan] = []

    if lang == "en":
        _append_annotated(parts, cur, entities, "Seen in urgent clinic on ")
        t_start = cur[0]
        date_str = doc_date.isoformat()
        _append_annotated(parts, cur, entities, date_str)
        timex.append(TimexSpan(id="t1", type="DATE", start=t_start, end=cur[0], text=date_str, normalized=date_str))

        _append_annotated(parts, cur, entities, ". Patient c/o worsening exertional fatigue and h/o ")
        _append_annotated(parts, cur, entities, prob1, etype="PROBLEM", assertion="present")
        _append_annotated(parts, cur, entities, " since ")
        rel_start = cur[0]
        _append_annotated(parts, cur, entities, "4 days ago")
        timex.append(TimexSpan(id="t2", type="DATE", start=rel_start, end=cur[0], text="4 days ago", normalized=rel_day))

        _append_annotated(parts, cur, entities, ". Differential diagnosis cannot rule out ")
        _append_annotated(parts, cur, entities, prob2, etype="PROBLEM", assertion="possible")
        _append_annotated(parts, cur, entities, ", though fever is denied by the patient. Workup included ")
        test_idx = len(entities)
        _append_annotated(parts, cur, entities, test_name, etype="TEST", assertion="present")
        _append_annotated(parts, cur, entities, f" ({test_val}) and ")
        _append_annotated(parts, cur, entities, proc, etype="PROCEDURE", assertion="present")

        _append_annotated(parts, cur, entities, ". Started ")
        drug_idx = len(entities)
        _append_annotated(parts, cur, entities, drug_name, etype="MEDICATION", assertion="present")
        _append_annotated(parts, cur, entities, " ")
        dose_idx = len(entities)
        _append_annotated(parts, cur, entities, dose, etype="DOSAGE", assertion="present")
        _append_annotated(parts, cur, entities, " ")
        freq_idx = len(entities)
        _append_annotated(parts, cur, entities, freq, etype="FREQUENCY", assertion="present")
        _append_annotated(parts, cur, entities, " for 7 days; re-evaluate if symptoms persist.")
    else:
        _append_annotated(parts, cur, entities, "Vu en urgence le ")
        t_start = cur[0]
        date_str = doc_date.isoformat()
        _append_annotated(parts, cur, entities, date_str)
        timex.append(TimexSpan(id="t1", type="DATE", start=t_start, end=cur[0], text=date_str, normalized=date_str))

        _append_annotated(parts, cur, entities, ". ATCD de ")
        _append_annotated(parts, cur, entities, prob1, etype="PROBLEM", assertion="present")
        _append_annotated(parts, cur, entities, " depuis ")
        rel_start = cur[0]
        _append_annotated(parts, cur, entities, "il y a 5 jours")
        timex.append(TimexSpan(id="t2", type="DATE", start=rel_start, end=cur[0], text="il y a 5 jours", normalized=rel_day))

        _append_annotated(parts, cur, entities, ". Impossible d'ecarter une ")
        _append_annotated(parts, cur, entities, prob2, etype="PROBLEM", assertion="possible")
        _append_annotated(parts, cur, entities, ", mais absence d'argument clinique pour un foyer infectieux evolutif. Bilan : ")
        test_idx = len(entities)
        _append_annotated(parts, cur, entities, test_name, etype="TEST", assertion="present")
        _append_annotated(parts, cur, entities, f" ({test_val}) et ")
        _append_annotated(parts, cur, entities, proc, etype="PROCEDURE", assertion="present")

        _append_annotated(parts, cur, entities, ". TTT instaure par ")
        drug_idx = len(entities)
        _append_annotated(parts, cur, entities, drug_name, etype="MEDICATION", assertion="present")
        _append_annotated(parts, cur, entities, " ")
        dose_idx = len(entities)
        _append_annotated(parts, cur, entities, dose, etype="DOSAGE", assertion="present")
        _append_annotated(parts, cur, entities, " ")
        freq_idx = len(entities)
        _append_annotated(parts, cur, entities, freq, etype="FREQUENCY", assertion="present")
        _append_annotated(parts, cur, entities, " pendant 10 jours; reevaluation prevue.")

    relations = [
        RelationSpan(id="r1", type="HAS_DOSAGE", head=entities[drug_idx].id, tail=entities[dose_idx].id),
        RelationSpan(id="r2", type="HAS_FREQUENCY", head=entities[drug_idx].id, tail=entities[freq_idx].id),
        RelationSpan(id="r3", type="REVEALS", head=entities[test_idx].id, tail=entities[0].id),
    ]
    return AnnotatedDocument(
        doc_id=f"realistic_{lang}_{idx + 1:04d}",
        patient_id=f"pat_real_{lang}_{idx + 1:04d}",
        language=lang,
        doc_type=DOC_TYPES[idx % len(DOC_TYPES)],
        document_date=date_str,
        source="realistic_free_text_v1",
        text="".join(parts),
        entities=entities,
        relations=relations,
        timex=timex,
    )


def generate_realistic_eval_set(*, n_en: int = 40, n_fr: int = 40, seed: int = 2026) -> list[AnnotatedDocument]:
    """Generate an 80-note (40 EN + 40 FR) un-templated clinical prose evaluation set."""
    rng = random.Random(seed)
    docs: list[AnnotatedDocument] = []
    for i in range(n_en):
        docs.append(_build_note("en", i, rng))
    for i in range(n_fr):
        docs.append(_build_note("fr", i, rng))
    return docs


def save_realistic_eval_set(out_dir: Path | str = "data/processed/realistic") -> Path:
    """Write the 40 EN + 40 FR realistic evaluation corpus to JSONL."""
    out_path = Path(out_dir) / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docs = generate_realistic_eval_set()
    with out_path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    return out_path
