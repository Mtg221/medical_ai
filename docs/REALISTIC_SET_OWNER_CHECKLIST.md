# Realistic Free-Text Evaluation Set — Owner Checklist

Status: **Waiting for owner-written & annotated notes** (see [`docs/REALISTIC_SET_PROTOCOL.md`](./REALISTIC_SET_PROTOCOL.md)).

## 1. Corpus Composition
- [ ] **40 English notes + 40 French notes** (`80` documents total, `test_realistic`).
- [ ] Zero PHI/PII (synthetic/fictional patient identities and dates only).
- [ ] Cover all 6 document types evenly (`discharge_summary`, `consultation`, `lab_report`, `prescription`, `radiology_report`, `operative_note`).
- [ ] Write genuine, varied clinical prose styles (no repeated sentence templates):
  - concise telegraphic ED/ICU shorthand (`c/o`, `h/o`, `s/p`, `ATCD`, `TTT`, `PEC`),
  - structured consultation summaries,
  - complex multi-clause negation and uncertainty (`"unlikely given..."`, `"possible ... vs ..."`, `"absence d'argument pour..."`, `"à réévaluer"`),
  - non-standard date/duration/frequency phrasings (e.g. `"early Oct"`, `"mi-janvier"`, `"q8h"`, `"2x/j"`).

## 2. Annotation Protocol
- [ ] Load [`configs/label_studio_medical_en_fr.xml`](../configs/label_studio_medical_en_fr.xml) into Label Studio.
- [ ] Use **only** the 14 canonical entity types (`SYMPTOM`, `DIAGNOSIS`, `MEDICATION`, `DOSAGE`, `FREQUENCY`, `ROUTE`, `DURATION`, `LAB_TEST`, `LAB_VALUE`, `LAB_UNIT`, `PROCEDURE`, `ANATOMY`, `DATE`, `PATIENT_ID`) and 6 assertion classes (`present`, `absent`, `possible`, `hypothetical`, `family`, `historical`).
- [ ] Annotate exact minimal noun-phrase spans without trailing punctuation or articles.
- [ ] Perform **20% double annotation** (`16` notes: 8 EN + 8 FR) by a second reviewer and record inter-annotator Cohen's kappa / span F1.

## 3. Import & Run
1. Import & validate labels:
   ```bash
   python scripts/import_label_studio_realistic.py --input path/to/label_studio_export.json --output data/processed/realistic/eval.jsonl
   ```
2. Run evaluation across all approaches and log the run in `docs/results/TEST_SET_LOG.md`:
   ```bash
   python scripts/eval_realistic.py --realistic data/processed/realistic/eval.jsonl --out docs/results/realistic_eval.json
   ```
