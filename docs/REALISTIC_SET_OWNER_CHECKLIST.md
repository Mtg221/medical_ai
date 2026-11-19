# Realistic Free-Text Evaluation Set — Owner Checklist

Status: **Waiting for owner-written & double-annotated notes** (see [`docs/REALISTIC_SET_PROTOCOL.md`](./REALISTIC_SET_PROTOCOL.md)).  
Note: `data/processed/realistic_llm/eval.jsonl` (`docs/results/realistic_llm_80_eval.json`) is an LLM-authored single-annotator proxy (`owner_written: false`, `double_annotated_iaa: false`) and does **not** replace human clinician/owner notes.

## 1. Corpus Composition
- [ ] **40 English notes + 40 French notes** (`80` documents total, `test_realistic`).
- [ ] Zero real PHI/PII (synthetic/fictional patient identities and dates only).
- [ ] Cover all 6 document types evenly (`discharge_summary`, `consultation`, `lab_report`, `prescription`, `radiology_report`, `operative_note`).
- [ ] Write genuine, varied clinical prose styles (no repeated sentence templates):
  - concise telegraphic ED/ICU shorthand (`c/o`, `h/o`, `s/p`, `ATCD`, `TTT`, `PEC`),
  - structured consultation summaries,
  - complex multi-clause negation and uncertainty (`"unlikely given..."`, `"possible ... vs ..."`, `"absence d'argument pour..."`, `"à réévaluer"`),
  - non-standard date/duration/frequency/relative phrasings (e.g. `"early Oct"`, `"mi-janvier"`, `"q8h"`, `"2x/j"`, `"il y a quinze jours"`).

## 2. Annotation Protocol (Entities, Assertions, Relations, TIMEX & IAA)
- [ ] Load [`configs/label_studio_medical_en_fr.xml`](../configs/label_studio_medical_en_fr.xml) into Label Studio.
- [ ] Use **only** the 14 canonical entity types (`SYMPTOM`, `DIAGNOSIS`, `MEDICATION`, `DOSAGE`, `FREQUENCY`, `ROUTE`, `DURATION`, `LAB_TEST`, `LAB_VALUE`, `LAB_UNIT`, `PROCEDURE`, `ANATOMY`, `DATE`, `PATIENT_ID`), 6 assertion classes (`present`, `absent`, `possible`, `hypothetical`, `family`, `historical`), 9 relation types (`TREATS`, `HAS_DOSAGE`, `HAS_FREQUENCY`, `HAS_ROUTE`, `HAS_VALUE`, `HAS_UNIT`, `REVEALS`, `CAUSES`, `LOCATED_IN`), and `RELATIVE` timex spans.
- [ ] Annotate exact minimal noun-phrase spans without trailing punctuation or articles.
- [ ] Perform **20%+ double annotation** (`>= 16` notes: 8 EN + 8 FR) by a second reviewer inside the same Label Studio project so each double-annotated task carries `>= 2` entries in `task["annotations"]`.

## 3. Import, Compute IAA & Evaluate
1. Import, validate entities/relations/TIMEX, and compute Inter-Annotator Agreement (strict/lenient entity F1, assertion Cohen's kappa, relation F1, TIMEX F1):
   ```bash
   python scripts/import_label_studio_realistic.py \
     --input path/to/label_studio_export.json \
     --output data/processed/realistic/eval.jsonl \
     --iaa-out docs/results/realistic_iaa.json
   ```
2. Run evaluation across all approaches (including `hybrid_v2`) and log the run in `docs/results/TEST_SET_LOG.md`:
   ```bash
   python scripts/eval_realistic.py \
     --realistic data/processed/realistic/eval.jsonl \
     --iaa-json docs/results/realistic_iaa.json \
     --owner-written \
     --out docs/results/realistic_eval.json
   ```

## 4. Unseen Human PHI Redaction Stress Test (`>= 50` EN/FR Sentences)
- [ ] Author `>= 50` unseen EN/FR clinical sentences containing synthetic PHI + `>= 10` tricky non-PHI eponym/lab sentences following [`docs/PHI_REDACTION_OWNER_TEMPLATE.jsonl`](./PHI_REDACTION_OWNER_TEMPLATE.jsonl).
- [ ] Run the strict full-token redaction evaluator once (unseen):
  ```bash
  python scripts/eval_redaction_human.py \
    --input path/to/owner_phi_50.jsonl \
    --owner-written \
    --out docs/results/p10_redaction_owner_eval.json
  ```
