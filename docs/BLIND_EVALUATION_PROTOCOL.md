# Blind Evaluation & Test-Set Freeze Protocol (`DATA-001`)

**Effective Date:** 2026-11-24  
**Baseline Commit:** `ff8e656`  
**Status:** Active — Existing Evaluation Splits Frozen as Development History

---

## 1. Freeze of Historical Evaluation Splits

Because `SYNTH-MED-V3.1`, `open_clinical_test`, `realistic_llm_80`, and `locked_test_v3`/`v4` have been evaluated across `~52` historical runs (`docs/results/TEST_SET_LOG.md`), they are hereby **reclassified as Development / Regression History** and **must not** be presented as untouched final blind test sets:

- `SYNTH-MED-V3.1` (`test_iid`, `test_unseen_terms`, `test_unseen_templates`) — frozen regression benchmark
- `realistic_llm_80` (LLM-authored, `owner_written: false`) — frozen proxy benchmark
- `open_clinical_test` (`PxCorpus`, `MACCROBAT2020`, `E3C Layer 1`) — frozen open-clinical benchmark
- `configs/vision/locked_test_v3.txt` & `locked_test_v4.txt` — frozen retrospective derm benchmarks

**Rule:** Do not tune new architectures, confidence thresholds, or hyperparameters using scores from these frozen splits without logging the run as `post_hoc: true`.

---

## 2. Sealed Independent Blind Evaluation Sets

True independent validation requires **two completely sealed human-authored evaluation sets** that remain physically outside the development tree (`var/blind_eval/`, gitignored) until the system, weights, and thresholds are frozen:

1. **Human Clinical Free-Text Set (`n = 80`):** 40 English + 40 French clinician/owner-authored notes (`owner_written: true`, `>= 20%` double-annotated for IAA in `configs/label_studio_medical_en_fr.xml`) covering entities, 3-axis assertions (`certainty`, `temporality`, `experiencer`), relations, and TIMEX.
2. **Unseen Human PHI Redaction Set (`n = 60`):** `>= 50` unseen EN/FR PHI sentences + `>= 10` tricky non-PHI medical eponyms (`docs/PHI_REDACTION_OWNER_TEMPLATE.jsonl`).

---

## 3. Pre-Registered One-Shot Evaluation Procedure

1. **Freeze Artifact State:** Record exact `git rev-parse HEAD`, `docs/results/ARTIFACTS_MANIFEST.json` SHA-256 hashes, and locked operating thresholds before touching the blind sets.
2. **Run Exactly Once:** Execute:
   - `python scripts/eval_realistic.py --input var/blind_eval/owner_notes_80.jsonl --owner-written`
   - `python scripts/eval_redaction_human.py --input var/blind_eval/owner_phi_60.jsonl --owner-written`
3. **Log Unmodified Outcome:** Append the result to `docs/results/TEST_SET_LOG.md` with `"post_hoc": false` regardless of whether scores rise or fall. Never delete or overwrite a logged blind run.
