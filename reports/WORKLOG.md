# Engineering Worklog — Existing Model Improvements (`feat/model-improvements-p0-p5`)

**Branch:** `feat/model-improvements-p0-p5` (created from `chore/completion-audit` @ `539e530`)  
**Constraints:** C1 No new models · C2 100% from-scratch · C3 Intel CPU 32 GB RAM · C4 Honesty · C5 Eval integrity · C6 Compatibility · C7 New branch, no push.

---

## Step 1 — Baseline Capture (`2026-11-26T08:28Z`)
- Captured per-model inventory, load time, inference speed, RSS memory, and EN/FR `val.jsonl` (`SYNTH-MED-V3.1`, `n=160`) metrics into `reports/baseline/model_inventory_and_baseline.json`.
- **Key Baseline Discovery:** `hybrid_v2` on `val.jsonl` (`n=160`):
  - Entity Strict F1 = `0.9932`, Lenient F1 = `0.9953`, Relations F1 = `0.9531` (`P 0.9765`, `R 0.9308`).
  - **Regression detected from earlier ConText cue expansion:** Assertion Accuracy = `0.9266` / Macro F1 = `0.7416` (down from `0.9488` / `0.7988` before NOISE-0027/0038 cue additions). Will diagnose and restore/improve in Step 3 (P1).
  - CPU Profile: `hybrid_v2` load `33.0 ms`, `53.2 ms/doc` on `val.jsonl`, RSS `140.6 -> 161.7 MB`.
  - Checkpoints: `30/30` verified (`344.6 MB`, `0` pretrained, `0` invalid).

## Step 2 — P0 Evaluation Validity, Patient Identity & External Derm Check (`2026-11-26T08:28Z`)
- **Patient ID Root Cause Confirmed:** Hardcoded `subject_id="subj_synth_001"` defaults still existed in `medresearch_platform/interfaces/routers/documents.py:267` (`Form("subj_synth_001")`) and `medresearch_platform/interfaces/schemas.py:29` (`subject_id: str = "subj_synth_001"`).
- **Fix:** Changing both defaults to `""` so `MedicalResearchPlatformService.ingest_document` automatically assigns a unique `subj_<uuid>` per independent document unless the caller explicitly provides a subject ID.
- **Splits & Leakage Audit:**
  - `SYNTH-MED-V3.1` (`LOCK_MANIFEST.json`): 4/4 SHA-256 verified, `0` leakage.
  - `open_clinical/test.jsonl`: `73/443` `PxCorpus` surface-text overlap with `train.jsonl` already flagged as non-definitive.
  - Vision `locked_test_v4` (`n=1,611`): `0` exact SHA-256 duplicates vs `train_v3`; `v3` `dHash <= 4` near-duplicate rate (`25.49%`) documented.
- **External Dermatology Ingestion (`PAD-UFES-20`, `Fitzpatrick17k`, `DDI`):** `scripts/ingest_external_clinical_derm.py` verified ready with `--license` tracking and SHA-256 dedup; datasets not present locally (institutional access/licenses required — never downloaded without authorization).
