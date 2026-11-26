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

## Step 3 — P1 Clinical Context, Section Awareness & NegEx Fix (`2026-11-26T10:00Z`)
- Diagnosed the `val.jsonl` assertion accuracy regression (`0.9488 -> 0.9266`): `post_window` matching French `"si "` in `"si besoin"` caused 65 false `hypothetical` predictions, and an initial `section == "pmh"` override caused 267 EN `present -> historical` flips.
- Fixed `medresearch_platform/approaches/context_negex.py`:
  - Added canonical section detector `detect_section_at_offset(text, offset)` (`hpi`, `pmh`, `family_history`, `medications`, `allergies`, `vitals`, `exam`, `labs`, `imaging`, `assessment`, `plan`, `return_precautions`, `general`) supporting bulleted/markdown headers.
  - Extended `AssertionAttributes` (backward-compatible) with `experiencer_detail` (`patient`, `mother`, `father`, `brother`, `sister`, `other_family`), `assertion_detail` (`present`, `absent`, `possible`, `suspected`, `uncertain`, `conditional`, `planned`, `discontinued`, `unresolved`), `temporality_detail` (`current`, `historical`, `resolved`, `recurrent`, `future_planned`), and `section`.
  - Added EN/FR pseudo-negation & clause-break guards (`"sans rendez-vous"`, `"not excluded"`, `"not excludcd"`, `"cannot be excluded"`, `_POST_CLAUSE_BREAK_RE` breaking on `, though / negative for / positive for / no / denies`) and unperformed-scan safety guard (`_UNPERFORMED_SCAN_RE`).
- **Measured Result on `val.jsonl` (`n=160`):**
  - Overall Assertion Acc: `0.9266 -> 0.9471` (+2.05 pp); Assertion Macro-F1: `0.7416 -> 0.8036` (+6.20 pp).
  - EN Assertion Acc: `0.9484 -> 0.9557` (+0.73 pp); Macro-F1: `0.8241 -> 0.8520` (+2.79 pp).
  - FR Assertion Acc: `0.9032 -> 0.9378` (+3.46 pp); Macro-F1: `0.6778 -> 0.7666` (+8.88 pp).
- Committed cleanly as `2336171`.

## Step 4 — P2 Structured Meds/Labs/Vitals & P3 Contradictions + OCR Robustness (`2026-11-26T10:04Z`)
- Enhanced `medresearch_platform/serving/extractor.py` (ZERO new models):
  - **Labs & Vitals:** Extended `_extract_structured_vitals_and_labs` with reference range parsing (`(reference <19 ng/L)` -> `reference_range` + `interpretation: within_range | high | low | uncertain`), SpO2 oxygen condition (`oxygen_condition: room_air | O2 ...`), and repeated timestamp capture (`measurement_time`). Fixed CRF misclassifications (`SpO2` as `LAB_VALUE`, `168/96` as `LAB_UNIT`, `air` as `MEDICATION`, `Pain | 7/10` as `DATE`).
  - **Medications:** Added `medication_status` (`ordered_not_administered`, `not_taking`, `discontinued`, `prn_active` + `last_administered_temporality: historical`, `uncertain`, `active`) and `normalized_frequency` (`OD` -> `once daily`, never `ROUTE`).
  - **OCR Normalization:** Added `_normalize_ocr_clinical_token` preserving `original_text`, setting `ocr_corrupted=True`, `certainty="possible"`, `needs_review=True`, and non-fabricating `normalized_candidate` (`None` whenever `?` is present).
  - **Contradiction Detection:** Added `detect_clinical_contradictions(records, text)` preserving both sides with spans for allergy conflicts (`NKDA` vs affirmed allergy), diagnosis/symptom polarity conflicts, conflicting medication doses, and unperformed scan vs negative imaging statements.
- Committed cleanly as `f8ca71b`.

## Step 5 — P4 Vision Verification, P5 EN/FR Calibration, Provisional Refs & Tests (`2026-11-26T10:09Z`)
- Saved `reports/eval/after_val_en_fr.json` and `reports/eval/vision_p4_verification.json` (`se_resnet_v2` 1.23M params, `convnext_derm` 2.49M params: finite loss, non-zero gradients, deterministic CPU eval verified; locked_test_v4 dermoscopic vs clinical vs tbp_tile reported separately).
- Created `tests/fixtures/noise_0027_ref_draft.json` and `tests/fixtures/noise_0038_ref_draft.json` (`status: "draft_unreviewed"`, derived strictly from document text).
- Created `tests/test_p1_p3_clinical_fixtures_and_ocr_noise.py` (all 18 fixture cases, all 8 OCR noise variants, empty/whitespace/determinism checks).
- Saved `reports/eval/noise_0027_0038_provisional_eval.json`: NOISE-0027 `13/15` (`86.67%`), NOISE-0038 `13/15` (`86.67%`, 7 contradictions detected) — explicitly marked provisional.
- Committed cleanly as `9ac6432`.

## Step 6 — Final QA & Report (`2026-11-26T10:15Z`)
- `ruff check`: `All checks passed!`
- `mypy --ignore-missing-imports`: `Success: no issues found in 4 source files`
- Clean import check: `IMPORT_OK`
- `pytest` (7 suites: P1–P3 fixtures/OCR, Phase 4 NLP, NOISE-0027, NOISE-0038, API/CLI, Phase 1 P1–P9, Vision Step 3): **`48 passed` in 274.57s** (`0` failures).
- Frontend `tsc --noEmit && vite build`: passed (`6.28s`); Vitest unit tests (`src/`): **`13 passed`** (`0` failures).
- Authored `reports/FINAL_REPORT.md` (Sections A–I).

---

## Optimization Pass v2 (`feat/existing-models-optimization-v2`, `2026-11-26T11:56Z–15:15Z`)
- **C2 Pretrained Audit:** Verified `39/39` model manifests (`0` pretrained), `torchvision`/`timm`/`transformers`/`spacy` NOT installed, `~/.cache/torch` absent.
- **P0 Re-measured Baseline & Leakage (`reports/baseline/v2_remeasured_baseline.json`):** Reproduced `SYNTH-MED-V3.1 val.jsonl` (`n=160`: Strict F1 `0.9932`, Assert Acc `0.9471`, Macro-F1 `0.8036`, Rel F1 `0.9531`, ECE `0.0060`) and `open_clinical/test.jsonl` (`73/443` exact `PxCorpus` surface overlap with `train.jsonl`; de-overlapped `n=370` Strict F1 `0.1297` / Lenient F1 `0.2953`).
- **P1–P4 Fixes (`c6785e1`):** Generalizable CRF vital/OCR fragment coalescing (`142/86`, `15B/9?`, `118/74 mmHg`, `12?/7B`), extended same-line return-precaution lookahead (EN/FR, up to 120 chars), pain-score adverbial guard (`Pain now 3/10`), and scope-aware contradiction detection (fixing the false cross-section Aspirin dose collision into `medication_order_vs_administration:aspirin`). Provisional NOISE-0027 `15/15`, NOISE-0038 `15/15`; **20/20** held-out synthetic corruption variants pass (`tests/test_p1_p4_heldout_generalization_variants.py`). Logged `n27_07` context ordering proposal in `reports/eval/reference_review_requests.md` (draft refs untouched per C7).
- **P5 Vision Smartphone Gap Investigation (`reports/eval/vision_smartphone_gap_hypotheses.json`, `reports/eval/vision_p5_smartphone_experiment.json`):** Quantified `3.5×` Laplacian variance (`425.2` vs `121.4`) and `−37.5°` peripheral ITA° shift (`15.3°` vs `52.8°`) on clinical smartphone vs dermoscopic. Tested 80% center-crop TTA tuned on `val_v3` (`n=352`) -> `locked_test_v4` clinical (`n=571`): `Sens@90%Spec 0.5449 -> 0.5739` (`+2.90 pp`), **inside 95% bootstrap CI `[0.4256, 0.6887]`** (classified as no significant gain; served calibration untouched). Subgroup analysis isolated the gap to `Fitzpatrick17k` (`n=368`, `Sens@90%Spec 0.2695`) vs `ISIC clinical` (`n=203`, `Sens@90%Spec 0.9888`).
- **P6 & QA:** `ruff` clean (`0` errors), `mypy` extended to **`30` source files (`0` errors)**, `pytest` **`68 passed`** (`312.36s`), frontend build + `13` Vitest passed. Authored `reports/eval/existing_models_optimization_report.md`.
