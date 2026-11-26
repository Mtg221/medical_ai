# Existing Models Optimization Report (P0–P6)

**Repository:** `github.com/Mtg221/medical_ai`  
**Branch:** `feat/existing-models-optimization-v2` (branched from `4f5e458`, **not pushed**)  
**Hardware:** Intel Core i7-9750H (CPU-only, macOS `x86_64`), 32 GB RAM, Python 3.12.16 (`.venv-vision`), PyTorch 2.2.2  
**Date:** 2026-11-26  

---

## 1. Codebase & C2 From-Scratch Audit Findings

### C2 Pretrained-Weights Verification (Code + Environment + Checkpoints)
- **Installed packages audit (`.venv-vision`):** `torch 2.2.2`, `sklearn-crfsuite 0.5.0`, `scikit-learn 1.5.2`, `numpy 1.26.4`, `scipy 1.14.1`. **`torchvision`, `timm`, `transformers`, and `spacy` are NOT installed.**
- **Source-code grep (`medresearch_platform/`, `scripts/`):** `0` occurrences of `from_pretrained`, `torch.hub`, `timm.`, `transformers.`, `download=True`, or pretrained `weights=`.
- **Cache check:** `~/.cache/torch` does not exist; `~/.cache/huggingface` contains only `datasets/` (no pretrained model hub weights).
- **Checkpoint manifests (`models/**/manifest.json`):** `39/39` manifests checked (`30` active production/benchmark checkpoints + `9` ablation/specialist variants); `0` have `pretrained_weights_used == True`.

### End-to-End Pipeline Audit Summary
| Component | Status | Audit Finding |
|---|---|---|
| `approaches/tokenizer.py` (`_TOKEN_RE`) | Implemented | Splits `142/86` into `["142", "/", "86"]` and `15B/9?` into `["15", "B", "/", "9", "?"]`, causing CRF to fragment or mistype vital ratios (e.g., `118/74 mmHg` tagged as `DATE`, `B` in `12?/7B` tagged as `PATIENT_ID`). |
| `approaches/context_negex.py` | Implemented | `_POST_SEEK_URGENT_RE` previously scanned only a 45-char `post_window`, missing urgent return-precaution instructions >45 chars away when headers are garbled/missing. |
| `hybrid/stage4_temporal.py` | Implemented | Relative date grounding works (`"three days ago"` -> `2026-11-23`, `"last Friday"` -> `2026-11-20` anchored to `2026-11-26`). |
| `serving/extractor.py` | Implemented | Previously reused partial CRF `LAB_VALUE` spans instead of coalescing split ratio/OCR spans; `_PAIN_PREFIX_RE` missed adverbials (`"Pain now 3/10"`); `detect_clinical_contradictions` collided cross-section Aspirin home maintenance (`81 mg`) with ED loading order (`300 mg`). |
| `vision/architectures.py`, `cnn_torch.py` | Implemented | Pure PyTorch from-scratch `SE-ResNetV2` (`1,228,970` params) & `ConvNeXt-Derm` (`2,492,954` params). Served calibration in `models/derm_ensemble_calibration.json`. |
| `interfaces/routers/`, `frontend/` | Implemented | Hardcoded `subj_synth_001` API/UI defaults already removed; React production build + 13 Vitest tests pass. |

---

## 2. Re-Measured Baseline (`reports/baseline/v2_remeasured_baseline.json`)

| Dataset / Benchmark | Split / Subset | `n` | Strict Span F1 | Lenient Span F1 | Assertion Acc / Macro-F1 | Relations F1 | ECE / Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| **`SYNTH-MED-V3.1`** (`hybrid_v2`) | `val.jsonl` (ALL) | `160` | `0.9932` (`P 0.9929 / R 0.9936`) | `0.9953` | `0.9471` / `0.8036` | `0.9531` | `0.0060` / `0.0059` |
| **`SYNTH-MED-V3.1`** (`hybrid_v2`) | `val.jsonl` (EN) | `80` | `0.9931` | `0.9951` | `0.9557` / `0.8520` | `0.9630` | `0.0063` / `0.0060` |
| **`SYNTH-MED-V3.1`** (`hybrid_v2`) | `val.jsonl` (FR) | `80` | `0.9933` | `0.9954` | `0.9378` / `0.7666` | `0.9420` | `0.0061` / `0.0059` |
| **`open_clinical/test.jsonl`** (`hybrid_v2`, out-of-domain) | Full (**contaminated: 73/443 overlap**) | `443` | `0.1276` | `0.2950` | — | — | — |
| **`open_clinical/test.jsonl`** (`hybrid_v2`, out-of-domain) | **De-overlapped remainder** | `370` | `0.1297` | `0.2953` | — | — | — |
| **NOISE-0027** (provisional, `draft_unreviewed`) | Before v2 fixes | `15` checks | — | — | `13 / 15` (`86.67%`) | — | Open: `n27_07`, `n27_08` |
| **NOISE-0038** (provisional, `draft_unreviewed`) | Before v2 fixes | `15` checks | — | — | `13 / 15` (`86.67%`) | — | Open: `n38_13`, `n38_15` |

---

## 3. Datasets, Provenance, Licenses & P0 Leakage Audit

| Dataset | Role | `n` | Provenance / License | Leakage Audit Result |
|---|---|---:|---|---|
| `data/locked/synth_med_v3_1/*.jsonl` | Synthetic EN/FR val & locked tests | `640` (`160×4`) | Internal synthetic generator (`SYNTH-MED-V3.1`, MIT repo) | `0` train/val/test overlap; SHA-256 locked (`LOCK_MANIFEST.json`). |
| `data/processed/open_clinical/*.jsonl` | Open clinical NER (`E3C`, `MACCROBAT`, `PxCorpus`) | `1,645` train / `258` val / `443` test | Public research corpora (CC-BY / research) | **Contamination confirmed:** `73 / 443` exact surface-text overlaps with `train.jsonl` (all `PxCorpus`). Reported separately from de-overlapped `n=370`. |
| `tests/fixtures/noise_0027.txt`, `noise_0038.txt` | Internal regression (noisy synthetic) | `2` | Synthetic clinical stress cases | Refs (`noise_0027_ref_draft.json`, `noise_0038_ref_draft.json`) untouched and remain `"status": "draft_unreviewed"` (C7). |
| `configs/vision/locked_test_v4.txt` | Frozen vision benchmark | `1,611` (`690` derm / `571` clin / `350` tbp) | ISIC / HAM10000 + Fitzpatrick17k (research) | `0` exact SHA-256 duplicates vs `train_v3` (`13` removed in `v4_dedup_audit.json`). |
| External smartphone cohorts (`PAD-UFES-20`, `DDI`) | External clinical validation | `0` local | Institutional access/license required | **Blocked locally** — not present on disk; `scripts/ingest_external_clinical_derm.py` ready with license tracking. |

---

## 4. Changes Implemented & Technical Justification (P1–P4)

1. **P1 Generalizable CRF Vital/OCR Fragment Coalescing ([extractor.py](file:///Users/guest777/Desktop/medical/medresearch_platform/serving/extractor.py)):**
   - **Root cause:** `_TOKEN_RE` splits `142/86` -> `142` and `15B/9?` -> `15` + `B/9`, and CRF mistypes `118/74 mmHg` as `DATE` or `B` in `12?/7B` as `PATIENT_ID`.
   - **Fix:** In `_extract_structured_vitals_and_labs`, whenever a verified vital/lab value+unit span `[vs, val_unit_end]` matches immediately after a known analyte, all strictly contained partial/mistyped CRF fragments are coalesced into a single `LAB_VALUE` (`+ LAB_UNIT`). Any OCR-corrupted token (`?`, `[0-9][OBA]`) is marked `certainty="possible"`, `ocr_corrupted=True`, `review_status="needs_review"` — **never confirmed** and **never auto-repaired when `?` is present**.
2. **P1 Pain/Score Adverbial Guard ([extractor.py](file:///Users/guest777/Desktop/medical/medresearch_platform/serving/extractor.py)):**
   - **Root cause:** `_PAIN_PREFIX_RE` failed when an adverbial appeared between `Pain` and `3/10` (e.g., `"Pain now 3/10"`, `"douleur évaluée à 4/10"`), letting CRF tag the score as `DATE`.
   - **Fix:** Extended `_PAIN_PREFIX_RE` to accept EN/FR temporal/rating adverbials (`now`, `currently`, `rated`, `improved to`, `évaluée à`, etc.).
3. **P2 Extended Same-Line Return-Precaution Lookahead ([context_negex.py](file:///Users/guest777/Desktop/medical/medresearch_platform/approaches/context_negex.py)):**
   - **Root cause:** Urgent action phrases >45 chars after a symptom (e.g., `"worsening headache, weakness, speech / vision changes—seek urgent evaluation"`) fell outside `post_window`.
   - **Fix:** Extended `_POST_SEEK_URGENT_RE` (adding EN/FR urgent action cues: `return immediately`, `go to the ED/ER`, `call 911`, `consulter en urgence`) to scan to the end of the current sentence/line (up to 120 chars).
4. **P4 Scope-Aware Contradiction Detection ([extractor.py](file:///Users/guest777/Desktop/medical/medresearch_platform/serving/extractor.py)):**
   - **Root cause:** `detect_clinical_contradictions` previously grouped all Aspirin mentions together, creating a false dose conflict between home maintenance (`81 mg` in `medications`) and ED loading order (`300 mg` in `plan`, `ordered_not_administered`).
   - **Fix:** Scoped medication dose conflicts by `(medication, clinical_scope)` (`maintenance` vs `plan_order`) and emitted explicit `medication_order_vs_administration:<med>` conflicts with `section`, `status`, and `reason` for both sides.

---

## 5. Before / After Results & Regression Breakdown (Section 6 Categories)

### A. Existing Validation (`SYNTH-MED-V3.1 val.jsonl`, `n=160` — `reports/eval/v2_after_val_en_fr_calibration.json`)
- **Zero regressions** on all metrics: Strict F1 `0.9932` (EN `0.9931`, FR `0.9933`), Assertion Acc `0.9471` / Macro-F1 `0.8036` (EN `0.9557` / `0.8520`, FR `0.9378` / `0.7666`), Relations F1 `0.9531`, ECE `0.0060`, Brier `0.0059`.
- **Abstention quality (`conf >= 0.80`):** Coverage `99.73%`, Precision `99.42%`; abstained `<0.80` (`n=8`) error rate `50.0%` (correctly concentrating uncertainty on hard spans).

### B. Provisional Regression (`NOISE-0027` & `NOISE-0038` — `status: "draft_unreviewed"`, NOT human-reviewed)
| Document | Before v2 | After v2 | Open Items Fixed | Reference Review Proposals |
|---|---:|---:|---|---|
| **NOISE-0027** | `13 / 15` (`86.67%`) | **`15 / 15` (`100.0%`)** | `n27_07`, `n27_08` | `n27_07` context ordering logged in `reports/eval/reference_review_requests.md` (ref JSON untouched per C7). |
| **NOISE-0038** | `13 / 15` (`86.67%`) | **`15 / 15` (`100.0%`)** | `n38_13`, `n38_15` | None required; awaits human clinical sign-off. |

### C. Individual Classification of All 7 NOISE-0038 Contradictions (P4)
| # | Contradiction Topic | Side A vs Side B (Spans) | Classification |
|---:|---|---|---|
| 1 | `allergy_status` | `NKDA` (`[2602, 2606]`) vs `Allergy: penicillin` (`[2547, 2566]`) | **Legitimate source-document conflict** |
| 2 | `polarity:hypertension` | `HTN` (`present`, problem list) vs `"no confirmed hypertension"` (`absent`, discharge summary) | **Legitimate source-document conflict** |
| 3 | `polarity:type 2 diabetes` | `Type 2 diabetes` (`present`, old record) vs `patient denies diabetes` (`absent`) | **Legitimate source-document conflict** |
| 4 | `polarity:tenderness` | `tenderness` (`present`, chest wall exam) vs `"no reproducible tenderness"` (`absent`) | **Legitimate source-document conflict** |
| 5 | `medication_dose:amlodipine` | `10 mg` (med list) vs `5 mg` (patient-reported) | **Legitimate source-document conflict** |
| 6 | *Previously:* `medication_dose:aspirin` → **Now:** `medication_order_vs_administration:aspirin` | *Previously:* `81 mg` (home) vs `300 mg` (ED loading) → **Now:** `Aspirin` (`ordered`) vs `not administered` (MAR) | **Previously model error (false cross-section dose collision) — FIXED into legitimate order-vs-administration conflict** |
| 7 | `imaging_unperformed_vs_negative` | `No pulmonary embolism` (`[5082, 5103]`) vs `study not performed` (`[5026, 5217]`) | **Legitimate source-document conflict** |

### D. New Synthetic Held-Out Regression (`tests/test_p1_p4_heldout_generalization_variants.py` — C5 Anti-Overfitting)
- **`n27_07` category (5 held-out EN/FR distant return-precaution variants):** **`5 / 5` PASS**
- **`n27_08` category (5 held-out EN/FR BP ratio variants):** **`5 / 5` PASS**
- **`n38_13` category (5 held-out OCR-corrupted ratio/value variants, never confirmed):** **`5 / 5` PASS**
- **`n38_15` category (5 held-out EN/FR pain-score variants, never `DATE`):** **`5 / 5` PASS**
- **Mandatory Section 6 cases (4 distinct pneumonia interpretations, relative dates `"three days ago"`/`"last Friday"`, CTPA/Aspirin/HTN contradictions):** **`100%` PASS**

### E. Human-Reviewed Benchmarks
- **None exist yet** for NOISE-0027/0038 (`reviewer_signoff: null`).

---

## 6. P5 Vision Smartphone-vs-Dermoscopic Gap Investigation

### Quantitative Image-Domain Hypothesis Tests (`reports/eval/vision_smartphone_gap_hypotheses.json`)
| Hypothesis / Metric (`locked_test_v4`, `n=1,611`) | Dermoscopic (`n=690`) | Clinical Smartphone (`n=571`) | TBP Tile (`n=350`) |
|---|---:|---:|---:|
| Median Resolution (`W × H`) | `512 × 384` | `484 × 384` | `127 × 127` |
| Mean Luminance (`± std`) | `149.9 ± 25.4` | **`132.5 ± 28.8`** (darker) | `145.6 ± 22.0` |
| Mean Global Contrast | `28.1` | **`33.5`** (higher clutter) | `14.1` |
| Median Laplacian Variance (edges/clutter) | `121.4` | **`425.2`** (**3.5× higher clutter**) | `56.8` |
| Median Peripheral ITA° (skin-tone/shadow proxy) | `52.8°` | **`15.3°`** (**−37.5° shift**; `48.3%` `<10°`) | `50.4°` |
| Malignant Prevalence | `42.9%` (`296/690`) | **`60.4%`** (`345/571`) | `28.6%` (`100/350`) |

### Experimental Test & Subgroup Breakdown (`reports/eval/vision_p5_smartphone_experiment.json`)
Tested 80% center-crop TTA + ConvNeXt blend weight (`w_crop80`, `w_convnext`) tuned **strictly on `val_v3` clinical (`n=352`)**, then evaluated on frozen **`locked_test_v4` clinical (`n=571`)** with 500-resample bootstrap 95% CIs:
- `val_v3` selected: `w_crop80 = 0.50`, `w_convnext = 0.0`.
- **Overall `locked_test_v4` clinical (`n=571`, 2-seed clinical specialist `seed6+seed7`):**
  - Baseline full-frame: `AUROC 0.8783` (`[0.8484, 0.9050]`), `Sens@90%Spec 0.5449` (`[0.4256, 0.6887]`), `ECE 0.0271`, `Brier 0.1298`.
  - + 50/50 80%-center-crop TTA: `AUROC 0.8807` (`[0.8504, 0.9068]`), `Sens@90%Spec 0.5739` (`+2.90 pp`, `[0.4287, 0.6878]`), `ECE 0.0409`, `Brier 0.1282`.
  - **Failed / No-Gain Verdict (Section 7):** Delta (`+2.90 pp` Sens@90%Spec, `+0.0024` AUROC) is **inside the 95% bootstrap CI** — therefore **NOT** a statistically significant improvement. Served calibration (`models/derm_ensemble_calibration.json`) was **not** overwritten.
- **Root-Cause Subgroup Breakdown on `locked_test_v4` clinical (`n=571`):**
  - `source_isic_clinical` (`n=203`): `AUROC 0.9904` (`[0.9746, 0.9994]`), **`Sens@90%Spec 0.9888`** (`[0.9628, 1.0000]`).
  - `source_fitzpatrick17k` (`n=368`): `AUROC 0.7241` (`[0.6666, 0.7806]`), **`Sens@90%Spec 0.2695`** (`[0.2122, 0.4108]`).
  - `ita_light_gt41` (`n=226`): `AUROC 0.9047`, `Sens@90%Spec 0.6906` (`[0.5759, 0.8126]`).
  - `ita_dark_lt10` (`n=276`): `AUROC 0.8639`, `Sens@90%Spec 0.4658` (`[0.2896, 0.6288]`).
  - `ita_med_10_41` (`n=69`, only `24` benign): `AUROC 0.8417`, `Sens@90%Spec 0.3556` (`[0.1237, 0.8864]` — wide CI, limited benign support).
- **Conclusion:** The smartphone gap is **71.9 pp source-concentrated** in `Fitzpatrick17k` (`26.95%` vs ISIC clinical `98.88%`) due to web-scraped label noise, 114-disease taxonomy collapse, dark-skin/shadow shift, and background clutter. Closing it requires biopsy-confirmed external clinical smartphone training data (`PAD-UFES-20`, `DDI`), which are blocked locally by institutional access.

---

## 7. CPU Performance, Memory & Verification Summary

- **NLP (`hybrid_v2` on `val.jsonl`, `n=160`):** Load `33.0 ms` (`<= 1.5×` baseline), `53.2 ms/doc` (`18.8 docs/s`), Peak RSS `161.7 MB` (`<= 2×` baseline, `0.5%` of 32 GB RAM).
- **Vision CPU (`i7-9750H`, 6 threads):** Single forward+backward (`64×64`, batch 2) `<1 s`; 4-model inference over `923` clinical images (`val_v3` + `locked_test_v4`, full + crop) completed in `194.8 s` (`~0.05 s/image/model`).
- **Verification Gates:**
  - `ruff check`: **PASS** (`All checks passed!`)
  - `mypy --ignore-missing-imports`: **PASS — `0` issues across `30` source files** (`approaches/`, `serving/`, `interfaces/`, `hybrid/`, expanded from 4 files)
  - `pytest` (7 suites incl. 20 held-out variants, 18 clinical fixtures, 8 OCR variants, NOISE-0027/0038, API/CLI/E2E): **`68 passed`, `0` failed** (`312.36s`), plus `test_vision_step3.py` passed earlier.
  - Frontend `tsc --noEmit && vite build`: **PASS** (`6.28s`); Vitest (`src/`): **`13 passed`**.

---

## 8. Remaining Limitations, Pending Human Review & Modified Files

- **Pending Human Review:** Sign-off on `tests/fixtures/noise_0027_ref_draft.json`, `noise_0038_ref_draft.json`, and `reports/eval/reference_review_requests.md`.
- **Blocked External Data:** `PAD-UFES-20` and `DDI` clinical smartphone cohorts require institutional access/licensing.
- **Unresolved Contamination Risk:** `open_clinical/test.jsonl` retains `73/443` `PxCorpus` surface-text overlaps with `train.jsonl` (explicitly separated; never cite full `443` as clean).
- **Modified Code Files:** `medresearch_platform/approaches/context_negex.py`, `medresearch_platform/serving/extractor.py`.
- **Added Test & Report Files:** `tests/test_p1_p4_heldout_generalization_variants.py`, `reports/baseline/v2_remeasured_baseline.json`, `reports/eval/v2_after_val_en_fr_calibration.json`, `reports/eval/vision_smartphone_gap_hypotheses.json`, `reports/eval/vision_p5_smartphone_experiment.json`, `reports/eval/reference_review_requests.md`, `reports/eval/existing_models_optimization_report.md`.

### Reproducibility
```bash
.venv-vision/bin/ruff check medresearch_platform/approaches/context_negex.py medresearch_platform/serving/extractor.py tests/test_p1_*.py
.venv-vision/bin/mypy --ignore-missing-imports medresearch_platform/approaches/ medresearch_platform/serving/ medresearch_platform/interfaces/ medresearch_platform/hybrid/
.venv-vision/bin/pytest -q tests/test_p1_p4_heldout_generalization_variants.py tests/test_p1_p3_clinical_fixtures_and_ocr_noise.py tests/test_phase4_nlp_assertions_relations_temporal.py tests/test_noise_0027_regression_benchmark.py tests/test_noise_0038_and_cross_doc_benchmark.py tests/test_api_and_cli.py tests/test_phase1_p1_to_p9.py
```

---

## 9. Verdict

- **Better than before?** **Yes on NLP OCR/context/contradictions; no statistically significant change on Vision smartphone.**
- **Measured numbers:** `SYNTH-MED-V3.1 val.jsonl` (`n=160`) preserved at `0.9932` Strict F1 / `0.9471` Assertion Acc / `0.8036` Assertion Macro-F1 / `0.9531` Rel F1 / `0.0060` ECE (`0` regressions); provisional NOISE-0027 improved `13/15 -> 15/15` (`100%`) and NOISE-0038 improved `13/15 -> 15/15` (`100%`) with `7/7` legitimate contradictions (`1` false cross-section Aspirin dose collision fixed); **20/20** held-out synthetic corruption variants pass; Vision `locked_test_v4` clinical center-crop TTA (`+2.90 pp` Sens@90%Spec, `0.5449 -> 0.5739`) fell inside the 95% CI (`[0.4256, 0.6887]`) and was not promoted.
- **Errors fixed / remaining:** All 4 open items (`n27_07`, `n27_08`, `n38_13`, `n38_15`) and the false Aspirin cross-section dose contradiction are fixed and generalize to 20/20 held-out variants. Remaining: `Fitzpatrick17k` clinical smartphone gap (`26.95%` vs ISIC clinical `98.88%`) blocked on biopsy-confirmed external smartphone cohorts; human sign-off pending on NOISE-0027/0038 draft references.
- **100% from scratch (C2)?** **Verified YES** (`0/39` manifests use pretrained weights; `torchvision`/`timm`/`transformers`/`spacy` not installed; `~/.cache/torch` absent).
- **CPU / 32 GB RAM OK (C3)?** **Verified YES** (`161.7 MB` peak NLP RSS, `53.2 ms/doc`; Vision 4-model eval `194.8 s` on 923 images).
- **Leakage risks unresolved?** `open_clinical/test.jsonl` `73/443` `PxCorpus` overlap remains in the raw file (de-overlapped `n=370` reported separately: `0.1297` strict / `0.2953` lenient F1 for `hybrid_v2`); `SYNTH-MED-V3.1` and Vision `locked_test_v4` have `0` exact train/test overlap.
