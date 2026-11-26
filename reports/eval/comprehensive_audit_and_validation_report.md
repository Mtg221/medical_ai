# Comprehensive Audit, Improvement & Scientific Validation Report

**Repository:** `github.com/Mtg221/medical_ai`  
**Branch:** `feat/integrate-conversation-dataset-en-fr`  
**Dataset Archive:** `medical_conversation_dataset_en_fr.zip` (`SHA-256: c99267e95e6e44adabc42de42857a322d7bbe131154dbc04bf17400aef093ac7`)  
**Hardware:** Intel Core i7-9750H (CPU-only, macOS `x86_64`), 32 GB RAM, Python 3.12.16, PyTorch 2.2.2  
**Constraints:** 100% From-Scratch (`0` pretrained weights), Zero New Model Architectures, Baseline Checkpoints Preserved  
**Date:** 2026-11-26  

---

## A. Repository Changes

| File | Action | Technical Necessity |
|---|---|---|
| `medresearch_platform/data/conversation_loader.py` | Modified | Added strict 3-axis orthogonal span metadata (`assertion`, `temporality`, `experiencer`, `speaker_role`, `turn_id`, `section`, `ocr_corrupted`) alongside backward-compatible canonical `AnnotatedDocument` conversion; enforces zero split leakage. |
| `medresearch_platform/approaches/context_negex.py` | Modified | Added bilingual EN/FR code-switch clause detection (`language="mixed"` / `"auto"` + cross-language cue detection), clinician question-turn detection (`Are you experiencing...?` / `Avez-vous...?` -> `hypothetical`), resolved-state detection (`disappeared` / `a disparu`), planned-exam cues (`will order` / `va prescrire`), prescribed-not-taken (`never took` / `jamais pris`), and FR `"impossible d'exclure"` / `"n'a pas "` / `"ne prend pas "`. |
| `scripts/train_conversation_models.py` | Modified | Fixed `PyTorchCharCNNBiLSTMCRF` import & `save`/`predict_entities` contract for Kaggle GPU `--train-torch` execution; zero mypy/ruff errors. |
| `notebooks/kaggle_train_existing_models_en_fr.ipynb` | Added | Reproducible Kaggle GPU notebook invoking `scripts/train_conversation_models.py` (uses existing models only; saves to `/kaggle/working/models_out`). |
| `tests/test_conversation_bilingual_and_turn_context.py` | Added | Regression suite for EN/FR questions vs answers, family history, planned exams, prescribed-not-taken meds, resolved symptoms, and code-switched turns. |
| `docs/HUMAN_EVALUATION_PROTOCOL.md` | Added | Independent human clinical evaluation protocol (stratified `n>=150` EN/FR/mixed, 3-axis guidelines, double-annotation IAA plan). |
| `reports/eval/conversation_deep_leakage_audit.json` | Added | Non-pretrained char-5gram, template-skeleton, turn-reuse, and entity-vocabulary leakage audit. |
| `reports/eval/comprehensive_scientific_validation.json` | Added | Orthogonal 3-axis metrics, unseen-entity recall, paired bootstrap significance, 4-split `SYNTH-MED-V3.1` retention, and `PyTorchCharCNNBiLSTMCRF` CPU smoke verification. |

---

## B. Existing Model Improvements (Zero New Architectures)

1. **`HybridPipeline` (`models/hybrid_v2_conv_en_fr_seed1`; baseline `models/hybrid_v2_seed1` preserved):**
   - **Original weaknesses:** Out-of-domain drop on EN/FR conversations (`Strict F1 0.3475`, `ECE 0.2527`); single-language ConText failed on code-switched `mixed` turns (`Assertion Macro-F1 0.3848`); clinician questions & resolved/planned/not-taken conversational turns were misattributed.
   - **Corrections & Training:** Trained on combined `SYNTH-MED-V3.1 train (600)` + `Conversation train (960)` = `1,560` docs (`1,110.6 s` CPU); upgraded Stage 2 `context_negex.py` with bilingual code-switching and conversation-turn rules.
2. **`ClassicalCRFModel` (`models/classical_crf_conv_en_fr_seed1`; baseline `models/classical_crf_seed1` preserved):**
   - **Original weakness:** Out-of-domain Strict F1 `0.4005`, Rel F1 `0.3695`, ECE `0.2916`.
   - **Training:** Trained on identical `1,560`-doc combined split (`177.4 s` CPU); achieved `0.9970` Strict F1 / `0.8406` Rel F1 / `0.0014` ECE on synthetic `test_118` (see Section D leakage caveat).
3. **`PyTorchCharCNNBiLSTMCRF` (`models/hospital_char_cnn_bilstm_crf_conv_en_fr_cpu_smoke`):**
   - **Verification:** Verified end-to-end from-scratch CPU training (`2` epochs, `60` docs, `11.18 s`), checkpoint save, reload, and CPU inference (`52.5 ms/doc`, `pretrained_weights_used: false`). At 2 epochs on 60 docs it has not converged (`test20 Strict F1 = 0.0`); full 16-epoch training on all `1,560` docs requires Kaggle GPU via `--train-torch`.

---

## C. Before-vs-After Results & Statistical Significance

### 1. Conversation Held-Out `test` (`n=118`, `4,716` spans)
- **`hybrid_v2` Entity Strict F1:** `0.3475 →` **`0.9707`** (`Δ = +0.6232`, 1,000-resample paired bootstrap **`p = 0.0020`**, McNemar `n10 = 0`, `n01 = 2,160`, **`p < 1e-6`**).
- **`hybrid_v2` Relation F1:** `0.4696 →` **`0.6627`** (`+19.31 pp`).
- **`hybrid_v2` Calibration:** ECE `0.2527 →` **`0.0653`** (`−74.2%`), Brier `0.1789 →` **`0.0411`** (`−77.0%`).
- **Unseen-Surface Test Entities (`240` mentions whose `(type, surface)` never appeared in `train`):**
  - `hybrid_v2` baseline: `117 / 240` (`48.75%`) → `hybrid_v2_conv_en_fr_seed1`: **`239 / 240` (`99.58%`)**; `classical_crf_conv_en_fr_seed1`: **`238 / 240` (`99.17%`)**.

### 2. Orthogonal 3-Axis Clinical Context (`test_118`, `4,716` spans — Separate Dimensions)

| Subset | `n` docs (`spans`) | Assertion (3-cls) Acc / Macro-F1 / Weighted-F1 | Temporality (2-cls) Acc / Macro-F1 | Experiencer (2-cls) Acc / Macro-F1 |
|---|---:|---:|---:|---:|
| **ALL** | `118` (`4,716`) | `0.9262` / `0.6479` / `0.9228` | `0.9056` / `0.6701` | **`0.9994` / `0.9934`** |
| **English (`en`)** | `54` (`2,149`) | `0.9195` / `0.6360` / `0.9163` | `0.9096` / `0.7080` | **`0.9986` / `0.9854`** |
| **French (`fr`)** | `54` (`2,158`) | `0.9374` / `0.6581` / `0.9349` | `0.9051` / `0.5932` | **`1.0000` / `1.0000`** |
| **Code-Switched (`mixed`)** | `10` (`409`) | **`0.9022` / `0.6555`** *(was `0.7838 / 0.3848`)* | **`0.8875` / `0.7380`** | **`1.0000` / `1.0000`** |

### 3. Catastrophic Forgetting / Retention Audit Across All 4 `SYNTH-MED-V3.1` Splits (`n = 640`)

| Frozen `SYNTH-MED-V3.1` Split | `n` | Baseline Strict F1 | Updated Strict F1 (`Δ`) | Baseline → Updated Assert Acc | Baseline → Updated Rel F1 |
|---|---:|---:|---:|---:|---:|
| `val` | `160` | `0.9932` | `0.9842` (`−0.0090`) | `0.9464 → 0.9459` | `0.9531 → 0.9283` |
| `test_iid` | `160` | `0.9920` | `0.9855` (`−0.0065`) | `0.9348 → 0.9361` | `0.9485 → 0.9219` |
| `test_unseen_templates` | `160` | `0.9461` | `0.9222` (`−0.0239`) | `0.9457 → 0.9433` | `0.9106 → 0.8845` |
| `test_unseen_terms` | `160` | `0.9660` | `0.9623` (`−0.0037`) | `0.9396 → 0.9420` | `0.8916 → 0.8998` (`+0.0082`) |

*Honest note:* Mild forgetting (`−2.39 pp` Strict F1) occurs on `test_unseen_templates` when mixing 960 dialogues with 600 structured notes; baseline `models/hybrid_v2_seed1` remains preserved intact.

---

## D. Deep Leakage Audit (`reports/eval/conversation_deep_leakage_audit.json`)

- **Document-level leakage:** **`0` exact duplicates**, **`0` normalized duplicates**, **`0` encounter/patient overlaps**, **`0` char-5gram cosine ≥ 0.70** (mean `0.5137`, max `0.6322`).
- **Template & Lexicon Reuse (Root Cause of `0.97–0.997` F1):**
  - **`82.14%` (`1,637 / 1,993`)** of test conversation turns share an **exact masked turn-template skeleton** with `train`.
  - **`94.91%` (`4,476 / 4,716`)** of test `(entity_type, surface)` pairs were seen verbatim in `train` (`240` unseen).
  - **`79.53%`** of test numerical tokens were seen in `train`.
- **Validity Verdict:** Original `test_118` metrics are **valid as an in-distribution synthetic conversation benchmark** (zero encounter leakage), **NOT** as proof of open-world hospital dialogue generalization due to high synthetic turn-template reuse (`82.14%`).

---

## E–G. Bilingual, Noise Robustness & CPU Inference Summary

- **Bilingual (`hybrid_v2` `test_118`):** EN Strict F1 `0.9646`, FR `0.9766`, Mixed `0.9714`; code-switched assertion Macro-F1 jumped **+27.1 pp** (`0.3848 → 0.6555`).
- **Noise Robustness (`hybrid_v2` `test_118`):** Clean `0.9741`, Moderate `0.9692`, High OCR/ASR Noise `0.9643` (**+68.1 pp** over baseline `0.2834`).
- **CPU Inference (Intel i7-9750H, 32 GB RAM):**
  - `hybrid_v2_conv_en_fr_seed1`: `50.5 ms` load, `215.8 ms/doc` on ~419-token dialogues (`28–99 ms` on single notes), `178.1 MB` peak inference RSS (`0.55%` of 32 GB RAM).
  - `classical_crf_conv_en_fr_seed1`: `16.8 ms` load, `38.8 ms/doc`.
  - `PyTorchCharCNNBiLSTMCRF`: `52.5 ms/doc` on CPU.

---

## H. Kaggle Reproducibility & I. Limitations

- **Kaggle GPU Execution Status:** Local environment has **no GPU**; full 16-epoch CUDA training of `PyTorchCharCNNBiLSTMCRF` was **not** executed locally and is ready to run on Kaggle via `notebooks/kaggle_train_existing_models_en_fr.ipynb` (`--train-torch`). CPU training + reload verification of `PyTorchCharCNNBiLSTMCRF` completed (`11.18 s`).
- **Limitations:**
  1. Single random seed (`seed=1`) trained for full `HybridPipeline`/`ClassicalCRFModel` due to CPU budget (`~21 min/seed`); multi-seed training (`seed=1..3`) is supported via `--seed` on Kaggle.
  2. `82.14%` synthetic turn-template reuse inflates absolute F1 on `test_118`.
  3. No human-annotated real-world conversation gold set exists yet (`docs/HUMAN_EVALUATION_PROTOCOL.md` defines the protocol).
  4. `NOISE-0027` and `NOISE-0038` draft references remain `"status": "draft_unreviewed"` pending human sign-off.

---

## J. Final Scientific Verdict

| Dimension | Classification | Evidence |
|---|---|---|
| **Engineering implementation** | **Verified & Production-Grade** | `267/267` targeted tests pass, `ruff` & `mypy` clean, 100% from-scratch (`0` pretrained), backward-compatible 3-axis taxonomy. |
| **Synthetic-data evaluation** | **Strong & Statistically Significant** | `hybrid_v2` Strict F1 `0.3475 → 0.9707` (`p = 0.0020`), ECE `0.2527 → 0.0653`, code-switched Assertion Macro-F1 `0.3848 → 0.6555`, Experiencer Macro-F1 `0.9934`. |
| **Independent human evaluation** | **Not Yet Executed (Protocol Ready)** | `docs/HUMAN_EVALUATION_PROTOCOL.md` authored; zero fabricated human metrics. |
| **Real-world clinical generalization** | **Unproven (Synthetic-Only)** | `82.14%` synthetic turn-template overlap and `−2.39 pp` mild forgetting on unseen structured templates require external human-annotated clinical evaluation before clinical deployment. |
| **CPU deployment readiness** | **Ready (Intel i7-9750H, 32 GB RAM)** | `16.8–50.5 ms` load, `38.8–215.8 ms/doc`, `178.1 MB` peak RSS, zero GPU dependency. |
