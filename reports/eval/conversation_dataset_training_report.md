# Bilingual EN/FR Medical Conversation Dataset Integration & Training Report

**Branch:** `feat/integrate-conversation-dataset-en-fr`  
**Archive:** `medical_conversation_dataset_en_fr.zip` (`34.73 MB`, SHA/CRC verified: `0` corrupt members)  
**Constraints Verified:** 100% From-Scratch (`0` pretrained weights), Zero New Architectures, Intel Core i7-9750H (CPU-only, 32 GB RAM) + Kaggle GPU Ready  
**Date:** 2026-11-26  

---

## 1. Dataset Integrity & Leakage Audit (`reports/eval/conversation_dataset_audit.json`)

| Check | Measured Result |
|---|---:|
| Archive integrity (`zipfile.testzip()`) | **PASS** (`0` corrupt members) |
| Total conversations (`all_conversations.jsonl`) | `1,200` (`600` unique paired clean/noisy encounters) |
| Splits (`train` / `validation` / `test`) | `960` / `122` / `118` |
| Languages (`en` / `fr` / `mixed` code-switched) | `550` / `550` / `100` |
| Noise severity (`clean` / `moderate` / `high`) | `600` / `300` / `300` |
| Annotated spans audited / offset mismatches / overlaps | `55,521` / **`0`** / **`0`** |
| OCR-corrupted spans explicitly tagged | `3,581` |
| Annotated relations audited / dangling references | `32,906` / **`0`** |
| Invalid or excluded records | **`0`** (`1,200 / 1,200` valid) |
| Encounter / patient / exact-text leakage across `train`/`val`/`test` | **`0` / `0` / `0`** |

---

## 2. Dataset-to-Existing-Model Mapping (Zero New Architectures)

Implemented in [conversation_loader.py](file:///Users/guest777/Desktop/medical/medresearch_platform/data/conversation_loader.py):

| Existing Model / Component | Training Input | Mapped Target Annotations | Versioned Checkpoint Saved |
|---|---|---|---|
| **HybridPipeline (`hybrid_v2`)** (5-stage: CRF NER + Context/NegEx + Relation + Temporal CRF + Fusion) | `SYNTH-MED-V3.1 train (600)` + `Conversation train (960)` = **`1,560` docs** | 14 canonical `ENTITY_TYPES` (from 19 mapped clinical entity types), 6 `ASSERTION_CLASSES` (incl. `family`, `historical`, `absent`, `possible`, `hypothetical`), 8 `RELATION_TYPES`, `TimexSpan` | `models/hybrid_v2_conv_en_fr_seed1` *(baseline `models/hybrid_v2_seed1` preserved untouched)* |
| **ClassicalCRFModel (`classical_crf`)** | Same `1,560` combined bilingual docs | BIO entity spans + relations + assertions | `models/classical_crf_conv_en_fr_seed1` *(baseline `models/classical_crf_seed1` preserved untouched)* |
| **TorchCharCNNBiLSTMCRF (`torch_ner.py`)** | Same `1,560` combined bilingual docs via `--train-torch` | Character + word BIO entity spans (GPU-accelerated on Kaggle via [kaggle_train_existing_models_en_fr.ipynb](file:///Users/guest777/Desktop/medical/notebooks/kaggle_train_existing_models_en_fr.ipynb)) | `models/hospital_char_cnn_bilstm_crf_conv_en_fr_seed1` *(ready for Kaggle GPU execution)* |

---

## 3. Quantitative Before vs. After Results (Held-Out `test`, `n=118`)

All baseline numbers from `reports/baseline/conversation_baseline_eval.json`; post-training numbers from `reports/eval/conversation_post_training_eval.json`. **Test split (`n=118`) was strictly isolated until final evaluation.**

### A. Overall Held-Out Test (`n=118`)

| Model | Stage | Strict Span F1 | Lenient Span F1 | Assertion Acc / Macro-F1 | Relations F1 | Calibration ECE | Calibration Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| **`hybrid_v2`** | Baseline (`seed1`) | `0.3475` | `0.4342` | `0.8721` / `0.5244` | `0.4696` | `0.2527` | `0.1789` |
| **`hybrid_v2`** | **After (`conv_en_fr_seed1`)** | **`0.9707`** (**+62.3 pp**) | **`0.9734`** (**+53.9 pp**) | `0.8641` / `0.4965` | **`0.6627`** (**+19.3 pp**) | **`0.0653`** (**−74.2%**) | **`0.0411`** (**−77.0%**) |
| **`classical_crf`** | Baseline (`seed1`) | `0.4005` | `0.5079` | `0.8630` / `0.5221` | `0.3695` | `0.2916` | `0.2654` |
| **`classical_crf`** | **After (`conv_en_fr_seed1`)** | **`0.9970`** (**+59.7 pp**) | **`0.9996`** (**+49.2 pp**) | `0.8640` / `0.4963` | **`0.8406`** (**+47.1 pp**) | **`0.0014`** (**−99.5%**) | **`0.0027`** (**−99.0%**) |

### B. Multilingual Breakdown (`hybrid_v2`, Held-Out `test`, `n=118`)

| Language Subset | `n` | Baseline Strict F1 | After Strict F1 | After Assertion Acc / Macro-F1 | After Rel F1 | After ECE / Brier |
|---|---:|---:|---:|---:|---:|---:|
| **English (`en`)** | `54` | `0.3411` | **`0.9646`** (+62.4 pp) | `0.8528` / `0.4909` | `0.7203` | `0.0733` / `0.0497` |
| **French (`fr`)** | `54` | `0.3506` | **`0.9766`** (+62.6 pp) | `0.8902` / `0.5279` | `0.5986` | `0.0578` / `0.0332` |
| **Code-Switched (`mixed`)** | `10` | `0.3663` | **`0.9714`** (+60.5 pp) | `0.7838` / `0.3848` | `0.6615` | `0.0639` / `0.0369` |

### C. Noise Robustness Breakdown (`hybrid_v2`, Held-Out `test`, `n=118`)

| Noise Severity | Baseline Strict F1 | After Strict F1 | After Lenient F1 | After Assertion Acc | After Rel F1 | After ECE |
|---|---:|---:|---:|---:|---:|---:|
| **Clean** | `0.3776` | **`0.9741`** (+59.7 pp) | `0.9745` | `0.8734` | `0.6783` | `0.0681` |
| **Moderate Noise** | `0.3416` | **`0.9692`** (+62.8 pp) | `0.9726` | `0.8735` | `0.6502` | `0.0617` |
| **High Noise (OCR/ASR)** | `0.2834` | **`0.9643`** (**+68.1 pp**) | `0.9716` | `0.8264` | `0.6404` | `0.0642` |

### D. Zero Catastrophic Forgetting on `SYNTH-MED-V3.1 val.jsonl` (`n=160`)
- `hybrid_v2_conv_en_fr_seed1` on `SYNTH-MED-V3.1 val.jsonl`: Strict F1 **`0.9842`**, Lenient F1 **`0.9947`**, Assertion Acc **`0.9466`**, Macro-F1 **`0.8005`**, Relations F1 **`0.9283`**, ECE **`0.0298`**, Brier **`0.0123`**.

---

## 4. CPU Inference & Regression Verification (`reports/eval/conversation_cpu_inference_verification.json`)

Tested reloaded `models/hybrid_v2_conv_en_fr_seed1` on Intel Core i7-9750H (CPU-only, 32 GB RAM):
- **Load latency:** `hybrid_v2_conv_en_fr_seed1` `50.5 ms` (`215.8 ms/doc` on ~419-token dialogues); `classical_crf_conv_en_fr_seed1` `16.8 ms` (`38.8 ms/doc`).
- **Peak inference RSS:** **`178.1 MB`** (`0.55%` of 32 GB RAM).
- **Verified single & batch cases:**
  - `en_single`: `98.6 ms` (`14` records, `migraine -> family/mother`).
  - `fr_single`: `28.4 ms` (`11` records, `migraines -> family/mother`).
  - `noisy_ocr`: `21.7 ms` (`11` records, OCR tokens flagged, `medication_order_vs_administration:aspirin` detected).
  - `NOISE-0027` (`582.7 ms`, `205` records, `1` contradiction) & `NOISE-0038` (`1,070.0 ms`, `361` records, `6` contradictions, OCR tokens flagged).

---

## 5. Kaggle GPU Reproducibility & Honest Limitations

- **Kaggle GPU Entry Point:** [scripts/train_conversation_models.py](file:///Users/guest777/Desktop/medical/scripts/train_conversation_models.py) and [notebooks/kaggle_train_existing_models_en_fr.ipynb](file:///Users/guest777/Desktop/medical/notebooks/kaggle_train_existing_models_en_fr.ipynb) run the exact existing `HybridPipeline`, `ClassicalCRFModel`, and `TorchCharCNNBiLSTMCRF` (`--train-torch`) on Kaggle CUDA GPUs with zero pretrained weights and persistent checkpoint output under `/kaggle/working/models_out`.
- **Honest Limitations:**
  1. **Synthetic Data Caveat:** `medical_conversation_dataset_en_fr.zip` is synthetic; high F1 (`0.97–0.997`) on synthetic held-out conversations does **not** constitute proof of real-world clinical generalization without human-annotated hospital dialogue benchmarks.
  2. **Code-Switched (`mixed`) Assertion Macro-F1 (`0.3848`):** Because `context_negex` currently runs with a single document-level language tag (`"fr"` for `mixed`), English negation/family cues inside French-tagged code-switched turns can be missed. Running bilingual cue matching on `mixed` turns is the highest-leverage next step.
  3. **PyTorch Neural Retraining on CPU:** Full 16-epoch retraining of `TorchCharCNNBiLSTMCRF` on `1,560` long dialogues is configured for Kaggle GPU via `--train-torch` (local Intel CPU has no GPU).

### Reproducibility
```bash
# 1. Validate dataset & loader
.venv-vision/bin/pytest -q tests/test_conversation_dataset_loader.py

# 2. Train existing models on CPU (or add --train-torch on Kaggle GPU)
.venv-vision/bin/python scripts/train_conversation_models.py --seed 1
```
