> **SUPERSEDED (2026-11-14):** Results in this v2 report are **invalid** because of a data-generator bug (doubled TIMEX/date/duration text in 100% of documents) and severe train/test template & term overlap (`94.8%` masked sentence overlap, `100%` lexical overlap on clinical categories). Archived raw JSON and audit output live in [docs/results/archive/2026-11-14_synth_v2_buggy/](results/archive/2026-11-14_synth_v2_buggy/README.md). See Phase 11 for the corrected evaluation.

# Research Report: Bilingual (EN/FR) Medical Information Extraction Platform

**Date:** 2026-11-14  
**Platform:** CPU-only local execution — Intel(R) Core(TM) i7-9750H @ 2.60GHz (6 cores / 12 threads), 32 GB RAM, macOS Darwin `x86_64`, Python 3.13.12  
**Primary Track Constraint:** 100% trained from scratch (random initialization only; no pretrained language models, no pretrained word embeddings, no pretrained tokenizers). Deep learning implemented in pure NumPy per amended [ADR-0005](../DECISIONS/ADR-0005-pytorch-cpu-from-scratch.md).

---

## 1. Dataset & Split Integrity

- **Corpus:** `SYNTH-MED-V2` (`configs/data/v1.yaml`) — 800 synthetic bilingual clinical notes (400 English + 400 French) generated with deterministic patient-stratified templates, negation/uncertainty/family-history/historical assertions, typed clinical relations, and `TIMEX` expressions.
- **Splits (patient-stratified 70/15/15):**
  - `train`: 555 documents
  - `val`: 124 documents
  - `test` (locked): 121 documents
- **Leakage Verification:** MinHash + patient-ID overlap check (`medextract data build`) confirmed `0` subject overlaps and `0` near-duplicate document leaks across `train`/`val`/`test`.
- **Note on Real Clinical Corpora:** No real patient data is included in this repository. External corpora (`n2c2`, `DEFT/CAS`) require institutional DUAs prior to local evaluation.

---

## 2. Locked Test-Split Comparison (121 Docs, Seeds 1, 2, 3)

All 6 extraction families (+ `classical_crf`) were trained on `train` (555 docs) and evaluated **once** on the locked `test` split (121 docs) using `scripts/run_locked_test_eval.py` (results archived in `docs/results/locked_test_evaluation.json`).

| Approach | Strict F1 (mean ± std) | Lenient F1 | EN F1 | FR F1 | Assert Acc | Rel F1 | TIMEX F1 | ECE | Wall (s) | Peak RAM (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| `rule_based` | `0.8583 ± 0.0000` | `0.8583` | `0.8673` | `0.8491` | `0.9936` | `0.8233` | `0.0000` | `0.0300` | `0.22` | `169.4` |
| `classical_ml` | `0.8800 ± 0.0000` | `1.0000` | `0.8800` | `0.8800` | `0.9944` | `0.8696` | `0.0000` | `0.1441` | `88.31` | `397.3` |
| `classical_crf` | `0.8800 ± 0.0000` | `1.0000` | `0.8800` | `0.8800` | `0.9944` | `0.8696` | `0.0000` | `0.1188` | `10.82` | `436.7` |
| `bilstm` | `0.8798 ± 0.0003` | `0.9998` | `0.8800` | `0.8800` | `0.9944` | `0.8696` | `0.0000` | `0.1198` | `29.90` | `452.4` |
| `bilstm_crf` | `0.8799 ± 0.0002` | `0.9998` | `0.8800` | `0.8800` | `0.9944` | `0.8696` | `0.0000` | `0.0656` | `51.77` | `463.9` |
| `transformer` | `0.7634 ± 0.0180` | `0.9171` | `0.8131` | `0.7473` | `0.9945` | `0.7537` | `0.1823` | `0.2058` | `5.30` | `471.5` |
| **`hybrid` (5-stage)** | **`0.9362 ± 0.0000`** | `0.9362` | **`0.9362`** | **`0.9362`** | `0.8103` | `0.8329` | **`0.3071`** | **`0.0139`** | `71.28` | `494.4` |

### Paired Significance Test (`hybrid` vs `bilstm_crf`, Seed 1)
- `bilstm_crf` Strict F1: `0.8800`
- `hybrid` Strict F1: `0.93617`
- **ΔF1:** `+0.05617`
- **Paired bootstrap p-value:** `0.003992` (< 0.01, statistically significant improvement in strict span F1).

---

## 3. Hybrid Pipeline Ablation Study (`val`, 124 Docs)

| Variant | NER F1 | EN F1 | FR F1 | Assert Acc | Rel F1 | TIMEX F1 | DocCls Acc | ECE |
|---|---|---|---|---|---|---|---|---|
| `full_hybrid` | `0.9362` | `0.9362` | `0.9362` | `0.8699` | `0.8477` | `0.3123` | `1.0000` | `0.0258` |
| `no_rules_layer` | `0.9362` | `0.9362` | `0.9362` | `0.8699` | `0.8477` | `0.3123` | `1.0000` | `0.0787` |
| `no_assertion_bilstm` | `0.9362` | `0.9362` | `0.9362` | `0.9912` | `0.8477` | `0.3123` | `1.0000` | `0.0258` |
| `no_relation_cnn` | `0.9362` | `0.9362` | `0.9362` | `0.8699` | `0.9091` | `0.3123` | `1.0000` | `0.0258` |
| `no_temporal_crf` | `0.9362` | `0.9362` | `0.9362` | `0.8699` | `0.8477` | `0.0000` | `1.0000` | `0.0258` |
| `no_transformer_fusion` | `0.8800` | `0.8800` | `0.8800` | `0.8699` | `0.8057` | `0.3123` | `1.0000` | `0.1092` |

Key ablation findings:
1. **Stage 5 Transformer Span Fusion** is the largest driver of Strict F1 (`0.8800` -> `0.9362`) and calibration (`ECE 0.1092` -> `0.0258`).
2. **Stage 2 Rules Layer** provides strong calibration support (`ECE 0.0787` -> `0.0258`).
3. **Stage 4 Temporal CRF** is essential for `TIMEX` recall (`0.0000` -> `0.3123`).

---

## 4. Document Processing & Real-World OCR Benchmark (Phase 5)

Measured with OpenCV preprocessing (upscale, deskew, median denoise, adaptive threshold) + Tesseract 5.5.4 (`eng+fra`) + `pypdfium2` (300 dpi) on skewed, noisy synthetic scans:

| Sample | Language | Mode | CER | WER | Mean Confidence | Word Boxes |
|---|---|---|---|---|---|---|
| `en_scan_pdf` (`skew=1.5°`, `noise=5.0`) | `en` | `TESSERACT_OCR` | `0.0000` | `0.0000` | `0.9528` | `18` |
| `en_scan_png` (`skew=2.0°`, `noise=8.0`) | `en` | `TESSERACT_OCR` | `0.0000` | `0.0000` | `0.9567` | `18` |
| `fr_scan_pdf` (`skew=-1.5°`, `noise=5.0`) | `fr` | `TESSERACT_OCR` | `0.0000` | `0.0000` | `0.9356` | `18` |
| `fr_scan_png` (`skew=-2.0°`, `noise=8.0`) | `fr` | `TESSERACT_OCR` | `0.0000` | `0.0000` | `0.9372` | `18` |

---

## 5. Error Analysis & Limitations

1. **Assertion Accuracy Trade-Off:** On the locked test set, `hybrid` achieves higher Strict Span F1 (`0.9362`) and best calibration (`ECE = 0.0139`), but its learned BiLSTM assertion classifier scores `0.8103` vs `0.9944` for the cue-lexicon baseline (`assertion_wrong = 372` vs `15`). Combining learned assertion confidence with high-precision cue overrides is a clear next improvement.
2. **Boundary & TIMEX Errors:** Single sequence models (`classical_ml`, `classical_crf`, `bilstm`, `bilstm_crf`) achieve `1.0000` / `0.9998` Lenient F1 but `0.8800` Strict F1 due to `363` exact-boundary mismatches (primarily multi-token `TIMEX` spans).
3. **Synthetic Corpus Ceiling:** `SYNTH-MED-V2` has structured phrasing; external validation on real de-identified clinical corpora (`n2c2`, `DEFT/CAS`) is required before claiming clinical generalization.
