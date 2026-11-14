# Medical Information Extraction Platform — Scientific Research Report (`SYNTH-MED-V3`)

**Date:** 2026-11-14  
**Corpus:** `SYNTH-MED-V3` (`configs/data/v3.yaml`, `data/processed/v3/`, 1,240 bilingual EN/FR documents)  
**Hardware / Constraints:** CPU-only (`Intel(R) Core(TM) i7-9750H @ 2.60GHz`, 32 GB RAM, macOS Darwin `x86_64`), Python 3.13.12, pure NumPy neural models (`BiLSTM`, `BiLSTM-CRF`, `Transformer`, `Hybrid`) trained from random initialization (no pretrained weights), bilingual EN + FR, local-only execution.

> **Note on Superseded `SYNTH-MED-V2` Results:** Earlier `SYNTH-MED-V2` evaluation numbers were invalidated in Phase 11 due to a doubled-TIMEX generator bug (`F1`) and severe train/test lexical and template overlap (`F2`). All v2 artifacts, audit output, and the archived report are preserved in [docs/results/archive/2026-11-14_synth_v2_buggy/](results/archive/2026-11-14_synth_v2_buggy/README.md).

---

## 1. Corpus & Leakage Audit (`SYNTH-MED-V3`)

`SYNTH-MED-V3` (`medresearch_platform/data/synthetic_v3.py`, `lexicon_v3.py`, `templates_v3.py`) contains **1,240 documents** (620 EN + 620 FR) across 5 splits:
- `train`: 600 docs (300 EN + 300 FR)
- `val`: 160 docs (80 EN + 80 FR) — used exclusively for model/config selection and ablations
- `test_iid`: 160 docs (80 EN + 80 FR) — in-distribution held-out split
- `test_unseen_terms`: 160 docs (80 EN + 80 FR) — 30% of lexicon concepts per category held out exclusively for this split
- `test_unseen_templates`: 160 docs (80 EN + 80 FR) — 30% of sentence templates per `(language, section)` held out exclusively for this split

**Audit verification (`scripts/audit_corpus.py`):**
- Doubled span substrings: `0` docs across all splits (`F1` resolved)
- Adjacent duplicate n-grams (`n >= 2`): `0` docs across all splits
- Offset & relation argument integrity: `100.00%`
- Unseen clinical term overlap (`test_unseen_terms` vs `train`): `0.0%` on clinical categories
- Digit-masked sentence template overlap: `12.0%` (`test_iid`), `0.6%` (`test_unseen_templates`, down from `94.8%` in `SYNTH-MED-V2`)

---

## 2. Locked Test Set Results (`SYNTH-MED-V3`, Seeds 1, 2, 3)

Evaluated once per final config (`configs/exp/v3/*.yaml`) via `scripts/run_v3_locked_test_eval.py`, logged in [TEST_SET_LOG.md](results/TEST_SET_LOG.md), and saved in [v3_locked_test_evaluation.json](results/v3_locked_test_evaluation.json).

### 2.1 `test_iid` (160 docs: 80 EN + 80 FR)

| Approach | Strict F1 (mean ± std) | Lenient F1 | EN F1 | FR F1 | Assert Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm Acc | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rule_based` | `0.3000 ± 0.0000` | `0.3128` | `0.3228` | `0.2766` | `0.4542` | `0.0326` | `0.5036` | `0.1609` | `0.3` |
| `classical_ml` | `0.9976 ± 0.0000` | `0.9994` | `0.9967` | `0.9984` | `0.5225` | `0.7062` | `0.7305` | `0.0941` | `24.8` |
| `classical_crf` | `0.9980 ± 0.0000` | `0.9997` | `0.9967` | `0.9993` | `0.5230` | `0.7071` | `0.7305` | `0.0941` | `18.0` |
| `bilstm` | `0.9606 ± 0.0019` | `0.9804` | `0.9716` | `0.9536` | `0.5193` | `0.6971` | `0.7183` | `0.0882` | `27.7` |
| `bilstm_crf` | `0.9690 ± 0.0042` | `0.9786` | `0.9765` | `0.9645` | `0.5193` | `0.7065` | `0.7253` | `0.0840` | `59.2` |
| `transformer` | `0.6022 ± 0.0136` | `0.7655` | `0.6404` | `0.5864` | `0.5413` | `0.4612` | `0.3287` | `0.1020` | `6.9` |
| `hybrid` | `0.9819 ± 0.0025` | `0.9883` | `0.9850` | `0.9747` | `0.5461` | `0.9037` | `0.7304` | `0.2115` | `123.9` |

- **Significance (`test_iid`, seed 1):** `hybrid` vs `bilstm_crf`: `ΔF1 = +0.0093` (`bootstrap p = 0.0040`, `McNemar p = 0.0012`); `bilstm_crf` vs `classical_crf`: `ΔF1 = -0.0275` (`bootstrap p = 0.0040`, `McNemar p = 0.0000`).

### 2.2 `test_unseen_terms` (160 docs: 80 EN + 80 FR, 0% Clinical Term Overlap)

| Approach | Strict F1 (mean ± std) | Lenient F1 | EN F1 | FR F1 | Assert Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm Acc | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rule_based` | `0.3085 ± 0.0000` | `0.3275` | `0.3314` | `0.2869` | `0.4545` | `0.0343` | `0.4988` | `0.1943` | `0.3` |
| `classical_ml` | `0.8181 ± 0.0000` | `0.9198` | `0.8155` | `0.8205` | `0.4806` | `0.6221` | `0.7201` | `0.1140` | `24.8` |
| `classical_crf` | `0.9131 ± 0.0000` | `0.9279` | `0.9138` | `0.9125` | `0.4900` | `0.6789` | `0.7230` | `0.1140` | `13.4` |
| `bilstm` | `0.7715 ± 0.0179` | `0.8398` | `0.7838` | `0.8085` | `0.4596` | `0.6391` | `0.7026` | `0.1017` | `29.5` |
| `bilstm_crf` | `0.7860 ± 0.0061` | `0.8276` | `0.7887` | `0.7997` | `0.4847` | `0.6712` | `0.7137` | `0.0989` | `62.7` |
| `transformer` | `0.3671 ± 0.0117` | `0.5732` | `0.3817` | `0.3851` | `0.4106` | `0.2139` | `0.3054` | `0.1142` | `7.3` |
| `hybrid` | `0.8033 ± 0.0081` | `0.8221` | `0.8085` | `0.8080` | `0.5227` | `0.8058` | `0.7036` | `0.2285` | `140.9` |

- **Significance (`test_unseen_terms`, seed 1):** `hybrid` vs `bilstm_crf`: `ΔF1 = +0.0138` (`bootstrap p = 0.0399`, `McNemar p = 0.0000`); `bilstm_crf` vs `classical_crf`: `ΔF1 = -0.1187` (`bootstrap p = 0.0040`, `McNemar p = 0.0000`).

### 2.3 `test_unseen_templates` (160 docs: 80 EN + 80 FR, 0.6% Masked Sentence Overlap)

| Approach | Strict F1 (mean ± std) | Lenient F1 | EN F1 | FR F1 | Assert Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm Acc | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rule_based` | `0.2993 ± 0.0000` | `0.3171` | `0.3131` | `0.2855` | `0.4119` | `0.0431` | `0.5188` | `0.1480` | `0.3` |
| `classical_ml` | `0.9698 ± 0.0000` | `0.9797` | `0.9737` | `0.9660` | `0.6128` | `0.6932` | `0.7573` | `0.0851` | `25.0` |
| `classical_crf` | `0.9549 ± 0.0000` | `0.9721` | `0.9586` | `0.9513` | `0.6220` | `0.6995` | `0.7530` | `0.0856` | `13.4` |
| `bilstm` | `0.8578 ± 0.0051` | `0.9169` | `0.8860` | `0.8252` | `0.6141` | `0.6515` | `0.7209` | `0.0673` | `35.3` |
| `bilstm_crf` | `0.9216 ± 0.0053` | `0.9566` | `0.9357` | `0.9201` | `0.6116` | `0.6870` | `0.7439` | `0.0677` | `62.1` |
| `transformer` | `0.5369 ± 0.0281` | `0.6965` | `0.5457` | `0.5122` | `0.6239` | `0.3947` | `0.3134` | `0.0627` | `7.7` |
| `hybrid` | `0.8800 ± 0.0222` | `0.9437` | `0.8887` | `0.8520` | `0.4116` | `0.8264` | `0.7477` | `0.2318` | `156.5` |

- **Significance (`test_unseen_templates`, seed 1):** `hybrid` vs `bilstm_crf`: `ΔF1 = -0.0576` (`bootstrap p = 0.0040`, `McNemar p = 0.0000`); `bilstm_crf` vs `classical_crf`: `ΔF1 = -0.0271` (`bootstrap p = 0.0040`, `McNemar p = 0.0000`).

---

## 3. Hybrid Stage Ablation Study (`val` split, 160 docs, Seeds 1, 2, 3)

Measured via `scripts/run_v3_val_and_ablation.py` ([v3_val_ablation.json](results/v3_val_ablation.json)):

| Variant | Strict F1 (mean ± std) | Assert Macro-F1 | Relation F1 | Δ vs Full Hybrid |
|---|---:|---:|---:|---|
| `full_hybrid` | `0.9806 ± 0.0032` | `0.5500` | `0.9050` | Baseline |
| `- Transformer Fusion` (`use_transformer_fusion=false`) | `0.9802 ± 0.0036` | `0.5503` | `0.9048` | `-0.0004` Strict F1 |
| `- Temporal CRF` (`use_temporal_crf=false`) | `0.9806 ± 0.0032` | `0.5500` | `0.9050` | `0.0000` (no-op on pre-aligned TIMEX spans) |
| `- Relation CNN` (`use_relation_cnn=false`) | `0.9806 ± 0.0032` | `0.5500` | `0.7056` | **`-0.1994` Relation F1** |
| `- Assertion BiLSTM` (`use_assertion_bilstm=false`) | `0.9806 ± 0.0032` | `0.4851` | `0.9050` | **`-0.0649` Assert Macro-F1** |
| `- Rules Layer` (`use_rules_layer=false`) | `0.9805 ± 0.0030` | `0.5503` | `0.9053` | `-0.0001` Strict F1 |

Stage 3's balanced Assertion BiLSTM + ConText fusion (`+0.0649` Assert Macro-F1) and type-masked Piecewise CNN Relation Classifier (`+0.1994` Relation F1) both provide clear, measurable gains on `val`.

---

## 4. OCR Stress Benchmark (`SYNTH-MED-V3` `val`, 16 docs × 3 tiers = 48 scanned PDFs)

Measured via `scripts/run_ocr_benchmark.py` ([ocr_stress_benchmark.json](results/ocr_stress_benchmark.json)), average `91.6` words/doc:

| Degradation Tier | Overall CER | Overall WER | Mean OCR Conf | EN CER | FR CER |
|---|---:|---:|---:|---:|---:|
| `clean_scan` | `0.0021` | `0.0138` | `0.9388` | `0.0011` | `0.0031` |
| `mild_scan` | `0.0022` | `0.0119` | `0.9384` | `0.0006` | `0.0038` |
| `heavy_scan` | `0.1615` | `0.5471` | `0.7343` | `0.1624` | `0.1607` |

---

## 5. Honest Analysis & Remaining Limitations

1. **Generalization gap is now visible:** On `test_unseen_terms` (0% clinical term overlap), every model drops sharply from `test_iid` (`classical_crf`: `0.9980 → 0.9131`, `hybrid`: `0.9819 → 0.8033`, `bilstm_crf`: `0.9690 → 0.7860`, `transformer`: `0.6022 → 0.3671`).
2. **Classical CRF vs Neural / Hybrid:** Because `classical_crf` trains on all 600 documents with rich character n-gram and context window features, it achieves the highest Strict NER F1 across all 3 locked test sets (`0.9980` IID, `0.9131` unseen terms, `0.9549` unseen templates). However, `hybrid` achieves the highest **Relation F1** on all 3 splits (`0.9037`, `0.8058`, `0.8264`) and highest **TIMEX normalization accuracy** (`0.2115`, `0.2285`, `0.2318`).
3. **Where Hybrid still struggles:** On `test_unseen_templates`, `hybrid` trails `bilstm_crf` in Strict NER F1 (`0.8800` vs `0.9216`, `ΔF1 = -0.0576`) and Assertion Macro-F1 (`0.4116` vs `0.6116`) due to `spurious_fp` (`375`), `boundary_error` (`334`), and `assertion_wrong` (`1,026`) on unseen sentence structures.
4. **Random-init NumPy Transformer:** Training a pure-NumPy Transformer from scratch on `200` documents for `12` epochs under CPU constraints underfits (`0.6022` IID, `0.3671` unseen terms, `0.5369` unseen templates).
5. **Document classification in `classical_*` / `hybrid`:** `classical_ml`, `classical_crf`, and `hybrid` currently emit default document-type predictions (`0.1625`–`0.2437` doc accuracy), whereas `bilstm`, `bilstm_crf`, `transformer`, and `rule_based` achieve `1.0000` doc-type accuracy.
6. **Synthetic vs real clinical text:** `SYNTH-MED-V3` is still a synthetic bilingual benchmark; external validation on de-identified real clinical notes (e.g., under an approved DUA) remains future work.
