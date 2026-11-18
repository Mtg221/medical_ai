# Bilingual Medical Information Extraction — Research Report (`SYNTH-MED-V3.1`)

**Date:** 2026-11-15  
**Dataset:** `SYNTH-MED-V3.1` (`data/processed/v3_1/`, 1,240 documents: 620 EN + 620 FR; 600 train, 160 val, 480 across 3 locked test splits)  
**Execution Hardware:** MacBook Pro Intel Core i7-9750H @ 2.60 GHz (6 cores / 12 threads), 32 GB RAM, **CPU-only**, **from-scratch pure NumPy** ([`DECISIONS/ADR-0009-numpy-vs-pytorch.md`](../DECISIONS/ADR-0009-numpy-vs-pytorch.md))  
**Total Final 5-Seed Locked Evaluation Runtime:** `3415.92s` (7 V3.1 approaches × 5 seeds `1..5` × 4 splits) + `4265.71s` (`hybrid_v2` `val` selection + 5-seed locked eval)

---

## 1. Executive Summary

This report documents the end-to-end scientific evaluation of seven information extraction approaches—five single-model paradigms (`rule_based`, `classical_ml`, `classical_crf`, `bilstm`, `bilstm_crf`), a stacked multi-head BPE `transformer` built from scratch, and our 5-stage `hybrid` pipeline (plus its `val`-selected `hybrid_v2` configuration)—under **strict split isolation**, **equal 600-document training budget**, **6/6 verified analytical gradients**, **5 random seeds (`1..5`)**, and **10,000-resample document-level bootstrap 95% CIs with Holm-Bonferroni correction** (`α = 0.05`, 21 pairwise comparisons per split).

### Headline Findings (5-Seed Means on `SYNTH-MED-V3.1`, Document-Level 95% Bootstrap CIs)

1. **Per-Task Winners (No Single Overall Winner):**
   - **In-Distribution Entity NER (`test_iid`, n=160):** `classical_ml` (`0.9983` [0.9966, 0.9997]), `hybrid_v2` (`0.9972` [0.9954, 0.9987]), `classical_crf` (`0.9968` [0.9948, 0.9986]), and `hybrid` (`0.9941` [0.9920, 0.9959]) all exceed `0.99` Strict F1.
   - **Unseen Terminology Entity NER (`test_unseen_terms`, n=160, 0% lexicon overlap):** `hybrid_v2` (`0.9204` [0.9078, 0.9322]) and `classical_crf` (`0.9169` [0.9046, 0.9286]) lead Strict F1 (because `hybrid_v2` enables `use_crf_ensemble: true`; paired `p = 0.1076`), substantially ahead of `classical_ml` (`0.8441`), `hybrid` (`0.8277`), `bilstm_crf` (`0.8141`), `bilstm` (`0.7859`), `transformer` (`0.5241`), and `rule_based` (`0.3233`).
   - **Unseen Template Entity NER (`test_unseen_templates`, n=160, 0.6% masked sentence overlap):** `classical_ml` (`0.9670` [0.9603, 0.9732]) and `classical_crf` (`0.9527` [0.9441, 0.9607]) win Strict F1, ahead of `bilstm_crf` (`0.9392`), `hybrid_v2` (`0.9310`), and `hybrid` (`0.8949`).
   - **Assertion Classification:** `classical_crf` leads on `test_unseen_terms` (`0.7511`) and `test_unseen_templates` (`0.6459`, tied with `transformer` `0.6462`), while `hybrid` reaches `0.7325` on `test_iid`.
   - **Relations & Temporal (`hybrid` / `hybrid_v2`):** `hybrid` / `hybrid_v2` lead **Relation F1** on every split (`0.8912–0.8922` `test_iid`, `0.8127–0.8362` `test_unseen_terms`, `0.8029–0.8315` `test_unseen_templates` vs `0.7197–0.7341` for `classical_crf`) and **TIMEX F1** (`0.9969–1.0000` vs `0.7160–0.7340`).
2. **TIMEX Normalisation Caveat (H7):** `hybrid` / `hybrid_v2` achieve `1.0000` TIMEX ISO normalisation on `SYNTH-MED-V3.1`, **however the normaliser was written for the generator's date formats; `1.0000` on synthetic data does not predict accuracy on real notes.** On a hand-written 40-expression out-of-format EN+FR check (`tests/test_timex_out_of_format.py`), the normaliser achieves **`4/40` (`0.1000`)**.
3. **Real OCR End-to-End Degradation (`test_iid`, 120 pages, H3-Fixed):** After fixing `map_ocr_spans_to_reference` to use character-level `difflib` alignment and enabling `detect_column_split`, real Tesseract OCR (`eng` + `fra`) yields overall `mean_cer = 0.0862` / `mean_wer = 0.2112`, reducing `classical_crf` Strict F1 by **`-0.0850`** (`0.9978 -> 0.9128`) and `hybrid` by **`-0.1139`** (`0.9937 -> 0.8799`). On clean 300 dpi scans (`tier_1`, `CER 0.0046`), `classical_crf` shows **`0.0000`** drop (`0.9961 -> 0.9961`).
4. **Free-Text Evaluation Status (H1, H2):** **Free-text evaluation: not done yet — waiting for owner-written notes (see [`docs/REALISTIC_SET_PROTOCOL.md`](REALISTIC_SET_PROTOCOL.md)).**

---

## 2. Corpus Design & Realism (`SYNTH-MED-V3.1`)

`SYNTH-MED-V3.1` (`data/processed/v3_1/`, built and audited via [`scripts/audit_corpus.py`](../scripts/audit_corpus.py), protocol in [`docs/REALISTIC_SET_PROTOCOL.md`](REALISTIC_SET_PROTOCOL.md)) fixes the header-leakage, template repetition, and equal-budget flaws of earlier synthetic iterations:

- **Splits (1,240 documents total, 50% EN / 50% FR):** `train` (`600`), `val` (`160`), `test_iid` (`160`), `test_unseen_terms` (`160`, 0% entity surface overlap with `train`), `test_unseen_templates` (`160`, 0.6% masked sentence overlap with `train`).
- **Document Types (6 types, non-trivial structural cues):** `discharge_summary`, `radiology_report`, `prescription`, `lab_report`, `consultation_note`, `pathology_report`. Header tags are stripped/randomized so `DocumentTypeClassifier` cannot rely on literal title strings; it still reaches `1.0000` doc-type accuracy from section vocabulary and clinical register.
- **Locked Test Discipline:** All hyperparameter tuning and ablation checks used `val` only; all `test_*` runs are logged chronologically in [`docs/results/TEST_SET_LOG.md`](results/TEST_SET_LOG.md).

---

## 3. Model Architectures, Gradient Verification & Fair Training Protocol

Every model trains **exclusively from random initialization** in **pure NumPy** ([`DECISIONS/ADR-0009-numpy-vs-pytorch.md`](../DECISIONS/ADR-0009-numpy-vs-pytorch.md)) on the exact same `600`-document `train` split per seed:

1. **Gradient Verification (6/6 Passed, `< 1e-4` relative error):** Centered finite-difference checks (`eps=1e-5`) pass for `BiLSTMSequenceTagger`, `BiLSTMCRFTagger`, `TransformerNER`, `BiLSTMAssertionClassifier`, `CNNRelationClassifier`, and `CRFTemporalTagger` (`tests/test_neural_gradient_checks.py`).
2. **Stacked Multi-Head BPE Transformer (`transformer`):** Upgraded from the single-block baseline (`0.5765` Strict F1 on `val`, seed 1) via controlled `val` sweeps (`docs/results/v3_1_transformer_tuning.json`):
   - `L2_H4_D32_ep15`: `0.7734` Strict / `0.8740` Lenient (`111.94s`)
   - `L2_H4_D48_ep18`: `0.8095` Strict / `0.9012` Lenient (`168.96s`)
   - `L3_H4_D48_ep18`: `0.7983` Strict / `0.8977` Lenient (`200.84s`)
   - **`L2_H4_D64_ep18` (selected):** **`0.8192`** Strict / `0.9073` Lenient (`222.75s`) — a `+0.2427` gain over single-block, yet still limited by pure-NumPy 600-doc capacity on OOV clinical subwords (`0.5241` on `test_unseen_terms`).

### Equal-Budget Learning Curve (`val`, seed 1, `docs/results/v3_1_learning_curve.json`)

| Train Docs | `classical_crf` Strict F1 | `bilstm_crf` Strict F1 | `transformer` Strict F1 (baseline) | `hybrid` Strict F1 |
|---|---|---|---|---|
| 100 | `0.9082` (`2.25s`) | `0.8898` (`23.58s`) | `0.5262` (`2.46s`) | `0.8891` (`66.59s`) |
| 200 | `0.9551` (`4.02s`) | `0.9439` (`38.77s`) | `0.5509` (`4.18s`) | `0.9540` (`83.77s`) |
| 400 | `0.9915` (`7.27s`) | `0.9832` (`94.66s`) | `0.5742` (`7.01s`) | `0.9885` (`127.88s`) |
| 600 | `0.9961` (`11.13s`) | `0.9888` (`95.58s`) | `0.5765` (`18.42s`) | `0.9961` (`168.75s`) |

Plot: [`docs/results/v3_1_learning_curve.svg`](results/v3_1_learning_curve.svg)

---

## 4. Final 5-Seed Benchmark Results (`seeds 1..5`, `SYNTH-MED-V3.1`)

All main Strict F1 CIs below are **95% document-level bootstrap CIs (`10,000` resamples)** for both deterministic and stochastic approaches (`strict_f1_pooled_boot_ci95`), with **Seed Std** reported in a separate column (`0.0000` for deterministic models). Full pairwise Holm-Bonferroni p-values (`α=0.05`, 21 pairs per split) are in `docs/results/v3_1_{val,test_iid,test_unseen_terms,test_unseen_templates}_significance_tables.md`.

> **TIMEX Normalisation Caveat (H7):** The TIMEX normaliser was written for the synthetic generator's date formats; `1.0000` on `SYNTH-MED-V3.1` does not predict accuracy on real clinical notes (`4/40 = 0.1000` on out-of-format expressions in `tests/test_timex_out_of_format.py`).

### 4.1 `test_iid` (n=160 docs, In-Distribution)

| Approach | Strict F1 [Doc Boot 95% CI] | Seed Std | Lenient F1 | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm (synthetic only) |
|---|---|---|---|---|---|---|---|
| `rule_based` | `0.3221` [0.2964, 0.3476] | `0.0000` | `0.3343` | `0.5878` | `0.0425` | `0.5101` | `0.1984` |
| `classical_ml` | **`0.9983`** [0.9966, 0.9997] | `0.0000` | **`0.9993`** | `0.7138` | `0.7340` | `0.7160` | `0.1217` |
| `classical_crf` | `0.9968` [0.9948, 0.9986] | `0.0000` | `0.9982` | `0.7194` | `0.7341` | `0.7167` | `0.1217` |
| `bilstm` | `0.9659` [0.9613, 0.9703] | `0.0037` | `0.9821` | `0.7198` | `0.7208` | `0.6997` | `0.1198` |
| `bilstm_crf` | `0.9884` [0.9859, 0.9907] | `0.0029` | `0.9928` | `0.7168` | `0.7342` | `0.7155` | `0.1204` |
| `transformer` | `0.7930` [0.7762, 0.8098] | `0.0171` | `0.8857` | `0.7305` | `0.5978` | `0.4557` | `0.1549` |
| `hybrid` | `0.9941` [0.9920, 0.9959] | `0.0033` | `0.9974` | **`0.7325`** | **`0.8922`** | **`0.9986`** | **`1.0000`** |

*Significance highlights (`test_iid`):* 20 of 21 pairs are significant at Holm-corrected `α=0.05` (`p = 0.0042`–`0.0156`); only `classical_ml` vs `classical_crf` (`Δ = -0.0015`, Holm `p = 0.1464`) is non-significant.

### 4.2 `test_unseen_terms` (n=160 docs, 0% Clinical Lexicon Overlap)

| Approach | Strict F1 [Doc Boot 95% CI] | Seed Std | Lenient F1 | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm (synthetic only) |
|---|---|---|---|---|---|---|---|
| `rule_based` | `0.3233` [0.2966, 0.3488] | `0.0000` | `0.3385` | `0.6602` | `0.0368` | `0.5096` | `0.2072` |
| `classical_ml` | `0.8441` [0.8261, 0.8617] | `0.0000` | **`0.9353`** | `0.7446` | `0.6701` | `0.7179` | `0.1265` |
| `classical_crf` | **`0.9169`** [0.9046, 0.9286] | `0.0000` | `0.9319` | **`0.7511`** | `0.7197` | `0.7179` | `0.1265` |
| `bilstm` | `0.7859` [0.7639, 0.8077] | `0.0102` | `0.8588` | `0.7411` | `0.6655` | `0.6994` | `0.1230` |
| `bilstm_crf` | `0.8141` [0.7941, 0.8334] | `0.0064` | `0.8416` | `0.7209` | `0.7018` | `0.7123` | `0.1222` |
| `transformer` | `0.5241` [0.4996, 0.5493] | `0.0104` | `0.6980` | `0.7075` | `0.3984` | `0.4505` | `0.1578` |
| `hybrid` | `0.8277` [0.8085, 0.8465] | `0.0234` | `0.8452` | `0.6672` | **`0.8127`** | **`1.0000`** | **`1.0000`** |

*Significance highlights (`test_unseen_terms`):* **All 21/21 pairwise Strict F1 comparisons are statistically significant** after Holm-Bonferroni correction (`p <= 0.0146`); `classical_crf` beats `classical_ml` (`+0.0728`, `p = 0.0042`) and `hybrid` (`+0.0891`, `p = 0.0042`).

### 4.3 `test_unseen_templates` (n=160 docs, 0.6% Masked Sentence Overlap)

| Approach | Strict F1 [Doc Boot 95% CI] | Seed Std | Lenient F1 | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm (synthetic only) |
|---|---|---|---|---|---|---|---|
| `rule_based` | `0.3066` [0.2799, 0.3327] | `0.0000` | `0.3165` | `0.5430` | `0.0506` | `0.5029` | `0.1781` |
| `classical_ml` | **`0.9670`** [0.9603, 0.9732] | `0.0000` | **`0.9801`** | `0.6453` | `0.7105` | `0.7340` | `0.1032` |
| `classical_crf` | `0.9527` [0.9441, 0.9607] | `0.0000` | `0.9701` | `0.6459` | `0.7235` | `0.7340` | `0.1032` |
| `bilstm` | `0.8619` [0.8497, 0.8740] | `0.0134` | `0.9226` | `0.6440` | `0.6453` | `0.7154` | `0.0999` |
| `bilstm_crf` | `0.9392` [0.9316, 0.9466] | `0.0094` | `0.9721` | `0.6399` | `0.6999` | `0.7309` | `0.0994` |
| `transformer` | `0.6886` [0.6720, 0.7048] | `0.0148` | `0.7816` | **`0.6462`** | `0.5255` | `0.4620` | `0.1271` |
| `hybrid` | `0.8949` [0.8841, 0.9055] | `0.0175` | `0.9644` | `0.5283` | **`0.8029`** | **`0.9969`** | **`1.0000`** |

*Significance highlights (`test_unseen_templates`):* **All 21/21 pairwise Strict F1 comparisons are statistically significant** after Holm-Bonferroni correction (`p = 0.0042`).

### 4.4 `val` Reference (n=160 docs, Development Split)

| Approach | Strict F1 [Doc Boot 95% CI] | Seed Std | Lenient F1 | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm (synthetic only) |
|---|---|---|---|---|---|---|---|
| `rule_based` | `0.3148` [0.2869, 0.3418] | `0.0000` | `0.3356` | `0.4354` | `0.0465` | `0.5073` | `0.1934` |
| `classical_ml` | `0.9973` [0.9949, 0.9992] | `0.0000` | `0.9993` | `0.7573` | `0.7192` | `0.7140` | `0.1184` |
| `classical_crf` | `0.9961` [0.9938, 0.9981] | `0.0000` | `0.9981` | `0.7567` | `0.7186` | `0.7140` | `0.1184` |
| `bilstm` | `0.9665` [0.9620, 0.9710] | `0.0042` | `0.9850` | `0.7548` | `0.7005` | `0.7009` | `0.1190` |
| `bilstm_crf` | `0.9897` [0.9870, 0.9922] | `0.0039` | `0.9933` | `0.7544` | `0.7155` | `0.7127` | `0.1176` |
| `transformer` | `0.7852` [0.7677, 0.8024] | `0.0242` | `0.8842` | `0.7499` | `0.5707` | `0.4740` | `0.1472` |
| `hybrid` | `0.9942` [0.9921, 0.9961] | `0.0033` | `0.9974` | `0.7350` | `0.8937` | `1.0000` | `1.0000` |

---

## 5. Hybrid Stage 1–5 Ablation Study (`seeds 1..5`, `docs/results/v3_1_step4_ablations.json`)

| Variant | `val` Strict / Assert / Rel / TIMEX | `test_unseen_terms` Strict / Assert / Rel / TIMEX | `test_unseen_templates` Strict / Assert / Rel / TIMEX | Interpretation |
|---|---|---|---|---|
| `full_hybrid` | `0.9942 / 0.7350 / 0.8937 / 1.0000` | `0.8277 / 0.6672 / 0.8127 / 1.0000` | `0.8949 / 0.5283 / 0.8029 / 0.9969` | Baseline 5-stage system |
| `no_rules_layer` (Stage 2) | `0.9942 / 0.7350 / 0.8937 / 1.0000` | `0.8267 / 0.6676 / 0.8126 / 1.0000` | `0.8955 / 0.5283 / 0.8030 / 0.9969` | **Negligible** (`+0.0010` unseen terms, `-0.0006` unseen templates) |
| `no_assertion_bilstm` (Stage 3b) | `0.9942 / 0.7536 / 0.8937 / 1.0000` | `0.8277 / 0.7041 / 0.8127 / 1.0000` | `0.8949 / 0.6405 / 0.8029 / 0.9969` | **Negative result:** BiLSTM assertion hurts vs ConText/NegEx (`-0.0186` to `-0.1122` Assert F1) |
| `no_relation_cnn` (Stage 3) | `0.9942 / 0.7350 / 0.7202 / 1.0000` | `0.8277 / 0.6672 / 0.7041 / 1.0000` | `0.8949 / 0.5283 / 0.6861 / 0.9969` | **Strong positive:** CNN adds `+0.1735` (`val`), `+0.1086` (unseen terms), `+0.1168` (unseen templates) Rel F1 |
| `no_temporal_crf` (Stage 4) | `0.9942 / 0.7350 / 0.8937 / 0.7140` | `0.8277 / 0.6672 / 0.8127 / 0.7163` | `0.8949 / 0.5283 / 0.8029 / 0.7305` | **Strong positive:** Stage 4 lifts TIMEX F1 by `+0.2664–0.2860` and synthetic Norm from `0.103–0.126` to `1.0000` |
| `no_transformer_fusion` (Stage 5) | `0.9942 / 0.7350 / 0.8937 / 1.0000` | `0.8277 / 0.6672 / 0.8127 / 1.0000` | `0.8949 / 0.5283 / 0.8029 / 0.9969` | **Zero delta (`0.0000`)** across all metrics and splits |

---

## 6. Full-Size Real OCR Stress & End-to-End Benchmark (`120 pages`, `test_iid`, H3-Fixed)

Executed with real Tesseract (`eng` + `fra`) via `scripts/run_v3_1_ocr_full_benchmark.py` (`docs/results/v3_1_ocr_full_benchmark.json`, `549.87s`) after fixing `map_ocr_spans_to_reference` to align at the **character level** (`difflib.SequenceMatcher`, 100% exact on identical text across 400 docs in `tests/test_ocr_span_mapping.py`) and enabling `detect_column_split`:

| Cohort / Tier | Pages | Mean CER | Mean WER | `classical_crf` Clean -> OCR (Drop) | `hybrid` Clean -> OCR (Drop) |
|---|---|---|---|---|---|
| **Overall (`test_iid`)** | **120** | **`0.0862`** | **`0.2112`** | `0.9978 -> 0.9128` (**`-0.0850`**) | `0.9937 -> 0.8799` (**`-0.1139`**) |
| English (`en`) | 60 | `0.0725` | `0.1811` | `0.9978 -> 0.9234` (`-0.0744`) | `0.9921 -> 0.8886` (`-0.1036`) |
| French (`fr`) | 60 | `0.0999` | `0.2412` | `0.9977 -> 0.9021` (`-0.0956`) | `0.9954 -> 0.8712` (`-0.1242`) |
| `tier_1_300dpi_clean` | 20 | `0.0046` | `0.0258` | `0.9961 -> 0.9961` (`0.0000`) | `0.9923 -> 0.9859` (`-0.0064`) |
| `tier_2_200dpi_mild_skew_jpeg` | 20 | `0.0056` | `0.0315` | `0.9974 -> 0.9857` (`-0.0117`) | `0.9961 -> 0.9804` (`-0.0157`) |
| `tier_3_phone_perspective_light` | 20 | `0.0092` | `0.0604` | `1.0000 -> 0.9834` (`-0.0166`) | `0.9931 -> 0.9574` (`-0.0357`) |
| `tier_4_two_col_table_stamps` | 20 | `0.0275` | `0.1112` | `0.9987 -> 0.9049` (`-0.0939`) | `0.9975 -> 0.8852` (`-0.1122`) |
| `tier_5_table_multifont_skew5` | 20 | `0.3673` | `0.5397` | `0.9972 -> 0.7754` (`-0.2218`) | `0.9931 -> 0.7489` (`-0.2442`) |
| `tier_6_150dpi_heavy_blur_noise` | 20 | `0.1027` | `0.4985` | `0.9970 -> 0.7988` (`-0.1983`) | `0.9896 -> 0.6881` (`-0.3015`) |

### Old (H3-Buggy Token Mapping) vs. New (Char-Level Alignment + Column Split)

| Tier / Overall | Old CER | New CER | Old `classical_crf` OCR F1 | New `classical_crf` OCR F1 | Old `hybrid` OCR F1 | New `hybrid` OCR F1 |
|---|---|---|---|---|---|---|
| **Overall (120 pages)** | `0.1927` | **`0.0862`** | `0.5655` (`-0.4322`) | **`0.9128`** (`-0.0850`) | `0.5332` (`-0.4605`) | **`0.8799`** (`-0.1139`) |
| `tier_1_300dpi_clean` | `0.0046` | `0.0046` | `0.6110` | **`0.9961`** | `0.6033` | **`0.9859`** |
| `tier_2_200dpi_mild_skew_jpeg` | `0.0056` | `0.0056` | `0.6319` | **`0.9857`** | `0.6301` | **`0.9804`** |
| `tier_3_phone_perspective_light` | `0.0092` | `0.0092` | `0.6630` | **`0.9834`** | `0.6382` | **`0.9574`** |
| `tier_4_two_col_table_stamps` | `0.6670` | **`0.0275`** | `0.4173` | **`0.9049`** | `0.3771` | **`0.8852`** |
| `tier_5_table_multifont_skew5` | `0.3673` | `0.3673` | `0.5120` | **`0.7754`** | `0.4772` | **`0.7489`** |
| `tier_6_150dpi_heavy_blur_noise` | `0.1027` | `0.1027` | `0.5208` | **`0.7988`** | `0.4356` | **`0.6881`** |

**Conclusion:** The previous `-0.39` to `-0.43` F1 drop on clean OCR (`tier_1`–`tier_3`) was an artifact of whole-token span mapping failing on adjacent punctuation (`63.7%` identity rate). With character-level alignment, OCR F1 drop on `tier_1`–`tier_3` (`0.0000` to `-0.0357`) is consistent with their `< 1%` CER, and column splitting recovers `tier_4` CER from `0.6670` to `0.0275`. Residual OCR degradation is concentrated in multi-font skewed tables (`tier_5`, `CER 0.3673`) and heavy blur/noise (`tier_6`, `CER 0.1027`).

---

## 7. Hybrid v2 (`val`-Selected) & Per-Task Winners (H5)

> **Methodological Disclosure:** An earlier 3-seed Hybrid v2 run (`docs/results/archive/2026-11-15_realistic_v1_invalid/phase12_hybrid_v2_and_realistic.json`) was configured after inspecting test-split ablations and is archived as **test-informed**. Below are the clean `val`-only 5-seed selection results and the single logged locked test evaluation (`scripts/run_v3_2_hybrid_v2_val_selection.py`, `4265.71s`).

### 7.1 5-Seed `val` Selection (`docs/results/v3_2_hybrid_v2_val_selection.json`)

| Candidate | `use_crf_ensemble` | `use_assertion_bilstm` | `use_transformer_fusion` | `val` Strict F1 [Doc Boot 95% CI] | Seed Std | Assert F1 | Rel F1 | 5-Seed Time |
|---|---|---|---|---|---|---|---|---|
| `v3_1_baseline` | `false` | `true` | `true` | `0.9942` [0.9921, 0.9961] | `0.0033` | `0.7350` | `0.8937` | `764.44s` |
| `crf_off_ctx_nofusion` | `false` | `false` | `false` | `0.9942` [0.9921, 0.9961] | `0.0033` | `0.7536` | `0.8937` | `615.35s` |
| `crf_on_bilstm_fusion` | `true` | `true` | `true` | `0.9966` [0.9962, 0.9980] | `0.0009` | `0.7364` | `0.8916` | `946.71s` |
| `crf_on_ctx_fusion` | `true` | `false` | `true` | `0.9966` [0.9962, 0.9980] | `0.0009` | `0.7567` | `0.8916` | `663.21s` |
| **`crf_on_ctx_nofusion` (selected)** | **`true`** | **`false`** | **`false`** | **`0.9966`** [0.9945, 0.9984] | `0.0009` | **`0.7567`** | `0.8916` | **`603.77s`** |

### 7.2 One-Time Locked Test Evaluation (`docs/results/v3_2_hybrid_v2_locked_eval.json`)

Note honestly what `hybrid_v2` is: **the hybrid's entity stage now uses the CRF ensemble (`use_crf_ensemble: true`); the hybrid's own contribution is relations + temporal.**

| Split | `hybrid_v2` Strict F1 [Doc Boot 95% CI] | Seed Std | `classical_crf` Strict F1 [Doc Boot 95% CI] | `hybrid_v2` Assert / Rel / TIMEX | `classical_crf` Assert / Rel / TIMEX | Paired `Δ` Strict F1 (160-doc Boot `p`) |
|---|---|---|---|---|---|---|
| `test_iid` (160) | **`0.9972`** [0.9954, 0.9987] | `0.0011` | `0.9968` [0.9948, 0.9986] | `0.7188 /` **`0.8912` / `0.9986`** | **`0.7194`** `/ 0.7341 / 0.7167` | `+0.0003` (`p = 0.3814`, n.s.) |
| `test_unseen_terms` (160) | **`0.9204`** [0.9078, 0.9322] | `0.0065` | `0.9169` [0.9046, 0.9286] | `0.7497 /` **`0.8362` / `1.0000`** | **`0.7511`** `/ 0.7197 / 0.7179` | `+0.0034` (`p = 0.1076`, n.s.) |
| `test_unseen_templates` (160) | `0.9310` [0.9219, 0.9397] | `0.0102` | **`0.9527`** [0.9441, 0.9607] | `0.6430 /` **`0.8315` / `0.9969`** | **`0.6459`** `/ 0.7235 / 0.7340` | `-0.0217` (`classical_crf` wins) |

---

## 8. Platform, Review Persistence, E2E & Accessibility Verification (H8)

- **Review Queue Persistence & Gold Export:** `POST /api/v1/reviews`, `GET /api/v1/reviews`, and `GET /api/v1/reviews/export` (`jsonl` and `brat`) persist reviewer decisions and corrected span offsets in SQLite (`tests/test_review_api_persistence.py`).
- **Full End-to-End Playwright Suite (`frontend/e2e/app.spec.ts`):** Verified against a live FastAPI backend (`scripts/run_frontend_e2e_and_lighthouse.py`) in both **EN and FR**: login → upload PDF (including `"Le patient presente une pneumonie et prend metformine 500 mg."` → `"500 mg"` highlighted as `DOSAGE` linked to `metformine` via `HAS_DOSAGE`) → open fact → review decision saved after reload → `/react/relations` renders graph nodes & edges (`node` click navigates to workspace span) → download markdown report (`?lang=en|fr`).
- **28 Light & Dark Screenshots (`docs/screenshots/`):** Playwright script generates light + dark screenshots for all 14 routes (`login`, `workspace`, `upload`, `pdf`, `relations`, `timeline`, `reports`, `review`, `corpus`, `benchmark`, `ablation`, `errors`, `schema`, `settings`).
- **Lighthouse Accessibility (`docs/results/lighthouse_accessibility_report.json`):** Scored **`100/100`** (`>= 95` threshold) across all 6 tested routes (`login`, `workspace`, `upload`, `relations`, `review`, `reports`).

---

## 9. Limitations & Open Engineering Work

1. **Free-Text Evaluation Pending Owner Notes (H1, H2):** Free-text evaluation: not done yet — waiting for owner-written notes (see [`docs/REALISTIC_SET_PROTOCOL.md`](REALISTIC_SET_PROTOCOL.md) and [`docs/REALISTIC_SET_OWNER_CHECKLIST.md`](REALISTIC_SET_OWNER_CHECKLIST.md)). Tooling (`configs/label_studio_medical_en_fr.xml`, `scripts/import_label_studio_realistic.py`, `scripts/eval_realistic.py`) is ready with strict label validation.
2. **TIMEX Generalization Outside Synthetic Formats (H7):** Out-of-format TIMEX accuracy is `4/40` (`0.1000`); rule/parser coverage must expand once real clinical notes arrive.
3. **Pure-NumPy CPU Ceiling:** Training all 7 models × 5 seeds takes `~57 min` on 6 CPU cores and limits neural model depth/batch size ([`DECISIONS/ADR-0009-numpy-vs-pytorch.md`](../DECISIONS/ADR-0009-numpy-vs-pytorch.md)).
4. **Severe OCR Noise Tiers (`tier_5`–`tier_6`):** Multi-font skewed tables (`CER 0.3673`) and heavy blur (`CER 0.1027`) still cause `-0.20` to `-0.30` F1 drops.

---

## 10. Changes Since the 11B Report (Correction Prompt 12B: H1–H9)

| ID | Finding in 11B / Initial Phase 12 | Remediation & How Numbers Moved |
|---|---|---|
| **H1, H2** | `realistic_80` was single-frame template-generated and used mismatched labels (`PROBLEM`, `TEST`, 43% unmatchable). | Archived `realistic_v1` & its `0.25–0.26` F1 results to [`docs/results/archive/2026-11-15_realistic_v1_invalid/`](results/archive/2026-11-15_realistic_v1_invalid/README.md); added Label Studio XML, strict importer failing on unknown labels, `eval_realistic.py`, and owner checklist. |
| **H3** | `map_ocr_spans_to_reference` mapped whitespace tokens (`63.7%` identity rate), inflating OCR drops (`-0.4322`). | Rewrote at character level (`difflib.SequenceMatcher`, `100%` exact on 400 docs) + column split. `classical_crf` overall OCR F1 moved `0.5655 -> 0.9128` (`tier_1`: `0.6110 -> 0.9961`, `tier_4`: `0.4173 -> 0.9049`). |
| **H4** | 11 tests failed on a fresh clone (git-ignored `v3`/`v3_1` + Windows `detect_column_split`). | Added session fixture in `tests/conftest.py`, fixed active-row `detect_column_split` & cross-platform fonts, and added Windows/macOS/Ubuntu Python 3.12/3.13 CI matrix. |
| **H5** | Initial `hybrid_v2` was chosen after viewing test ablations. | Archived test-informed run; ran 5-seed `val`-only selection (`crf_on_ctx_nofusion`) and 1 logged locked test run (`0.9972` `test_iid`, `0.9204` `test_unseen_terms`, `0.9310` `test_unseen_templates`), reporting per-task winners honestly. |
| **H6** | Deterministic models reported zero-width seed CIs (`[0.9169, 0.9169]`). | Reported 10,000-resample **document-level bootstrap 95% CIs** for every approach, with `Seed Std` in a separate column. |
| **H7** | `1.0000` TIMEX normalisation lacked circularity caveat. | Added explicit synthetic-only caveat wherever cited and added 40-expression out-of-format unit check (`4/40 = 0.1000`). |
| **H8** | Screenshots covered 1/14 pages; E2E omitted upload/highlights/report download. | Added EN+FR full workflow E2E in CI, 28 screenshots (14 pages × 2 themes), fixed `ReportsPage.tsx` download, and achieved `100/100` Lighthouse accessibility on all 5 main routes. |
| **H9** | Docs contained machine-local `file:///Users/guest777/...` links. | Replaced all `file:///` links with relative repo paths and added `tests/test_no_file_uri_in_docs.py` in CI. |

---

## 11. Served Model (`hybrid_v2`)

- **Served Default Model:** `hybrid_v2` (`configs/exp/v3_2/hybrid_v2.yaml`, seed `1`), saved to `models/hybrid_v2_seed1/` via `medextract train --config configs/exp/v3_2/hybrid_v2.yaml --seed 1 --out models/hybrid_v2_seed1/` (`138.1s` on CPU). Optional `classical_crf` saved to `models/classical_crf_seed1/` (`10.77s` on CPU). Both directories include SHA-256 `manifest.json` (`medresearch_platform/hybrid/persistence.py`) and fail loudly if missing.
- **Serving Path (`medresearch_platform/serving/extractor.py`):** `ExtractionService` backs `workflows/pipeline.py`, `/api/v1/documents/upload`, `/api/v1/models`, and the Settings provenance panel (`SettingsDiagnosticsPage.tsx`). Legacy `ai/nlp_model.py` and `ai/ocr_model.py` are moved to `extras/legacy/`.
- **API-vs-Pipeline Consistency (`tests/test_api_pipeline_consistency.py`):** Verified that entities returned by `/api/v1/documents/upload` match `HybridPipeline.extract_document` identically across `30` `test_iid` TXT documents and `10` rendered PDFs via Tesseract + character-level span alignment.

---

## 12. Changes Since 12B (Correction Prompt 13: J1–J5)

| ID | Finding in `ac743d6` | Remediation & Verification |
|---|---|---|
| **J1** | App served legacy `ai/nlp_model.py` + `ai/ocr_model.py` instead of evaluated `HybridPipeline`. | Moved legacy models to `extras/legacy/`, built `ExtractionService` wired into `workflows/pipeline.py` and `/api/v1/models` + Settings UI, and passed `tests/test_api_pipeline_consistency.py` (30 TXT + 10 PDF `test_iid` docs) + E2E `"500 mg"` → `DOSAGE` (`HAS_DOSAGE` to `metformine`). |
| **J2** | `HybridPipeline` & `classical_crf` had no `save()`/`load()`. | Added `save`/`load` with NumPy `.npz`, CRF `.crfsuite`, BPE vocab/merges, config, corpus hash, git commit, and SHA-256 `manifest.json` + `medextract train` CLI (`tests/test_model_save_load.py`). |
| **J3** | Bootstrap CIs pooled 5 seeds × 160 docs into 800 items and missed their own point estimates. | Added `bootstrap_mean_seed_f1_ci` resampling the 160 independent documents with the exact mean-across-seeds estimator (`tests/test_statistics_and_holm_bonferroni.py`), regenerated all 4 `v3_1_*_significance` tables and `v3_2_hybrid_v2_locked_eval.json`, and logged in `TEST_SET_LOG.md`. |
| **J4** | Relation graph page rendered blank (`ent.surface_text.slice` crashed when API returned `source_text`). | Fixed `getEntityText` (`source_text ?? surface_text`), added node-click → workspace span navigation, verified `relations_light.png` (`58K`) / `relations_dark.png` (`55K`), E2E `>=1` node & edge assertions, and `100/100` Lighthouse on `/react/relations`. |
| **J5** | Workspace highlights ignored entity type colours and assertion styles. | Added shared `frontend/src/utils/entityStyles.ts` (type colours + `NOT` strikethrough, `?` dashed, `HYP` italic, `FAM`, clock for historical) + visible workspace legend + Vitest unit tests (`entityStyles.test.ts`). |

---

## 13. Changes Since 13 (K1–K6)

| ID | Finding in `edcb2c1` | Remediation & Verification |
|---|---|---|
| **K1, K2, K6** | Clean checkout errored (`122 errors`: `HybridConfig` did not exist in `tests/conftest.py`); API test required 3 models; `README.md` showed stale v2 counts; CI used retired `macos-13` and had no clean-checkout job. | Fixed `tests/conftest.py` to train tiny fixture models via the real `HybridPipeline(seed=1, cfg=cfg)` & `ClassicalCRFModel(seed=1)` APIs (`tests/test_conftest_fresh_clone.py`); isolated `test_api_and_cli.py` from repo `models/`; added `fresh-clone-tests` CI job (`ubuntu-latest` + `windows-latest`, Python `3.12`/`3.13`); updated `README.md` to `SYNTH-MED-V3.1` / `hybrid_v2`. |
| **K3** | `context_negex.py` used raw substring matching (`c in pre_window`), so French `"nie "` matched inside `"pneumonie"`, `"insomnie"`, `"neutropénie"`, `"hernie"`, turning later medications `absent`. | Compiled Unicode word-boundary regexes (`(?<!\w)...(?!\w)`, `re.UNICODE | re.IGNORECASE`), added `"ni"` coordination and clause terminators (`" and takes "`, `" et prend "`, `" traitée par "`, `" started "`), verified via `tests/test_context_negex_word_boundaries.py` (`149 passed`), updated E2E to accented `"Le patient présente une pneumonie et prend metformine 500 mg."` (`data-assertion="present"`), and logged re-measured assertion macro-F1 in `TEST_SET_LOG.md` (`docs/results/v3_2_k3_assertion_recheck.json`). |
| **K4** | Low-confidence OOD predictions (`0.00`–`0.05` on owner free text such as `"Mme D"`, `"67 ans"`, `"Toux productive"`) were persisted and shown as normal facts. | Calibrated `hybrid_v2` on `val` (`2,947` entities, `overall_ece = 0.0019`, `[0.5, 0.6]` bin precision `0.6667 < 0.80` → `review_threshold: 0.60` in `configs/exp/v3_2/hybrid_v2.yaml`, `docs/results/v3_2_val_calibration.json`). Facts below `review_threshold` are flagged `review_status = "needs_review"` (`review_reason = "low_confidence"`), sorted first in the review queue with amber badges, excluded from confirmed report counts, and listed under **To Verify (Low Confidence)** / **À vérifier (faible confiance)** (`tests/test_low_confidence_review.py`). |
| **K5** | Two `val` selection CI rows (`v3_1_baseline`, `crf_off_ctx_nofusion`) still showed stale `0.9942` (old CI `[0.9974, 0.9988]`). | Updated lines 177–178 to `0.9942` [0.9921, 0.9961] (`docs/results/v3_1_val_significance.json`) and added `tests/test_report_ci_bounds.py` to enforce `low <= estimate <= high` on every CI in this report. |

### 13.1 Assertion Macro-F1 Before vs After K3 Word-Boundary Fix

| Split | `hybrid_v2` Before K3 (Overall) | `hybrid_v2` After K3 (Overall) | After K3 (EN) | After K3 (FR) |
|---|---|---|---|---|
| `val` (160) | `0.7567` | **`0.7585`** | `0.8059` | `0.7244` |
| `test_iid` (160) | `0.7188` | `0.7155` | `0.7587` | `0.7015` |
| `test_unseen_terms` (160) | `0.7497` | `0.7485` | `0.7909` | `0.7324` |
| `test_unseen_templates` (160) | `0.6430` | **`0.6463`** | `0.7052` | `0.6034` |

### 13.2 Owner-Style Free-Text Limitation (Measured Probes, Corrected in L1)

`hybrid_v2` is trained exclusively on synthetic clinical notes (`SYNTH-MED-V3.1`). On short, un-templated owner-style free text (`tests/data/owner_style_probe.jsonl`, e.g. `"Mme D, 67 ans. Toux productive depuis 3 jours, fièvre à 38.7°C. Antécédents : hypertension traitée par amlodipine 5 mg."`), the served model produces **both** low-confidence flagged errors and **confidently wrong** unflagged errors:

- **Flagged (`needs_review`, `low_confidence < 0.60`):** `"Toux productive"` → `LAB_TEST` (`0.0566`), `"fièvre"` → `ANATOMY` (`0.0001`), `"°C"` → `LAB_UNIT` (`0.2484`).
- **Confidently wrong (NOT flagged, `>= 0.60`):** `"Mme"` → `DIAGNOSIS` (`0.9169`), `"67 ans"` → `DATE` (`0.8792`).
- **Missed:** `"hypertension"` is omitted.

A confidence threshold cannot catch confident out-of-distribution mistakes; only owner-annotated real clinical notes (training and/or evaluation per [`docs/REALISTIC_SET_PROTOCOL.md`](REALISTIC_SET_PROTOCOL.md)) can measure and reduce them.

---

## 14. Changes Since 14 (Correction Prompt 15: L1–L4)

| ID | Finding in `90c0be6` | Remediation & Verification |
|---|---|---|
| **L1** | §13.2 claimed `"Mme D"`, `"67 ans"`, `"Toux productive"` were all low-confidence and gated. | Rewrote §13.2 with exact measured confidences (`"Mme"` → `DIAGNOSIS 0.9169`, `"67 ans"` → `DATE 0.8792` unflagged; `"Toux productive"` `0.0566` flagged) and added `tests/data/owner_style_probe.jsonl` + `tests/test_owner_style_probes.py` (`xfail`). |
| **L2** | `ExtractedRecord.review_status` stayed `"unreviewed"` while `provenance["review_status"]` held `"needs_review"`. | Extended `ReviewStatus` / `ReviewStatusType` in `contract.py` & `schema_models.py` with `"needs_review"`, set `record.review_status` directly in `serving/extractor.py`, removed `review_status` from `provenance`, updated `workflows/pipeline.py` & `ReviewQueuePage.tsx`, and re-exported `schema/medical_extraction.schema.json`. |
| **L3, L4** | `context_negex.py` had phrase-specific terminators (`" and takes "`, `" et prend "`, etc.) and missed `"aucun "`, `"not "`, `"never "`, `"free of "`, `"non retrouvé(e)"`, `"pas d'argument pour "`, and comma clause-breaks. | Removed all phrase-specific terminators; replaced with general list-scope & conjunction/comma clause-break rules (`_apply_general_clause_scope`) and missing EN/FR cues (`tests/test_context_negex_word_boundaries.py`: `178 passed`), logged in `TEST_SET_LOG.md` (`docs/results/v3_2_l3_l4_assertion_recheck.json`). |

### 14.1 Assertion Macro-F1 Before (K3) vs After (L3/L4)

| Split | `hybrid_v2` Before → After (Overall) | After (EN / FR) | `classical_crf` After (Overall, EN / FR) | `rule_based` Before → After (Overall) |
|---|---|---|---|---|
| `val` (160) | `0.7585` → **`0.7782`** | `0.8285` / `0.7407` | `0.7780` (`0.8282` / `0.7407`) | `0.4354` → **`0.4581`** |
| `test_iid` (160) | `0.7155` → **`0.7289`** | `0.7715` / `0.7131` | `0.7334` (`0.7715` / `0.7233`) | `0.5878` → **`0.6295`** |
| `test_unseen_terms` (160) | `0.7485` → **`0.7674`** | `0.8131` / `0.7499` | `0.7701` (`0.8177` / `0.7407`) | `0.6671` → **`0.6835`** |
| `test_unseen_templates` (160) | `0.6463` → **`0.6524`** | `0.7114` / `0.6095` | `0.6548` (`0.7132` / `0.6128`) | `0.5539` → **`0.5820`** |

---

## 15. Skin Image Analysis (`derm-corpus`, Steps 1–7) & W9/W10 Text Updates

> **Safety Notice (EN / FR):** Research tool — not a diagnosis. A dermatologist must examine the lesion. / Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion.

### 15.1 Lesion-Similarity Grouping, Dataset Audit & `v3` Splits (`DERM_FIELDS.md`, `DERM_CORPUS_CARD_V3.md`)
- **Raw Manifest Field & Dataset Audit ([DERM_FIELDS.md](data/DERM_FIELDS.md)):** Inspected all 14 manifest pages (`573,024` records across `isic`: `549,541`, `fitzpatrick17k`: `11,394`, `scin`: `10,379`, `skindisnet`: `1,710`). **0** records have an explicit `lesion_id` or `patient_id` (`mk` in SCIN stores rater-country counts like `us=3;india=3`). `catalog.py` filtered on `ok == true`, which excluded `fitzpatrick17k` (`11,394`, `CC-BY-NC-4.0`, `3,190` with `mg`), `scin` (`10,379`, `0` with `mg`), and `245,287` `CC-BY-NC` ISIC records. With written owner approval, the `3,190` labelled `fitzpatrick17k` clinical images (atlas labels, lower confidence) were synced and included in `v3`; `scin` (`0` `mg` labels) stayed excluded.
- **Similarity Grouping (`configs/vision/lesion_groups_v3.json`, [lesion_grouping_stats.json](results/vision/lesion_grouping_stats.json)):** Connected components linking `(dHash <= 10 AND centred_cos >= 0.990)` or `dHash <= 2` within `(modality, body_site, diagnosis_harmonised)` yielded **`24,877`** groups across `29,856` cached thumbnails (`23,853` singletons, `982` size 2–5, largest group `1,169` = `3.92%`).
- **Stratified `v3` Corpus ([DERM_CORPUS_CARD_V3.md](data/DERM_CORPUS_CARD_V3.md)):**
  - `train`: `31,565` real thumbnails (`20,630` benign, `8,125` malignant, `2,810` masked `mg`; `8,003` clinical, `22,107` dermoscopic, `1,455` TBP tile).
  - `val`: `1,497` real thumbnails (`900` benign, `597` malignant), filled strictly after `locked_test_v3`.
  - `locked_test_v3` ([locked_test_v3.txt](../configs/vision/locked_test_v3.txt), `sha256=8191dd603b851ff551934666d88736a70a6bff1d7194de91554bddfb4a768209`): **`1,624`** labelled thumbnails (`700` dermoscopic [`400` ben / `300` mal], `574` clinical [`227` ben / `347` mal], `350` TBP tile [`250` ben / `100` mal]; Tone `I`: `75/75`, `II`: `75/84`, `III`: `75/75`, `IV`: `75/75`, `V`: `75/60`, `VI`: `75/27`). **0** group overlap across splits.
- **Superseded History (`locked_test_v2` — W1):** `locked_test_v2` (`n=1,853`, AUROC `0.9398` seed1 / `0.9421` ensemble / `0.9461` no-meta; `classical_logreg` `0.7321`; `numpy_frozen_random_features` `0.7034`; `metadata_only_logreg` `0.6364`) is marked **superseded: images of the same lesion may cross splits (W1)** (`380` similarity groups spanned >1 `v2` split; `78` `locked_test_v2` images shared a group with `train`/`val`).

### 15.2 Real 5-Class PyTorch Picture Triage Router & Text-Presence Safety Guard (`docs/results/vision/step3_real_router_eval.json`)
- **Architecture:** `PyTorchPictureTriageCNN-v1.0.0` (`98,565` params, `96×96` RGB, random init, `pretrained_weights_used = false`, `127.9s` CPU train) + pre-skin `detect_dense_text_lines()` safety guard ([quality.py](../medresearch_platform/vision/quality.py), `0/1,497` false positives on `val_v3` skin images).
- **Training & Evaluation Data:** Real skin thumbnails + phone-photo-augmented rendered SYNTH-MED-V3.1 notes (perspective, shadow, crumple, blur/JPEG, hand edge) + **`334` CC0/PD Wikimedia document scan photos** (`217` train / `34` val / `83` held-out eval; `0` owner-taken phone photos) + `346` Wikimedia Commons CC0/PD everyday photos (`other`, `data/router_real/SOURCES.csv`).
- **Held-Out Evaluation (`n = 472`):** Overall Accuracy = **`0.9004`** (`val` best `0.9029`, `T = 0.9`); `skin -> document_scan` = **`2`**, **`document_scan -> skin` = `0`** (`raw_router_document_scan_to_skin = 1`, blocked by `detect_dense_text_lines`), `other -> skin` = **`1 / 69`**. On the **`83` held-out CC0 document photos** (`>= 50`, none in train/val; `0` owner photos), **`71/83` routed to `document_scan`**, `12/83` to `other`, and **`0/83` to `skin`** (raw router sent `1` to `skin_clinical`, caught by the text-presence safety guard). On `ood_check` (`n = 50`), **`1/50`** routed to skin (`38` `other`, `11` `document_scan`).

### 15.3 Served-Path End-to-End Evaluation & `locked_test_v3` Benchmark (`p18_step1_served_path_eval.json`, `step5_v3_skin_benchmark.json`)

**1. Val-Only Metadata Selection (`val_v3`, `n = 1,497`, `seed=1`, `p18_step1_val_metadata_ablation.json`):**

| Variant (`seed=1`) | Overall Val AUROC | `fitzpatrick17k` (`n=209`) | `isic` (`n=1,288`) | `dermoscopic` (`n=850`) | `clinical` (`n=352`) | `tbp_tile` (`n=295`) |
|---|---:|---:|---:|---:|---:|---:|
| `pixels_only` | `0.7835` | `0.5341` | `0.8035` | `0.8377` | `0.6358` | `0.6547` |
| `modality_only` | `0.8251` (`+0.0416`) | `0.5810` (`+0.0469`) | `0.8450` (`+0.0415`) | `0.8588` | `0.7196` | `0.6100` |
| **`full_body_site_dropout_40`** | **`0.8678`** (`+0.0843`) | **`0.5994`** (`+0.0653`) | **`0.8885`** (`+0.0850`) | **`0.8847`** | **`0.7876`** | **`0.6618`** |

- Selected **`full_body_site_dropout_40`** (`models/derm_cnn_bsdrop40_seed1`, deployed to `models/derm_cnn_seed1`) because its AUROC gain over `pixels_only` survives within both `fitzpatrick17k` (`+0.0653`) and `isic` (`+0.0850`). `serving.py` passes the router's `modality_hint` and user `body_site` to `predict_image(...)` and logs them in `provenance`.

**2. Lead Result — Served-Path End-to-End Evaluation on `locked_test_v3` (`POST /api/v1/images/upload`, `n = 1,624`, `p18_step1_served_path_eval.json`):**
- Router disposition of the `1,624` `locked_test_v3` skin images: `skin_dermoscopic = 759`, `skin_clinical = 392`, `skin_tbp_tile = 384`, `other = 89` (`5.5%` routed away & abstained), `document_scan = 0`; `skin_scored = 934`, `skin_abstained = 601`.

| Served Modality | `n` (`mal` / `ben`) | AUROC [95% CI] | Sens @ 90% Spec | Bal Acc | ECE | Brier | Abstain Rate | High-Risk Prec |
|---|---|---|---|---|---|---|---|---|
| **`dermoscopic`** | `700` (`300` / `400`) | **`0.8849`** [0.8630, 0.9099] | `0.6167` | `0.7929` | `0.0611` | `0.1421` | `0.2629` | `0.8500` |
| **`clinical`** | `574` (`347` / `227`) | **`0.7195`** [0.6739, 0.7613] | **`0.2161`** | `0.6930` | `0.1189` | `0.1991` | `0.2979` | `0.7522` |
| **`tbp_tile`** | `350` (`100` / `250`) | **`0.4819`** [0.4458, 0.5131] | `0.9200` | `0.4640` | `0.2465` | `0.2556` | `0.9571` | `0.7500` |
| **Overall** | `1,624` | **`0.7842`** [0.7646, 0.8057] | `0.4632` | `0.6744` | `0.0555` | `0.1867` | `0.4249` | `0.7972` |

- Corpus-prevalence-reweighted served overall (`3.28%` prev): `PPV = 0.3837`, `NPV = 0.9864`, `ECE = 0.4335`, `abstention = 0.8045`.

**3. Offline Multi-Seed Benchmark (`step5_v3_skin_benchmark.json`, `n = 1,624`, Direct Model Calls Without Router):**

| Modality | `n` (`mal` / `ben`) | AUROC [95% CI] | Sens @ 90% Spec | Bal Acc | ECE | Brier | Abstain Rate | Diagnosis Status (W2) |
|---|---|---|---|---|---|---|---|---|
| **`dermoscopic`** | `700` (`300` / `400`) | **`0.9551`** [0.9397, 0.9690] | `0.8467` | `0.8729` | `0.1033` | `0.0981` | `0.1686` | `not_enough_labelled_data` (`0` specific dx) |
| **`clinical`** | `574` (`347` / `227`) | **`0.7803`** [0.7380, 0.8199] | `0.2478` | `0.7288` | `0.0813` | `0.1735` | `0.2003` | `evaluated` (`n=280`, Macro-F1 `0.3036`, Top-3 `0.7250`) |
| **`tbp_tile`** | `350` (`100` / `250`) | **`0.7923`** [0.7386, 0.8388] | `0.5700` | `0.6500` | `0.0589` | `0.1592` | `0.1171` | `not_enough_labelled_data` (`0` specific dx) |

- **Overall Model Comparison (`n = 1,624`; note overall AUROC mixes modalities with very different base rates):**
  - `pytorch_resnet_3seed_ensemble`: AUROC **`0.8855`** `[0.8705, 0.9012]`, ECE `0.0649`, Brier `0.1366` ($p_{\text{Holm}} < 0.001$ vs all baselines & single seed)
  - `pytorch_resnet_seed1`: `0.8275` `[0.8086, 0.8483]` (`seed=1` stopped epoch 7, `val=0.8302`; `seed=2` `val=0.8833` served; `seed=3` `val=0.8825`)
  - `classical_logreg`: `0.7880` `[0.7661, 0.8087]`, `clinical` Macro-F1 `0.3047`
  - `pytorch_resnet_no_meta` (pixels only): `0.7772` `[0.7562, 0.8021]` (`+0.1218` over `metadata_only_logreg`)
  - `numpy_frozen_random_features`: `0.7343` `[0.7110, 0.7609]`
  - `metadata_only_logreg` (no pixels): `0.6554` `[0.6312, 0.6812]`
- **View (b) — Corpus-Prevalence-Reweighted Clinical Metrics (Ensemble):**
  - `dermoscopic` (corpus prev `9.35%`): `PPV = 0.5457`, `NPV = 0.9957`, `ECE = 0.1836`, `abstention = 0.1443`
  - `clinical` (corpus prev `48.22%`): `PPV = 0.6997`, `NPV = 0.9186`, `ECE = 0.0990`, `abstention = 0.2026`
  - **`tbp_tile` (corpus prev `0.14%`):** **`PPV = 0.0327` (`3.3%`)**, `NPV = 0.9992`, **`ECE = 0.2082`** -> **App abstains automatically on `tbp_tile` inputs** with EN/FR explanation.
  - Overall reweighted (corpus prev `3.28%`): `PPV = 0.5163`, `NPV = 0.9980`, `ECE = 0.1997`, `abstention = 0.0853`.
- **Grad-CAM Sanity (`n = 220` dermoscopic `val_v3` images, `models/derm_cnn_seed2`):** Stated threshold `adebayo_mean_l1_shift >= 0.30`. Measured `adebayo_mean_l1_shift = 0.2270` (**`adebayo_passed = false`**) and `dermoscopic_center_heat_share = 0.2316` (diffuse/partially peripheral on `96×96` inputs; caption: *"Where the model looked — not proof of a diagnosis."*).

### 15.4 Fitzpatrick Skin Tone Fairness & Shortcut Audit (`locked_test_v3`, Ensemble)

| Tone | `n` (`mal` / `ben`) | Ensemble AUROC [95% CI] | Tone+Collection Shortcut AUROC | Flag |
|---|---|---|---|---|
| `I` | `150` (`75` / `75`) | `0.8027` [0.7210, 0.8635] | `0.6462` | `shortcut_flagged (>= 0.60)` |
| `II` | `159` (`84` / `75`) | `0.8016` [0.7261, 0.8685] | `0.5633` | `adequate (>= 30)` |
| `III` | `150` (`75` / `75`) | `0.8471` [0.7887, 0.9038] | `0.7995` | `shortcut_flagged (>= 0.60)` |
| `IV` | `150` (`75` / `75`) | `0.8812` [0.8214, 0.9265] | `0.8828` | `shortcut_flagged (>= 0.60)` |
| `V` | `135` (`60` / `75`) | `0.8898` [0.8062, 0.9479] | `0.9496` | `shortcut_flagged (>= 0.60)` |
| `VI` | `102` (`27` / `75`) | `0.9694` [0.9293, 0.9972] | **`0.9733`** | **`shortcut_flagged (>= 0.60)`** |
| `UNKNOWN` | `778` (`351` / `427`) | `0.9247` [0.9072, 0.9397] | `0.5331` | `adequate (>= 30)` |

- **Critical Shortcut Finding:** Tones `IV–VI` malignant images come overwhelmingly from `fitzpatrick17k` while benign images come mostly from `isic`, allowing a pixel-free `tone + dataset` classifier to reach `0.9733` AUROC on Tone `VI`. Dark-skin AUROC (`0.9694`) therefore reflects dataset-source confounding rather than clinically verified dark-skin generalization.

### 15.5 General Text Rules (W9) & P10 Redaction Recall (W10)
- **W9 (`hybrid_v2_seed1` rebuilt & logged in `TEST_SET_LOG.md`, [w9_text_reeval.json](results/w9_text_reeval.json)):** Replaced probe-specific filters with pre-NER [person_age_detector.py](../medresearch_platform/security/person_age_detector.py) and general `rules out` / `excluded` / `éliminé` / `écarté` / `exclu` ConText triggers. On `SYNTH-MED-V3.1`: `val` strict F1 `0.9937` (`en: 0.9931`, `fr: 0.9944`); `test_iid` `0.9920` (assertion `0.7286` vs `0.7188` prior); `test_unseen_terms` **`0.9662`** (`+0.0458` over `0.9204`); `test_unseen_templates` **`0.9438`** (`+0.0128` over `0.9310`).
- **W10 (`p10_redaction_recall.json`):** Measured on `52` hand-labelled EN+FR sentences (`111` PHI items across DOB, Senegal/FR/intl phones, untitled names, `CLINICIAN_NAME`, `IPP`/`MRN`, and EN/FR street addresses): **Recall = `1.0` (`111/111`), Precision = `1.0` (`0` false positives)**.
| Skin Tone | `UNKNOWN` | `1,363` (`648` / `715`) | `0.9220` [0.9086, 0.9360] | `0.7908` | `0.1486` | `0.1225` | `adequate (>= 30)` |
| Modality | `dermoscopic` | `1,160` (`443` / `717`) | **`0.9783`** [0.9710, 0.9857] | `0.9376` | `0.0481` | `0.0784` | `adequate (>= 30)` |
| Modality | `clinical` | `399` (`333` / `66`) | `0.9029` [0.8522, 0.9340] | `0.8202` | `0.1358` | `0.1629` | `adequate (>= 30)` |
| Modality | `tbp_tile` | `294` (`288` / `6`) | `0.7095` [0.4811, 0.9070] | `0.5642` | `0.6503` | `0.2789` | `adequate (>= 30)` (heavy class skew) |
| Dataset | `isic` | `1,853` (`1,064` / `789`) | `0.9398` [0.9304, 0.9498] | `0.8389` | `0.1554` | `0.1284` | `adequate (>= 30)` |

### 15.5 Key Limitations
1. **Fitzpatrick Tone Class Skew in `locked_test_v2`:** Malignant-first oversampling placed single-class subsets into tones `I–III` (malignant only) and `VI` (benign only), and tone `V` had `n = 27 < 30` after cross-split near-duplicate removal.
2. **Fine-Grained Diagnosis Head:** `PyTorchDermResNet` excels at binary malignancy (`0.9398–0.9461`) while `classical_logreg` leads on 9-class diagnosis (`0.7447` Macro-F1); `tbp_tile` (`288` mal / `6` ben) has high ECE (`0.6503`).
3. **No Prospective Clinical Validation:** See [MODEL_CARD_DERM_CNN_V1.md](model-cards/MODEL_CARD_DERM_CNN_V1.md) and [ADR-0010](../DECISIONS/ADR-0010-pytorch-cpu-for-vision.md).
