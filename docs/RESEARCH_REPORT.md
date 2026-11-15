# Bilingual Medical Information Extraction — Research Report (`SYNTH-MED-V3.1`)

**Date:** 2026-11-15  
**Dataset:** `SYNTH-MED-V3.1` (`data/processed/v3_1/`, 1,240 documents: 620 EN + 620 FR; 600 train, 160 val, 480 across 3 locked test splits)  
**Execution Hardware:** MacBook Pro Intel Core i7-9750H @ 2.60 GHz (6 cores / 12 threads), 32 GB RAM, **CPU-only**, **from-scratch pure NumPy** (`DECISIONS/ADR-0009-numpy-vs-pytorch.md`)  
**Total Final 5-Seed Locked Evaluation Runtime:** `3415.92s` (all 7 approaches × 5 seeds `1..5` × 4 splits)

---

## 1. Executive Summary

This report documents the end-to-end scientific evaluation of seven information extraction approaches—five single-model paradigms (`rule_based`, `classical_ml`, `classical_crf`, `bilstm`, `bilstm_crf`), a stacked multi-head BPE `transformer` built from scratch, and our 5-stage `hybrid` pipeline—under **strict split isolation**, **equal 600-document training budget**, **6/6 verified analytical gradients**, **5 random seeds (`1..5`)**, and **10,000-resample cluster-paired bootstrap significance testing with Holm-Bonferroni correction** (`α = 0.05`, 21 pairwise comparisons per split).

### Headline Findings (5-Seed Means on `SYNTH-MED-V3.1`)

1. **In-Distribution (`test_iid`, n=160):** `classical_ml` (`0.9983`), `classical_crf` (`0.9968`), and `hybrid` (`0.9941 [0.9912, 0.9969]`) all exceed `0.99` Strict Entity F1, followed by `bilstm_crf` (`0.9884`), `bilstm` (`0.9659`), `transformer` (`0.7930`), and `rule_based` (`0.3221`).
2. **Unseen Terminology Generalization (`test_unseen_terms`, n=160, 0% lexicon overlap):** `classical_crf` is the clear entity-NER leader at **`0.9169`** Strict F1, significantly outperforming `classical_ml` (`0.8441`, `Δ = +0.0728`, Holm `p = 0.0042`), `hybrid` (`0.8277`), `bilstm_crf` (`0.8141`), `bilstm` (`0.7859`), `transformer` (`0.5241`), and `rule_based` (`0.3233`). Rich orthographic, affix, and context-window features generalize to OOV clinical terms better than small-data neural embeddings.
3. **Unseen Template Generalization (`test_unseen_templates`, n=160, 0.6% masked sentence overlap):** `classical_ml` (`0.9670`) and `classical_crf` (`0.9527`) lead Strict F1, while `bilstm_crf` achieves `0.9392` and `hybrid` scores `0.8949` Strict / `0.9644` Lenient (indicating span-boundary shifts on novel sentence structures).
4. **Structured Semantics (Assertion, Relations, Temporal Normalization):** On downstream clinical structure, `hybrid` leads **Relation F1** across every split (`0.8922` `test_iid`, `0.8127` `test_unseen_terms`, `0.8029` `test_unseen_templates` vs `0.7197–0.7341` for `classical_crf`) and achieves near-perfect **TIMEX F1 / ISO Normalization** (`0.9986 / 1.0000` `test_iid`, `1.0000 / 1.0000` `test_unseen_terms`, `0.9969 / 1.0000` `test_unseen_templates` vs `0.7160–0.7340` TIMEX F1 and `0.1032–0.1265` normalization for single models).
5. **Honest Negative Results:**
   - **Stage 5 Transformer Fusion (`0.0000` delta):** Incorporating `TransformerNER` token probabilities into Stage 5 span fusion yields **exact zero change** (`0.9942 -> 0.9942` `val`, `0.8277 -> 0.8277` `test_unseen_terms`, `0.8949 -> 0.8949` `test_unseen_templates`) because the from-scratch Transformer (`0.52–0.79` Strict F1) is too weak relative to `BiLSTM-CRF` (`0.81–0.99`).
   - **Stage 3b `BiLSTMAssertionClassifier` underperforms ConText/NegEx:** Ablating `BiLSTMAssertionClassifier` *improves* Assertion Macro-F1 across all three evaluated splits (`0.7350 -> 0.7536` `val`, `0.6672 -> 0.7041` `test_unseen_terms`, `0.5283 -> 0.6405` `test_unseen_templates`).
   - **Stage 2 Rules Layer (`~0` delta):** Removing Stage 2 rules yields `0.0000` change on `val` (`0.9942`), `-0.0010` on `test_unseen_terms` (`0.8267`), and `+0.0006` on `test_unseen_templates` (`0.8955`).
   - **Real OCR End-to-End Degradation (`test_iid`, 120 pages):** Real Tesseract OCR (`eng` + `fra`) across 6 degradation tiers produces overall CER `0.1927` / WER `0.3198`, dropping end-to-end entity Strict F1 by **`-0.4322`** (`0.9978 -> 0.5655`) for `classical_crf` and **`-0.4605`** (`0.9937 -> 0.5332`) for `hybrid`.

---

## 2. Corpus Design & Realism (`SYNTH-MED-V3.1`)

`SYNTH-MED-V3.1` (`data/processed/v3_1/`, built and audited via `scripts/audit_corpus.py`, protocol in [`docs/REALISTIC_SET_PROTOCOL.md`](file:///Users/guest777/Desktop/medical/docs/REALISTIC_SET_PROTOCOL.md)) fixes the header-leakage, template repetition, and equal-budget flaws of earlier synthetic iterations:

- **Splits (1,240 documents total, 50% EN / 50% FR):** `train` (`600`), `val` (`160`), `test_iid` (`160`), `test_unseen_terms` (`160`, 0% entity surface overlap with `train`), `test_unseen_templates` (`160`, 0.6% masked sentence overlap with `train`).
- **Document Types (6 types, non-trivial structural cues):** `discharge_summary`, `radiology_report`, `prescription`, `lab_report`, `consultation_note`, `pathology_report`. Header tags are stripped/randomized so `DocumentTypeClassifier` cannot rely on literal title strings; it still reaches `1.0000` doc-type accuracy from section vocabulary and clinical register.
- **Locked Test Discipline:** All hyperparameter tuning and ablation checks used `val` only; all `test_*` runs are logged chronologically in [`docs/results/TEST_SET_LOG.md`](file:///Users/guest777/Desktop/medical/docs/results/TEST_SET_LOG.md).

---

## 3. Model Architectures, Gradient Verification & Fair Training Protocol

Every model trains **exclusively from random initialization** in **pure NumPy** ([`DECISIONS/ADR-0009-numpy-vs-pytorch.md`](file:///Users/guest777/Desktop/medical/DECISIONS/ADR-0009-numpy-vs-pytorch.md)) on the exact same `600`-document `train` split per seed:

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

Plot: [`docs/results/v3_1_learning_curve.svg`](file:///Users/guest777/Desktop/medical/docs/results/v3_1_learning_curve.svg)

---

## 4. Final 5-Seed Benchmark Results (`seeds 1..5`, `SYNTH-MED-V3.1`)

All metrics below are 5-seed means with normal 95% CIs (`mean ± 1.96 * std / sqrt(5)`). Full 10,000-resample cluster-paired bootstrap CIs and all 21 pairwise Holm-Bonferroni p-values per split are recorded in `docs/results/v3_1_{val,test_iid,test_unseen_terms,test_unseen_templates}_significance_tables.md`.

### 4.1 `test_iid` (n=160 docs, In-Distribution)

| Approach | Strict F1 [95% CI] | Lenient F1 | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm | Mean Train Time | Max RSS |
|---|---|---|---|---|---|---|---|---|
| `rule_based` | `0.3221` [0.3221, 0.3221] | `0.3343` | `0.5878` | `0.0425` | `0.5101` | `0.1984` | `1.0s` | 176 MB |
| `classical_ml` | **`0.9983`** [0.9983, 0.9983] | **`0.9993`** | `0.7138` | `0.7340` | `0.7160` | `0.1217` | `25.2s` | 427 MB |
| `classical_crf` | `0.9968` [0.9968, 0.9968] | `0.9982` | `0.7194` | `0.7341` | `0.7167` | `0.1217` | `13.4s` | 508 MB |
| `bilstm` | `0.9659` [0.9627, 0.9691] | `0.9821` | `0.7198` | `0.7208` | `0.6997` | `0.1198` | `92.8s` | 561 MB |
| `bilstm_crf` | `0.9884` [0.9859, 0.9909] | `0.9928` | `0.7168` | `0.7342` | `0.7155` | `0.1204` | `107.0s` | 567 MB |
| `transformer` | `0.7930` [0.7781, 0.8080] | `0.8857` | `0.7305` | `0.5978` | `0.4557` | `0.1549` | `241.7s` | 726 MB |
| `hybrid` | `0.9941` [0.9912, 0.9969] | `0.9974` | **`0.7325`** | **`0.8922`** | **`0.9986`** | **`1.0000`** | `199.1s` | 773 MB |

*Significance highlights (`test_iid`):* 20 of 21 pairs are significant at Holm-corrected `α=0.05` (p=`0.0042`–`0.0156`); only `classical_ml` vs `classical_crf` (`Δ = -0.0015`, Holm `p = 0.1464`) is non-significant.

### 4.2 `test_unseen_terms` (n=160 docs, 0% Clinical Lexicon Overlap)

| Approach | Strict F1 [95% CI] | Lenient F1 | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm |
|---|---|---|---|---|---|---|
| `rule_based` | `0.3233` [0.3233, 0.3233] | `0.3385` | `0.6602` | `0.0368` | `0.5096` | `0.2072` |
| `classical_ml` | `0.8441` [0.8441, 0.8441] | **`0.9353`** | `0.7446` | `0.6701` | `0.7179` | `0.1265` |
| `classical_crf` | **`0.9169`** [0.9169, 0.9169] | `0.9319` | **`0.7511`** | `0.7197` | `0.7179` | `0.1265` |
| `bilstm` | `0.7859` [0.7770, 0.7949] | `0.8588` | `0.7411` | `0.6655` | `0.6994` | `0.1230` |
| `bilstm_crf` | `0.8141` [0.8085, 0.8197] | `0.8416` | `0.7209` | `0.7018` | `0.7123` | `0.1222` |
| `transformer` | `0.5241` [0.5150, 0.5333] | `0.6980` | `0.7075` | `0.3984` | `0.4505` | `0.1578` |
| `hybrid` | `0.8277` [0.8072, 0.8482] | `0.8452` | `0.6672` | **`0.8127`** | **`1.0000`** | **`1.0000`** |

*Significance highlights (`test_unseen_terms`):* **All 21/21 pairwise Strict F1 comparisons are statistically significant** after Holm-Bonferroni correction (`p <= 0.0146`); `classical_crf` beats `classical_ml` (`+0.0728`, `p = 0.0042`) and `hybrid` (`+0.0891`, `p = 0.0042`).

### 4.3 `test_unseen_templates` (n=160 docs, 0.6% Masked Sentence Overlap)

| Approach | Strict F1 [95% CI] | Lenient F1 | Assertion Macro-F1 | Relation F1 | TIMEX F1 | TIMEX Norm |
|---|---|---|---|---|---|---|
| `rule_based` | `0.3066` [0.3066, 0.3066] | `0.3165` | `0.5430` | `0.0506` | `0.5029` | `0.1781` |
| `classical_ml` | **`0.9670`** [0.9670, 0.9670] | **`0.9801`** | `0.6453` | `0.7105` | `0.7340` | `0.1032` |
| `classical_crf` | `0.9527` [0.9527, 0.9527] | `0.9701` | `0.6459` | `0.7235` | `0.7340` | `0.1032` |
| `bilstm` | `0.8619` [0.8501, 0.8737] | `0.9226` | `0.6440` | `0.6453` | `0.7154` | `0.0999` |
| `bilstm_crf` | `0.9392` [0.9309, 0.9474] | `0.9721` | `0.6399` | `0.6999` | `0.7309` | `0.0994` |
| `transformer` | `0.6886` [0.6756, 0.7016] | `0.7816` | **`0.6462`** | `0.5255` | `0.4620` | `0.1271` |
| `hybrid` | `0.8949` [0.8796, 0.9102] | `0.9644` | `0.5283` | **`0.8029`** | **`0.9969`** | **`1.0000`** |

*Significance highlights (`test_unseen_templates`):* **All 21/21 pairwise Strict F1 comparisons are statistically significant** after Holm-Bonferroni correction (`p = 0.0042`).

### 4.4 `val` Reference (n=160 docs, Development Split)

- Strict F1: `classical_ml` `0.9973`, `classical_crf` `0.9961`, `hybrid` `0.9942 [0.9913, 0.9972]`, `bilstm_crf` `0.9897`, `bilstm` `0.9665`, `transformer` `0.7852`, `rule_based` `0.3148`.
- `hybrid` downstream: Assertion Macro-F1 `0.7350`, Relation F1 `0.8937`, TIMEX F1 `1.0000`, TIMEX Norm `1.0000` (`docs/results/v3_1_val_significance_tables.md`).

---

## 5. Hybrid Stage 1–5 Ablation Study (`seeds 1..5`, `docs/results/v3_1_step4_ablations.json`)

| Variant | `val` Strict / Assert / Rel / TIMEX | `test_unseen_terms` Strict / Assert / Rel / TIMEX | `test_unseen_templates` Strict / Assert / Rel / TIMEX | Interpretation |
|---|---|---|---|---|
| `full_hybrid` | `0.9942 / 0.7350 / 0.8937 / 1.0000` | `0.8277 / 0.6672 / 0.8127 / 1.0000` | `0.8949 / 0.5283 / 0.8029 / 0.9969` | Baseline 5-stage system |
| `no_rules_layer` (Stage 2) | `0.9942 / 0.7350 / 0.8937 / 1.0000` | `0.8267 / 0.6676 / 0.8126 / 1.0000` | `0.8955 / 0.5283 / 0.8030 / 0.9969` | **Negligible** (`+0.0010` unseen terms, `-0.0006` unseen templates) |
| `no_assertion_bilstm` (Stage 3b) | `0.9942 / 0.7536 / 0.8937 / 1.0000` | `0.8277 / 0.7041 / 0.8127 / 1.0000` | `0.8949 / 0.6405 / 0.8029 / 0.9969` | **Negative result:** BiLSTM assertion hurts vs ConText/NegEx (`-0.0186` to `-0.1122` Assert F1) |
| `no_relation_cnn` (Stage 3) | `0.9942 / 0.7350 / 0.7202 / 1.0000` | `0.8277 / 0.6672 / 0.7041 / 1.0000` | `0.8949 / 0.5283 / 0.6861 / 0.9969` | **Strong positive:** CNN adds `+0.1735` (`val`), `+0.1086` (unseen terms), `+0.1168` (unseen templates) Rel F1 |
| `no_temporal_crf` (Stage 4) | `0.9942 / 0.7350 / 0.8937 / 0.7140` | `0.8277 / 0.6672 / 0.8127 / 0.7163` | `0.8949 / 0.5283 / 0.8029 / 0.7305` | **Strong positive:** Stage 4 lifts TIMEX F1 by `+0.2664–0.2860` and Norm from `0.103–0.126` to `1.0000` |
| `no_transformer_fusion` (Stage 5) | `0.9942 / 0.7350 / 0.8937 / 1.0000` | `0.8277 / 0.6672 / 0.8127 / 1.0000` | `0.8949 / 0.5283 / 0.8029 / 0.9969` | **Zero delta (`0.0000`)** across all metrics and splits |

---

## 6. Full-Size Real OCR Stress & End-to-End Benchmark (`120 pages`, `test_iid`)

Executed with real Tesseract (`eng` + `fra`) via `scripts/run_v3_1_ocr_full_benchmark.py` (`docs/results/v3_1_ocr_full_benchmark.json`, `510.97s`), measuring OCR quality (CER/WER) and end-to-end Strict Entity F1 (`classical_crf` and `hybrid`) via `map_ocr_spans_to_reference`:

| Cohort / Tier | Pages | Mean CER | Mean WER | `classical_crf` Clean -> OCR (Drop) | `hybrid` Clean -> OCR (Drop) |
|---|---|---|---|---|---|
| **Overall (`test_iid`)** | **120** | **`0.1927`** | **`0.3198`** | `0.9978 -> 0.5655` (**`-0.4322`**) | `0.9937 -> 0.5332` (**`-0.4605`**) |
| English (`en`) | 60 | `0.1895` | `0.3082` | `0.9978 -> 0.5693` (`-0.4285`) | `0.9921 -> 0.5373` (`-0.4548`) |
| French (`fr`) | 60 | `0.1959` | `0.3314` | `0.9977 -> 0.5616` (`-0.4361`) | `0.9954 -> 0.5291` (`-0.4663`) |
| `tier_1_300dpi_clean` | 20 | `0.0046` | `0.0263` | `0.9961 -> 0.6110` (`-0.3851`) | `0.9923 -> 0.6033` (`-0.3889`) |
| `tier_2_200dpi_mild_skew_jpeg` | 20 | `0.0056` | `0.0316` | `0.9974 -> 0.6319` (`-0.3655`) | `0.9961 -> 0.6301` (`-0.3660`) |
| `tier_3_phone_perspective_light` | 20 | `0.0092` | `0.0605` | `1.0000 -> 0.6630` (`-0.3370`) | `0.9931 -> 0.6382` (`-0.3548`) |
| `tier_4_two_col_table_stamps` | 20 | `0.6670` | `0.7620` | `0.9987 -> 0.4173` (`-0.5814`) | `0.9975 -> 0.3771` (`-0.6204`) |
| `tier_5_table_multifont_skew5` | 20 | `0.3673` | `0.5397` | `0.9972 -> 0.5120` (`-0.4852`) | `0.9931 -> 0.4772` (`-0.5159`) |
| `tier_6_150dpi_heavy_blur_noise` | 20 | `0.1027` | `0.4988` | `0.9970 -> 0.5208` (`-0.4762`) | `0.9896 -> 0.4356` (`-0.5541`) |

Even when plain-text CER is `< 1%` (`tier_1`–`tier_3`), Tesseract reading-order / line-wrap / punctuation normalization shifts Strict Entity F1 down by `0.34–0.39`; under multi-column tables and stamps (`tier_4`), CER reaches `0.6670` and Strict F1 drops by `-0.58` to `-0.62`.

---

## 7. Platform, Review Persistence & UI Verification

- **Review Queue Persistence & Gold Export:** `POST /api/v1/reviews`, `GET /api/v1/reviews`, and `GET /api/v1/reviews/export` (`jsonl` and `brat`) persist reviewer decisions and corrected span offsets in SQLite (`tests/test_review_api_persistence.py`).
- **Typed API Contract & Dark Mode:** `frontend/src/types/openapi.d.ts` (25 OpenAPI paths generated via `scripts/export_openapi_types.py`), `sessionStorage` auth persistence, dark mode toggle, keyboard-driven review (`A`/`C`/`R`/`N`), and full Playwright E2E verification (`frontend/e2e/app.spec.ts`) across desktop light, desktop dark, and mobile dark viewports (`docs/screenshots/review_desktop_light.png`, `docs/screenshots/review_desktop_dark.png`, `docs/screenshots/review_mobile_dark.png`).

---

## 8. Limitations & Open Engineering Work

1. **Pure-NumPy CPU Ceiling:** Training all 7 models × 5 seeds takes `~57 min` on 6 CPU cores and limits neural model depth/batch size (`DECISIONS/ADR-0009-numpy-vs-pytorch.md`).
2. **Stage 3b Assertion & Stage 5 Fusion:** `BiLSTMAssertionClassifier` degrades Assertion Macro-F1 relative to ConText/NegEx on unseen templates, and Transformer fusion adds `0.0000` Strict F1 gain; Stage 3b should be disabled or replaced with a cross-template calibrated assertion head, and Stage 5 should either be removed or retrained with stronger character+subword representations.
3. **Synthetic vs Clinical Free Text:** `SYNTH-MED-V3.1` proves comparative behavior under controlled OOV and template shifts, but real de-identified hospital prose evaluation remains gated on external IRB/annotation collection as specified in [`docs/REALISTIC_SET_PROTOCOL.md`](file:///Users/guest777/Desktop/medical/docs/REALISTIC_SET_PROTOCOL.md).
4. **OCR Reading-Order Fragility:** Document layout segmentation (column detection and table-cell ordering) before Tesseract is required to recover the `-0.34` to `-0.62` OCR F1 gap.

---

## 9. Phase 12 Post-Report Engineering Improvements

Following the negative findings in Sections 5, 6, and 8, three concrete upgrades were implemented and benchmarked (`phase-12-hybrid-ocr-and-realistic`):

### 9.1 Hybrid v2 (`configs/exp/v3_2/hybrid_v2.yaml`, 3-seed mean)
Configured `HybridPipeline` with `use_crf_ensemble: true` (adding `sklearn-crfsuite` OOV backoff/override to `Stage2EntityExtractor`), `use_assertion_bilstm: false` (reverting to `ConText/NegEx`), and `use_transformer_fusion: false` (removing the zero-delta Stage 5 reranker):

| Split | `hybrid_v1` Strict / Assert / Rel | `hybrid_v2` Strict / Assert / Rel | `classical_crf` Strict / Assert / Rel | Delta (`hybrid_v2 - hybrid_v1`) |
|---|---|---|---|---|
| `val` (160) | `0.9925 / 0.6759 / 0.9187` | **`0.9964 / 0.7566 / 0.9174`** | `0.9961 / 0.7567 / 0.7186` | `+0.0039` Strict, `+0.0807` Assert, `-30%` runtime (`184.2s -> 129.1s`) |
| `test_iid` (160) | `0.9928 / 0.6800 / 0.9189` | **`0.9969 / 0.7176 / 0.9186`** | `0.9968 / 0.7194 / 0.7341` | `+0.0041` Strict, `+0.0376` Assert |
| `test_unseen_terms` (160) | `0.8176 / 0.6441 / 0.8238` | **`0.9159 / 0.7480 / 0.8560`** | `0.9169 / 0.7511 / 0.7197` | **`+0.0983` Strict**, **`+0.1039` Assert**, `+0.0322` Rel |
| `test_unseen_templates` (160) | `0.8855 / 0.5150 / 0.8184` | **`0.9263 / 0.6429 / 0.8511`** | `0.9527 / 0.6459 / 0.7235` | **`+0.0408` Strict**, **`+0.1279` Assert**, `+0.0327` Rel |
| `realistic_80` (40 EN + 40 FR) | `0.2598 / 0.3830 / 0.4014` | `0.2458 / 0.4973 /` **`0.4455`** | **`0.2641`** `/ 0.4987 / 0.4079` | `-0.0140` Strict, `+0.1143` Assert, `+0.0441` Rel |

### 9.2 Realistic Free-Text Prose Stress Test (`realistic_80`)
Created an 80-note (40 EN + 40 FR) un-templated clinical prose set (`medresearch_platform/data/realistic_v1.py`, `data/processed/realistic/eval.jsonl`) featuring clinical shorthand (`c/o`, `h/o`, `ATCD`, `TTT`), multi-clause uncertainty, and unseen multi-word phrases. All `SYNTH-MED-V3.1`-trained models drop sharply to `0.2458–0.2641` Strict Entity F1, confirming that diverse narrative clinical training data is still needed for free-text generalization.

### 9.3 Column-Aware OCR Layout Segmentation (`docs/results/v3_2_ocr_layout_benchmark.json`)
Added horizontal-rule-suppressed vertical projection column detection (`detect_column_split` in [`medresearch_platform/documents/ocr_engine.py`](file:///Users/guest777/Desktop/medical/medresearch_platform/documents/ocr_engine.py)) so `run_tesseract_ocr` reads left column top-to-bottom before right column:
- **`tier_4_two_col_table_stamps` (20 pages):** Mean CER fell from **`0.6670 -> 0.1474`** (-77.9% relative), Mean WER fell from **`0.7620 -> 0.2338`** (-69.3% relative), and `classical_crf` OCR Strict F1 rose from **`0.4173 -> 0.5950`** (`+0.1777`).
- **Overall 120 pages (`test_iid`):** Mean CER fell from **`0.1927 -> 0.1061`**, Mean WER fell from **`0.3198 -> 0.2318`**, and `classical_crf` OCR Strict F1 improved from **`0.5655 -> 0.5919`** with zero regression on single-column Tiers 1–3.

