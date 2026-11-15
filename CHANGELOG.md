# Changelog

## [Unreleased] — Phase 11 ("Make the results true")

### Phase 11B Step 9 — Final 5-Seed Locked Evaluation & Honest Research Report (2026-11-15)
- Executed `scripts/run_v3_1_final_locked_eval.py` across all 7 `SYNTH-MED-V3.1` approaches × 5 seeds (`1..5`) on `val` (160), `test_iid` (160), `test_unseen_terms` (160), and `test_unseen_templates` (160) in `3415.92s`, generating `docs/results/v3_1_final_locked_eval.json` and 10,000-resample Holm-Bonferroni significance tables (`docs/results/v3_1_*_significance{,_tables}.{json,md}`), logged in `docs/results/TEST_SET_LOG.md`.
- Rewrote `docs/RESEARCH_REPORT.md` and updated `PROJECT_STATUS.md` with exact 5-seed metrics, 95% CIs, Holm-Bonferroni p-values, Stage 1–5 ablations (including negative results for Stage 3b BiLSTM assertion and Stage 5 fusion), 120-page real Tesseract OCR F1 drops, and frontend/E2E verification.

### Phase 11B Step 8 — Realistic Set Protocol & ADR-0009 (G11) (2026-11-15)
- Added `docs/REALISTIC_SET_PROTOCOL.md` documenting `SYNTH-MED-V3.1` split separation (`val`, `test_iid`, `test_unseen_terms`, `test_unseen_templates`), document-type structural signals, mandatory `scripts/audit_corpus.py` gates, 120-page OCR stress protocol, and external/real clinical corpus governance.
- Added `DECISIONS/ADR-0009-numpy-vs-pytorch.md` reconciling `ADR-0003` and `ADR-0005`, documenting the single pure-NumPy implementation, `< 1e-4` finite-difference gradient verification across all 6 neural classes, CPU throughput trade-offs, and PyTorch migration criteria.

### Phase 11B Step 7 — Frontend to the Phase 7 Standard (G10) (2026-11-15)
- Added persisted review endpoints in `medresearch_platform/interfaces/api.py` and `medresearch_platform/workflows/review.py`: `POST /api/v1/reviews` (`approve`/`correct`/`flag`/`reject` with span/type/assertion updates, reviewer, timestamp, old/new audit log), `GET /api/v1/reviews`, `GET /api/v1/reviews/export` (`jsonl` and `brat` gold export), plus `/react/{full_path:path}` SPA fallback. Added `tests/test_review_api_persistence.py`.
- Generated TypeScript API types directly from FastAPI `/api/v1/openapi.json` via `scripts/export_openapi_types.py` (`openapi-typescript` -> `frontend/src/types/openapi.d.ts`, `npm run gen:api`).
- Added Dark Mode (`frontend/src/store/themeStore.ts`, `@custom-variant dark` in `index.css`, toggle in `Layout.tsx`, `dark:` styling on `Card.tsx` and `ReviewQueuePage.tsx`) and `sessionStorage` auth persistence in `authStore.ts`.
- Upgraded `frontend/src/pages/ReviewQueuePage.tsx` with `POST /api/v1/reviews` persistence, progress bar, selectable document text for span offsets, inline correction form, `A`/`C`/`R`/`N` keyboard shortcuts, JSONL/BRAT export buttons, and saved audit log.
- Added Playwright E2E suite (`frontend/playwright.config.ts`, `frontend/e2e/app.spec.ts`) running against real FastAPI + built React UI (`/react/`), capturing desktop light, desktop dark, and mobile dark screenshots in `docs/screenshots/`.

### Phase 11B Step 6 — Full-Size OCR Benchmark & Clean Skip (G9) (2026-11-15)
- Extended `medresearch_platform/documents/ocr_benchmark.py` with `tesseract_available(("eng", "fra"))`, full degradation rendering (`dpi` 150–300, JPEG, blur, noise, rotation `±5°`, perspective keystone warp, uneven lighting, stamps/stains, two columns, table grid, >= 3 fonts), and `map_ocr_spans_to_reference` for end-to-end OCR span evaluation.
- Updated `tests/test_ocr_stress_benchmark.py` (4 tests) to `pytest.skip` cleanly when Tesseract or `fra` is missing, and added macOS/Windows Tesseract + `fra` setup instructions to `README.md`.
- Ran 120-page (`60 EN + 60 FR`) OCR + end-to-end extraction benchmark via `scripts/run_v3_1_ocr_full_benchmark.py` (`docs/results/v3_1_ocr_full_benchmark.json`, `510.97s`, logged in `TEST_SET_LOG.md`): overall CER `0.1927` (`en` `0.1895`, `fr` `0.1959`), WER `0.3198` (`en` `0.3082`, `fr` `0.3314`), `classical_crf` F1 `0.9978 -> 0.5655` (`-0.4322`), `hybrid` F1 `0.9937 -> 0.5332` (`-0.4605`).

### Phase 11B Step 5 — 5-Seed Statistics, 10k Paired Bootstrap & Holm-Bonferroni (G7) (2026-11-15)
- Upgraded `bootstrap_f1_ci` and `paired_significance_test` in `medresearch_platform/eval_harness/metrics.py` (and `runner.py`) to vectorized `10,000` resamples by default; added `per_doc_strict_counts`, `holm_bonferroni_adjust`, and `pooled_paired_bootstrap` (cluster-paired document bootstrap pooled across seeds).
- Updated all 7 `configs/exp/v3_1/*.yaml` configs to `seeds: [1, 2, 3, 4, 5]`.
- Added `tests/test_statistics_and_holm_bonferroni.py` (4 unit tests verifying `n_resamples >= 10000`, exact Holm-Bonferroni monotonicity, no 250-resample floor, and 5-seed pooled paired bootstrap).
- Ran all 7 approaches × 5 seeds on `SYNTH-MED-V3.1` `val` via `scripts/run_v3_1_val_5seed.py` (`docs/results/v3_1_val_5seed_benchmark.json`, `3538.37s`) and generated `docs/results/v3_1_val_significance.json` and `docs/results/v3_1_val_significance_tables.md` via `scripts/generate_significance_tables.py` (21 pairwise comparisons, 10,000 resamples, Holm-Bonferroni corrected, 95% CIs, method footnotes under each table).

### Phase 11B Step 4 — Make Every Hybrid Stage Earn Its Place (G4, G8) (2026-11-14)
- Upgraded `normalize_timex` in `medresearch_platform/hybrid/stage4_temporal.py` to handle all EN/FR `DATE` (`YYYY-MM-DD`, `DD/MM/YYYY`, `Month DD, YYYY`, `DD mois YYYY`, `Month DDth`, `DD mois`, `Month YYYY`), `RELATIVE` (`yesterday`/`la veille`, `today`/`aujourd'hui`, `POD N`/`J N post-op`, `in/dans N days/weeks/months`, `N ... ago`/`il y a`/`depuis`), `DURATION` (`P{n}D/W/M/Y`), and `FREQUENCY` (`QD`/`BID`/`TID`/`PRN`) expressions anchored to `doc.document_date`; updated `CRFTemporalTagger.fit` to train on all 600 EN+FR train docs instead of `train_docs[:250]` (EN-only).
- Expanded EN/FR `CONTEXT_CUES` and enabled `post_window` checking for `possible` cues in `medresearch_platform/approaches/context_negex.py`; fixed `BiLSTMAssertionClassifier.fit` in `medresearch_platform/hybrid/stage3_context_relations.py` to shuffle `train_docs` before `present` capping (so FR `present` examples are included) and strengthened minority class weights.
- Added `tests/test_hybrid_stage4_and_assertion_fixes.py` (35 parametrized regression tests; 28 failed before fixes, 35/35 pass after).
- Ran 5-seed (`1..5`) ablations via `scripts/run_v3_1_step4_ablations.py` on `SYNTH-MED-V3.1` `val`, `test_unseen_terms`, and `test_unseen_templates` (`docs/results/v3_1_step4_ablations.json`, logged in `docs/results/TEST_SET_LOG.md`):
  - Temporal CRF (`stage4`): TIMEX F1 `1.0000` (`val` & `unseen_terms`) / `0.9969` (`unseen_templates`) and TIMEX norm accuracy `1.0000` across all 3 splits (+0.8738 to +0.8969 over `no_temporal_crf`).
  - Relation CNN (`stage3`): `+0.1735` (`val`), `+0.1086` (`unseen_terms`), `+0.1168` (`unseen_templates`) relation F1.
  - Assertion (`stage3`): `full_hybrid` assertion macro-F1 `0.7350` (`val`), `0.6672` (`unseen_terms`), `0.5283` (`unseen_templates`); upgraded ConText/NegEx alone (`no_assertion_bilstm`) reaches `0.7536` / `0.7041` / `0.6405`.
  - Rules layer (`stage2`): `0.0000` delta on `val`, `+0.0010` on `unseen_terms`, `-0.0006` on `unseen_templates`.
  - Transformer fusion (`stage5`): `0.0000` delta across all 3 splits (documented negative result).

### Step 0 — Audit & Honesty Reset (2026-11-14)
- Added `scripts/audit_corpus.py` to detect doubled span text, adjacent duplicate n-grams, offset mismatches, lexical overlap, and digit-masked sentence/template overlap.
- Audited `SYNTH-MED-V2` (`data/processed/v1`): confirmed 100% of docs had doubled TIMEX/date/duration spans (F1) and `94.8%` masked test sentence / `62.7%` unique entity overlap (`100%` across clinical entity types) (F2).
- Archived v2 locked test results and audit output in `docs/results/archive/2026-11-14_synth_v2_buggy/`.
- Marked `docs/RESEARCH_REPORT.md` as Superseded and reset `PROJECT_STATUS.md` to `Phase 11 — correction`.

### Step 1 — Fix Doubled TIMEX Generator Bug (F1) (2026-11-14)
- Added `_DocBuilder.add_entity_with_timex` in `medresearch_platform/data/synthetic_v2.py` and replaced all 5 `add_entity` + `add_timex` pairs so temporal mentions are written once with identical `[start, end)` offsets for both `EntitySpan` and `TimexSpan`.
- Documented dual Entity + TIMEX scoring rules in `docs/ANNOTATION_GUIDELINES.md` and verified `Evaluator.evaluate` scores both layers independently on shared offsets.
- Updated `scripts/audit_corpus.py` tokenization so ISO dates (`YYYY-MM-DD`) and `PAT-XX-NNN` IDs remain atomic tokens (avoiding false-positive `- DD` duplicates when month == day).
- Added regression test `tests/test_synthetic_generator_integrity.py` (400 docs: 200 EN + 200 FR) checking zero doubled spans, zero duplicate n-grams, 100% offset match, normalized TIMEX values, and token-boundary alignment.
- Built `SYNTH-MED-V3` (`configs/data/v3.yaml`, `data/processed/v3`) and verified 0 doubled docs; re-ran `rule_based` (`val`: strict F1 `0.9623`, lenient F1 `0.9623`, TIMEX F1 `1.0000`, TIMEX norm acc `0.2500`) and `bilstm_crf` (`val`: strict F1 `1.0000`, lenient F1 `1.0000`, TIMEX F1 `1.0000`, TIMEX norm acc `0.2500`).


### Step 2 — Generalization Corpus `SYNTH-MED-V3` (F2) (2026-11-14)
- Authored bilingual clinical lexicon (`medresearch_platform/data/lexicon_v3.py`): 85 diagnoses, 62 symptoms, 82 medications, 52 lab tests, 41 procedures, and 31 anatomy concepts per language (EN + FR) with abbreviations, synonyms, surface variants, and ICD-10/ATC/LOINC/SNOMED codes, partitioned 70/30 at the concept level.
- Authored 640 independent EN/FR clinical sentence patterns (`medresearch_platform/data/templates_v3.py`: 40 per section × 8 sections × 2 languages) covering varied section orderings, missing sections, bullet lists, lab tables, harder assertion scope (negation termination, double negation, rule-out, uncertainty, hypothetical, family/historical), and multi-format/relative/partial TIMEX normalized to ISO-8601, partitioned 70/30 per `(language, section)`.
- Implemented `medresearch_platform/data/synthetic_v3.py` and wired `generator: v3` in `medresearch_platform/data/builder.py` and `configs/data/v3.yaml` to build 5 locked splits (`train` 600, `val` 160, `test_iid` 160, `test_unseen_terms` 160, `test_unseen_templates` 160 = 1,240 docs).
- Verified via `scripts/audit_corpus.py`: `0` doubled docs, `0` duplicate n-gram docs, `100.00%` offset/relation integrity, `0.0%` unseen clinical term overlap in `test_unseen_terms`, and `0.6%` masked sentence overlap in `test_unseen_templates` (`12.0%` in `test_iid`).
- Added `tests/test_v3_corpus.py` verifying lexicon/template minimums, integrity, token alignment, and split isolation.

### Step 3 — Seed Sensitivity (F3), Char Morphology & Eval Split Upgrades (2026-11-14)
- Upgraded `_ortho_features` in `medresearch_platform/approaches/neural_seq.py` (`CHAR_FEAT_DIM = 32`) with orthographic/shape flags, 8 prefix n-gram hash buckets, and 16 suffix n-gram hash buckets to support unseen medical morphology.
- Added seeded word-dropout (`<UNK>`) and inverted char-feature dropout in `_train_and_predict_neural` driven by `np.random.default_rng(seed)` (`dropout`, default `0.15`), ensuring seeds reach weight initialization, epoch shuffling, and regularization.
- Extended `medresearch_platform/eval_harness/runner.py` (`VALID_EVAL_SPLITS`) to support `test_iid`, `test_unseen_terms`, and `test_unseen_templates` alongside `val` and `test`.
- Added `tests/test_seed_sensitivity_and_eval_splits.py` verifying v3 eval split support, char morphology similarity on unseen `-statin` terms, and distinct predictions across seeds.
- Verified 3-seed `bilstm_crf` on `SYNTH-MED-V3` `val` (160 docs): `Strict F1 = 0.9604 ± 0.0017` (non-zero cross-seed std, realistic error profile).

### Step 4 — Hybrid Stage 3 (F4) & Text-Reading Stage 5 Fusion (F5) (2026-11-14)
- Fixed `medresearch_platform/data/synthetic_v3.py` assertion labels (`absent`, `possible`) and relation types (`REVEALS`, `LOCATED_IN`) to strictly match `ASSERTION_CLASSES`, `TEMPORALITY_CLASSES`, and `RELATION_TYPES` in `schema.py`; rebuilt `SYNTH-MED-V3` (`data/processed/v3`).
- Fixed Stage 3 (`medresearch_platform/hybrid/stage3_context_relations.py`):
  - `BiLSTMAssertionClassifier.fit` now balances all non-`present` examples (preventing rare-class starvation) and exposes `predict_proba` fused with ConText fallback in `pipeline.py` (using BiLSTM when `max_prob >= 0.75`, ConText otherwise).
  - `PiecewiseCNNRelationClassifier` now covers all schema relation pairs (`_PAIR_TO_REL_TYPES`) and masks impossible cross-type relation logits during inference.
- Fixed Stage 5 (`medresearch_platform/hybrid/stage5_integration.py` + `pipeline.py`): `TransformerCandidateReranker` now reads document text via learned token embeddings (`E` with full backprop) over candidate span tokens, 3-token left/right context windows, and 32-dim char morphology (`99` input features per candidate instead of `19` scalar-only features).
- Added `tests/test_hybrid_stage3_stage5_fixes.py` verifying Stage 3 relation pair coverage, Stage 5 context-dependent scoring, and end-to-end `HybridPipeline` execution on `SYNTH-MED-V3`.

### Step 5 — Realistic OCR Stress Benchmark (F6) (2026-11-14)
- Extended `render_noisy_scan_image` and `parse_and_benchmark_scan` in `medresearch_platform/documents/ocr_benchmark.py` with line wrapping, Gaussian blur, JPEG compression artifacts, and contrast degradation.
- Added `scripts/run_ocr_benchmark.py` evaluating 16 `SYNTH-MED-V3` `val` documents (8 EN + 8 FR, avg `91.6` words/doc) across 3 degradation tiers (`clean_scan`, `mild_scan`, `heavy_scan` = 48 scanned PDFs), saving `docs/results/ocr_stress_benchmark.json`:
  - `clean_scan`: CER `0.0021`, WER `0.0138`, conf `0.9388` (EN CER `0.0011`, FR CER `0.0031`)
  - `mild_scan`: CER `0.0022`, WER `0.0119`, conf `0.9384` (EN CER `0.0006`, FR CER `0.0038`)
  - `heavy_scan`: CER `0.1615`, WER `0.5471`, conf `0.7343` (EN CER `0.1624`, FR CER `0.1607`)
- Added `tests/test_ocr_stress_benchmark.py` verifying non-zero CER degradation under heavy scan noise/blur/JPEG compression.

### Step 6 — Fresh-Clone CI & `test_frontend_assets_and_security_headers` Fix (F8) (2026-11-14)
- Updated `tests/test_api_and_cli.py` (`test_frontend_assets_and_security_headers`) to handle fresh-clone environments where `frontend/dist` has not been built yet (`404` JSON message) while still verifying `200`, CSP headers, and `/react/assets/` when `frontend/dist/index.html` is present.
- Updated `.github/workflows/ci.yml`: added `actions/setup-node@v4` (Node 22), added `scripts/` to `ruff check`, and added `medextract data build --config configs/data/v3.yaml && python scripts/audit_corpus.py --data-dir data/processed/v3 --fail-on-doubled` to CI.

### Step 7 — Complete React Frontend (F7) (2026-11-14)
- Completed all 14 React pages in `frontend/src/pages/` and wired routes + navigation in `App.tsx` and `Layout.tsx`:
  - Existing: `LoginPage`, `WorkspacePage`, `UploadJobsPage`, `BenchmarkPage` (upgraded with Recharts bar chart), `TimelineAuditPage`.
  - Added 9 new pages: `RelationGraphPage`, `ReviewQueuePage`, `PdfViewerPage` (using `pdfjs-dist`), `AblationPage` (Recharts), `ErrorTaxonomyPage` (Recharts), `CorpusExplorerPage` (Recharts), `ReportsPage`, `SchemaExplorerPage`, `SettingsDiagnosticsPage`.
- Added `recharts`, `pdfjs-dist`, `vitest`, `@testing-library/react`, `@testing-library/jest-dom`, and `jsdom` to `frontend/package.json`.
- Added `frontend/vitest.config.ts` and `frontend/src/__tests__/pages.test.tsx` (7/7 Vitest component tests passing).
- Added `npm --prefix frontend test` to `.github/workflows/ci.yml`.

### Step 8 — Multi-Seed `SYNTH-MED-V3` Validation, Ablation & Locked Test Evaluation (2026-11-14)
- Added committed `SYNTH-MED-V3` experiment configs in `configs/exp/v3/` (`rule_based.yaml`, `classical_ml.yaml`, `classical_crf.yaml`, `bilstm.yaml`, `bilstm_crf.yaml`, `transformer.yaml`, `hybrid.yaml`) with `seeds: [1, 2, 3]`.
- Added `scripts/run_v3_val_and_ablation.py` and executed multi-seed `val` benchmark (`docs/results/v3_val_benchmark.json`) and 5-variant hybrid ablation (`docs/results/v3_val_ablation.json`), confirming Stage 3 Assertion BiLSTM (`0.5500` vs `0.4851` without) and Stage 3 Relation CNN (`0.9050` vs `0.7056` without) now improve performance.
- Added `scripts/run_v3_locked_test_eval.py` and executed the one-time locked test evaluation across `test_iid`, `test_unseen_terms`, and `test_unseen_templates` (160 docs each, seeds 1–3) with paired significance tests, saving `docs/results/v3_locked_test_evaluation.json` and logging the run in `docs/results/TEST_SET_LOG.md`.

### Step 9 — Honest Research Report & Final Phase 11 Status (2026-11-14)
- Rewrote `docs/RESEARCH_REPORT.md` from committed `SYNTH-MED-V3` JSON outputs (`v3_locked_test_evaluation.json`, `v3_val_ablation.json`, `ocr_stress_benchmark.json`), reporting all Strict/Lenient NER F1, EN/FR F1, Assertion Macro-F1, Relation F1, TIMEX F1/normalization, paired significance, OCR stress metrics, and honest limitations.
- Finalized `PROJECT_STATUS.md` with Phase 11 complete.

## [Unreleased] — Phase 11B ("Finish Phase 11 properly")

### 11B Step 1 — Fix Document-Type Labels (G1) & Remove Gold `doc_type` Copying (G2) (2026-11-14)
- Added `tests/test_hidden_doc_type_and_learnability.py` first: verified `dummy`, `rule_based`, `bilstm`, `bilstm_crf`, and `transformer` leaked `__hidden__` before fixing.
- Added `SYNTH-MED-V3.1` (`configs/data/v3_1.yaml`, `version="v3_1"` in `medresearch_platform/data/synthetic_v3.py`, wired in `builder.py`): each of the 6 `DOC_TYPES` now has its own EN/FR header pool (with 12% ambiguous headers), section set, section order, and section labels. Audited `data/processed/v3_1`: `0` doubled docs, `100.00%` offset/relation integrity, `0.0%` unseen clinical term overlap on `test_unseen_terms`, `6.4%` masked sentence overlap on `test_unseen_templates`.
- Removed `predicted_doc_type=d.doc_type` from `medresearch_platform/eval_harness/runner.py` (`_dummy_predictor` returns `None`), `medresearch_platform/approaches/rule_based.py` (added `_predict_doc_type_rules`), and `medresearch_platform/approaches/neural_seq.py` (added train-only TF-IDF + `LinearSVC` document classifier).
- Verified all 9 tests in `tests/test_hidden_doc_type_and_learnability.py` pass (`21.52s`) and all 7 approaches achieve `1.0000` doc-type accuracy on `SYNTH-MED-V3.1` `val` (160 docs).

### 11B Step 2 — Equal 600-Doc Training Budget & Learning Curves (G3) (2026-11-14)
- Updated `medresearch_platform/approaches/neural_seq.py`, `medresearch_platform/hybrid/stage2_ner.py`, `medresearch_platform/hybrid/pipeline.py`, and `medresearch_platform/approaches/classical_ml.py` (`_select_train_docs`) so `max_train_docs: null` uses all 600 training documents uniformly across every approach (and shuffled slices in `HybridPipeline` see balanced EN+FR).
- Added equal-budget configs in `configs/exp/v3_1/` (`max_train_docs: null`, `data_dir: data/processed/v3_1`) and regression tests in `tests/test_equal_budget_configs.py`.
- Added `scripts/run_v3_1_learning_curve.py` and ran `100 / 200 / 400 / 600` training-doc learning curves on `SYNTH-MED-V3.1` `val` (160 docs, seed 1), saving `docs/results/v3_1_learning_curve.json` and `docs/results/v3_1_learning_curve.svg`:
  - `classical_crf`: `0.9082` (`100`) -> `0.9551` (`200`) -> `0.9915` (`400`) -> `0.9961` (`600`, `11.1s`, `333.5 MB`)
  - `bilstm_crf`: `0.8898` (`100`) -> `0.9439` (`200`) -> `0.9832` (`400`) -> `0.9888` (`600`, `95.6s`, `402.5 MB`)
  - `transformer`: `0.5262` (`100`) -> `0.5509` (`200`) -> `0.5742` (`400`) -> `0.5765` (`600`, `18.4s`, `431.7 MB`)
  - `hybrid`: `0.8891` (`100`) -> `0.9540` (`200`) -> `0.9885` (`400`) -> `0.9961` (`600`, `168.8s`, `474.3 MB`)

### 11B Step 3 — Model Consolidation, Gradient Checks (G6) & Stacked Multi-Head Transformer (G5) (2026-11-14)
- Isolated duplicate legacy `medresearch_platform/extraction/` models (`bilstm.py`, `bilstm_crf.py`, `transformer.py`, `classical_ml.py`, `rule_based.py`, `hybrid_multimodal.py`, `nn_utils.py`, `benchmark.py`) to `extras/legacy/extraction/` and moved `tests/test_six_extraction_approaches.py` to `extras/tests/`; updated `/api/v1/benchmark` in `medresearch_platform/interfaces/api.py` to serve evaluated `v3_locked_test_evaluation.json` results.
- Exposed `loss_and_grads` on all 6 evaluated neural models (`NeuralBiLSTMTagger`, `NeuralBiLSTMCRFTagger`, `NeuralTransformerTagger`, `BiLSTMAssertionClassifier`, `PiecewiseCNNRelationClassifier`, `TransformerCandidateReranker`) and added finite-difference gradient checks (`float64`, `rel_err < 1e-4`) in `tests/test_evaluated_neural_gradients.py` (6/6 passing in `0.78s`).
- Upgraded `NeuralTransformerTagger` in `medresearch_platform/approaches/neural_seq.py` to a stacked multi-layer (`num_layers=2..3`), multi-head (`num_heads=4`), pre-LayerNorm Transformer with residual connections, sinusoidal positional encoding, padding mask, train-only `TrainOnlyBPETokenizer` sub-word embeddings, 32-dim char morphology features, and linear warm-up + cosine LR decay.
- Tuned on `SYNTH-MED-V3.1` `val` (600 train docs, `scripts/tune_v3_1_transformer.py`, `docs/results/v3_1_transformer_tuning.json`): `L2_H4_D64_ep18` improved Strict F1 from `0.5765` to **`0.8192`** (`Lenient = 0.9073`, `222.8s`); updated `configs/exp/v3_1/transformer.yaml`.
