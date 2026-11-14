# Changelog

## [Unreleased] — Phase 11 ("Make the results true")

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
