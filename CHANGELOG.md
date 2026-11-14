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
