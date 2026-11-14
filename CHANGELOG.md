# Changelog

## [Unreleased] — Phase 11 ("Make the results true")

### Step 0 — Audit & Honesty Reset (2026-11-14)
- Added `scripts/audit_corpus.py` to detect doubled span text, adjacent duplicate n-grams, offset mismatches, lexical overlap, and digit-masked sentence/template overlap.
- Audited `SYNTH-MED-V2` (`data/processed/v1`): confirmed 100% of docs had doubled TIMEX/date/duration spans (F1) and `94.8%` masked test sentence / `62.7%` unique entity overlap (`100%` across clinical entity types) (F2).
- Archived v2 locked test results and audit output in `docs/results/archive/2026-11-14_synth_v2_buggy/`.
- Marked `docs/RESEARCH_REPORT.md` as Superseded and reset `PROJECT_STATUS.md` to `Phase 11 — correction`.
