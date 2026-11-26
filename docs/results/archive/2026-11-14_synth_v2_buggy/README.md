# Archived Results: `SYNTH-MED-V2` (2026-11-14, Commit `9e8d443`)

These results are **invalid** and archived for historical transparency per Phase 11 Rule 2.

## Why these results are invalid
1. **F1 (Data-generator doubled text bug):** `_DocBuilder.add_entity()` followed by `_DocBuilder.add_timex()` in `medresearch_platform/data/synthetic_v2.py` appended the same date, duration, and frequency strings **twice** (`2026-02-262026-02-26`, `for 7 daysfor 7 days`, `as neededas needed`), affecting 100% of documents (`555/555` train, `124/124` val, `121/121` test).
2. **F2 (Train/test term & template overlap):** `62.7%` of unique test entity strings (and `100.0%` of clinical categories like `DIAGNOSIS`, `SYMPTOM`, `MEDICATION`, `PROCEDURE`, `LAB_TEST`) appeared in `train`, and `94.8%` of digit-masked test sentences were identical to a `train` sentence.

See `audit.txt` in this directory for the exact reproduction output from `scripts/audit_corpus.py`.
