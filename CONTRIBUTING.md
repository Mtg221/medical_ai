# Contributing to `medical_ai`

## Locked Test-Set Discipline

- **Single evaluation per frozen model:** A locked test set (`SYNTH-MED-V3.1` test splits, `locked_test_v3`, etc.) is evaluated **once** per frozen model. All hyperparameter tuning, metadata selection, threshold calibration, and error analysis must happen on **`val` only**.
- **`post-hoc (test seen)` labelling:** If any code, lexicon, or post-processing rule changes *after* a locked test set has been evaluated, the subsequent re-evaluation **must** be logged in [`docs/results/TEST_SET_LOG.md`](docs/results/TEST_SET_LOG.md) and labelled **`post-hoc (test seen)`** in reports.
- **No deletions:** Never delete or overwrite historical rows in [`docs/results/TEST_SET_LOG.md`](docs/results/TEST_SET_LOG.md); annotate superseded runs in place.

## Local Checks Before Every Push

```bash
ruff check medresearch_platform scripts tests
mypy medresearch_platform
pytest -q
npm --prefix frontend test && npm --prefix frontend run build
```

- Never commit `.env`, `var/`, `models/`, or corpus/clinical images (`data/router_real/`, `var/cache/derm/`).
