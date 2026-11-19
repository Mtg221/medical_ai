# Contributing to `medical_ai`

## Locked Test-Set Discipline

- **Single evaluation per frozen model:** A locked test set (`SYNTH-MED-V3.1` test splits, `locked_test_v3`, etc.) is evaluated **once** per frozen model. All hyperparameter tuning, metadata selection, threshold calibration, and error analysis must happen on **`val` only**.
- **`post-hoc (test seen)` labelling:** If any code, lexicon, or post-processing rule changes *after* a locked test set has been evaluated, the subsequent re-evaluation **must** be logged in [`docs/results/TEST_SET_LOG.md`](docs/results/TEST_SET_LOG.md) and labelled **`post-hoc (test seen)`** in reports.
- **No deletions:** Never delete or overwrite historical rows in [`docs/results/TEST_SET_LOG.md`](docs/results/TEST_SET_LOG.md); annotate superseded runs in place.

## Local Checks Before Every Push

Run the exact CI suite locally before every push:

```bash
# macOS / Linux
make check            # or: bash scripts/check_all.sh

# Windows (PowerShell)
pwsh scripts/check_all.ps1
```

Install the optional local pre-push hook so `git push` runs `scripts/check_all.sh` automatically:

```bash
cat <<'EOF' > .git/hooks/pre-push
#!/usr/bin/env bash
set -euo pipefail
bash scripts/check_all.sh
EOF
chmod +x .git/hooks/pre-push
```

- Never commit `.env`, `var/`, `models/`, or corpus/clinical images (`data/router_real/`, `var/cache/derm/`).
