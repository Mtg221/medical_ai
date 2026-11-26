$ErrorActionPreference = "Stop"

Write-Host "[1/7] Lint (ruff)"
ruff check medresearch_platform tests scripts

Write-Host "[2/7] Type check (mypy)"
mypy medresearch_platform

Write-Host "[3/7] Build & audit SYNTH-MED-V3 and V3.1"
medextract data build --config configs/data/v3.yaml
python scripts/audit_corpus.py --data-dir data/processed/v3 --fail-on-doubled
medextract data build --config configs/data/v3_1.yaml
python scripts/audit_corpus.py --data-dir data/processed/v3_1 --fail-on-doubled

Write-Host "[4/7] Backend pytest"
pytest -q

Write-Host "[5/7] Docs link check"
pytest tests/test_no_file_uri_in_docs.py -q

Write-Host "[6/7] Frontend tests + production build"
npm --prefix frontend test -- --run
npm --prefix frontend run build

Write-Host "[7/7] Text + image acceptance suite"
python scripts/acceptance_test.py

Write-Host "All local CI checks PASSED."
