# Archived: `2026-11-15_realistic_v1_invalid`

**Status:** INVALIDATED & WITHDRAWN (Findings H1, H2, H5 in Correction Prompt 12B)

Why these artifacts are archived and excluded from active benchmarks:
1. **H1 (Template-generated):** `realistic_v1.py::_build_note` used one fixed sentence frame per language rather than genuinely un-templated owner-authored clinical prose (`docs/REALISTIC_SET_PROTOCOL.md`).
2. **H2 (Mismatched label set):** Used `PROBLEM` and `TEST` entity types instead of the canonical `SYNTH-MED-V3.1` label set (`DIAGNOSIS` / `SYMPTOM` / `LAB_TEST`), making 240/560 (43%) of gold spans unmatchable.
3. **H5 (Test-informed Hybrid v2):** `phase12_hybrid_v2_and_realistic.json` also contains a `hybrid_v2` run whose configuration was chosen after inspecting test-split ablations rather than `val`-only selection. Validated `val`-selected Hybrid v2 results supersede this file.
