# Locked Test Set Evaluation Log

> Per Phase 11 Rule 4: Model selection uses `val` only. Each locked test set is evaluated **once** per final config, and every evaluation is recorded here.

| Date (UTC) | Corpus | Split(s) Evaluated | Docs / Split | Configs | Script | Output JSON | Base Commit | Run Count | Operator |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-14 | `SYNTH-MED-V3` (`data/processed/v3`) | `test_iid`, `test_unseen_terms`, `test_unseen_templates` | 160 each (80 EN + 80 FR) | `configs/exp/v3/{rule_based,classical_ml,classical_crf,bilstm,bilstm_crf,transformer,hybrid}.yaml` (seeds 1, 2, 3) | `scripts/run_v3_locked_test_eval.py` | `docs/results/v3_locked_test_evaluation.json` | `27fdebc` (`phase-11-step-7-frontend-complete`) | 1 (one-time locked evaluation) | Automated Phase 11 Step 8 runner |
| 2026-11-14 | `SYNTH-MED-V3.1` (`data/processed/v3_1`) | `val`, `test_unseen_terms`, `test_unseen_templates` | 160 each (80 EN + 80 FR) | `configs/exp/v3_1/hybrid.yaml` (6 ablation variants, seeds 1–5) | `scripts/run_v3_1_step4_ablations.py` | `docs/results/v3_1_step4_ablations.json` | `045dbc1` (`phase-11b-step-3-models-and-gradients`) | 1 (Phase 11B Step 4 ablation after Stage 3/4 bug fixes) | Automated Phase 11B Step 4 runner |
