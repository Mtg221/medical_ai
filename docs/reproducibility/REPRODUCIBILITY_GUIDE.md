# Reproducibility & Random-Init Provenance Guide

**Specification References:** Sections 7, 17, 19, 20, 49

---

## 1. Random Initialization Verification
Every model in the primary research track (`PRIMARY_RANDOM_INIT`) records:
- `architecture_name` and `architecture_version`
- `init_method` (`he_normal`, `xavier_uniform`, `normal_small`)
- `seed` (default `20261112`)
- `parameter_count` and exact `tensor_shapes`
- `initial_weights_sha256` (computed prior to training)
- `pretrained_weights_used = False`

Before training, `verify_random_init_provenance(manifest, weight_builder_fn)` re-runs random initialization from the recorded seed and verifies an exact bit-for-bit SHA-256 match against `initial_weights_sha256`.

## 2. Reproducing the Phase 1 Benchmark
```bash
MEDRESEARCH_RANDOM_SEED=20261112 python3 -m medresearch_platform.interfaces.cli run-experiment
```

Expected reproducible outputs:
- **Immutable Manifest:** `var/storage/experiments/exp_phase1_baseline_v1_manifest.json`
- **CustomOCRConvNet-v1 Initial Weights SHA-256:** `1d58c59e6192df7cd72753733590ce1f8014b2357730c2d09aeed74fe03ac25d`
- **CustomMedicalNLPNet-v1 Initial Weights SHA-256:** `b324edda246624b1848b67d804d0e0e45ebbbb9878d99a10f86c063d1a20ea4f`
- **Split Leakage Check:** `leak_detected: false` (0 overlapping subjects/families across `train`, `val`, `locked_test`).

## 3. Verification Test Suite
Run all 17 unit, security, scientific, and integration tests:
```bash
python3 -m pytest -v
```
