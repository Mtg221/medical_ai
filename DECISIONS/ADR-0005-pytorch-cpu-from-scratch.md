# ADR-0005: Pure-NumPy Random-Init Neural Models + `sklearn-crfsuite` on Python 3.13 (macOS `x86_64`)

- **Status:** Amended (2026-11-14)
- **Date:** 2026-11-14
- **References:** `docs/MASTER_PROMPT.md` Parts B.1, C.1; `docs/AUDIT.md`

## Context

The initial ADR-0005 proposed PyTorch (CPU build) for primary neural models. Upon attempting installation on the target workstation (`macOS x86_64`, `Python 3.13.12`), `pip install torch` failed because PyTorch dropped macOS `x86_64` binary wheels after `torch 2.2.2` (which only supports up to Python 3.12), and neither Python 3.11 nor 3.12 is installed.

## Decision

1. **Pure-NumPy Random-Init Neural Engines:** Rebuild and fix the pure-NumPy random-initialized `BiLSTM`, `BiLSTM-CRF`, and `Transformer` encoders directly on Python 3.13 + `numpy 2.4.2`, fixing class-weighted loss, masking, char/shape features, AdamW learning rates, and `val` early stopping so all three converge cleanly on CPU.
2. **Classical CRF Baseline:** Use `sklearn-crfsuite` (`python-crfsuite 0.9.12`, compiled cleanly on Python 3.13 `x86_64`) + `scikit-learn` for classical ML baselines.
3. **Strict Random Initialization:** Zero pretrained weights, zero pretrained word vectors, and zero pretrained tokenizers are permitted.

## Consequences

- **Positive:** Zero external deep-learning binary wheel issues; 100% reproducible CPU execution on Python 3.13 `x86_64`; complete transparency of every gradient and forward pass.
- **Negative / Limitations:** Training throughput is lower than PyTorch C++ kernels, so hidden dimensions and sequence windows are sized for fast CPU convergence.
