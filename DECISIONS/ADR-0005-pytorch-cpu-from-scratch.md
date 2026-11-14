# ADR-0005: PyTorch (CPU Build) with Strict Random Initialization for Neural Models

- **Status:** Accepted
- **Date:** 2026-11-14
- **References:** `docs/MASTER_PROMPT.md` Parts B.1, C.1; `docs/AUDIT.md`

## Context

The initial implementation built BiLSTM, BiLSTM-CRF, and Transformer encoders in pure NumPy with manual backprop. On CPU, hand-written NumPy training loops are slow, fragile, and produced `F1 = 0.00` for BiLSTM and Transformer on the baseline benchmark.

## Decision

1. **PyTorch (CPU Build) as Primary Deep Learning Engine:** Use `torch` (CPU-only) for all primary neural architectures (BiLSTM, BiLSTM-CRF, Transformer encoder, BiLSTM assertion classifier, CNN relation extractor) starting in Phase 3.
2. **Strict Random Initialization:** All neural models in the primary research track must be initialized strictly from random weights (`torch.nn.init` / fixed seeds). Zero pretrained weights, zero pretrained word vectors, and zero pretrained tokenizers are permitted.
3. **Educational Reference Retained:** Keep the existing NumPy BiLSTM/CRF modules and their finite-difference gradient-check tests as a verified educational reference.

## Consequences

- **Positive:** Correct autograd, 10–50× faster CPU training, clean batching/masking/early stopping, and reliable convergence on real-sized corpora (≥ 600 documents).
- **Negative / Limitations:** Adds `torch` (CPU wheel) as a project dependency; training configs must remain sized for 6-core Intel CPU / 32 GB RAM.
