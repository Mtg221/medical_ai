# ADR-0009: Pure Vectorized NumPy vs. PyTorch for Evaluated From-Scratch Neural Models

- **Status:** Accepted
- **Date:** 2026-11-15
- **Supersedes / Reconciles:** `ADR-0003`, `ADR-0005`
- **References:** `tests/test_evaluated_neural_gradients.py`, `medresearch_platform/approaches/neural_seq.py`, `medresearch_platform/hybrid/`

---

## 1. Context

`medical_ai` requires all primary neural models to run CPU-only on the target workstation (`macOS x86_64`, `Python 3.13.12`), initialized strictly from scratch with zero pretrained weights or external deep-learning hubs. Earlier in the project:
1. PyTorch dropped official `macOS x86_64` wheels after `torch 2.2.2` (Python `<= 3.12`), blocking native `pip install torch` on Python `3.13`.
2. Two parallel neural implementations briefly existed (`medresearch_platform/extraction/*` vs. `medresearch_platform/approaches/neural_seq.py` + `medresearch_platform/hybrid/*`), creating a gap where gradient checks covered legacy classes rather than the evaluated models.

---

## 2. Decision

1. **Standardize on One Pure-NumPy Implementation:**
   - All evaluated neural models live in `medresearch_platform/approaches/neural_seq.py` (`NeuralBiLSTMTagger`, `NeuralBiLSTMCRFTagger`, `NeuralTransformerTagger`) and `medresearch_platform/hybrid/` (`BiLSTMAssertionClassifier`, `PiecewiseCNNRelationClassifier`, `TransformerCandidateReranker`).
   - Legacy duplicate models were isolated to `extras/legacy/` so one codebase is trained, tested, evaluated, and served by FastAPI.
2. **Vectorized CPU Performance:**
   - Use mini-batched padded tensor operations with attention/padding masks and `float32` buffers (for inference/large-batch ops) so all models train on the full `600`-document `SYNTH-MED-V3.1` training split under an equal data budget.
3. **Mandatory Finite-Difference Gradient Verification:**
   - Every evaluated neural model exposes `loss_and_grads(...)` and is verified in `float64` against central finite differences (`relative error < 1e-4`) in `tests/test_evaluated_neural_gradients.py`, including CRF transition gradients in `NeuralBiLSTMCRFTagger`.

---

## 3. Comparison: Pure NumPy vs. PyTorch (CPU)

| Criterion | Pure Vectorized NumPy (Chosen) | PyTorch (CPU) |
|---|---|---|
| **Python 3.13 `macOS x86_64` install** | Native wheel (`numpy 2.4.2`), zero build issues | No official `x86_64` macOS wheel for Python 3.13 |
| **From-scratch auditability** | Every forward/backward equation & CRF Viterbi/forward-backward pass is explicit | Autograd abstracts backward passes |
| **Gradient correctness risk** | Higher manual implementation risk — mitigated by `< 1e-4` finite-difference tests on all 6 models | Automatic differentiation |
| **CPU training throughput** | Slower for deep/wide Transformers (`L=2, H=4, d=64` takes ~185s/seed) | Faster C++ ATen/oneDNN kernels |
| **Deployment footprint** | Minimal (`numpy`, `scipy`, `scikit-learn`) | Large (`~200–700 MB` binary dependency) |

---

## 4. Consequences & Migration Criteria

- **Positive:** 100% reproducible, dependency-light, CPU-only execution on Python 3.13 `x86_64`, with mathematically verified gradients across all 6 neural classes.
- **Negative:** Deeper attention architectures (`> 4` layers or large vocabularies) are compute-bound on pure NumPy; this is documented honestly alongside the Transformer tuning results (`0.8192` peak on `val`).
- **Re-evaluation / Migration Trigger:** If the project migrates to `arm64` Apple Silicon or Linux `x86_64` containers and requires `> 10x` larger corpora or multi-GPU training, PyTorch CPU/CUDA backends may be introduced behind the same `Evaluator` and `APPROACH_REGISTRY` interfaces while retaining pure-NumPy finite-difference parity tests as reference checks.
