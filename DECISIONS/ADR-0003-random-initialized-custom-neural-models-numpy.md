# ADR-0003: Custom Neural Architectures with Verified Random Initialization in Pure NumPy/SciPy

- **Status:** Accepted
- **Date:** 2026-11-12
- **Specification References:** Sections 6, 7, 8, 9, 10, 19, 20, 21, 23

## Context

Section 7 mandates that the primary research track:
- Use custom architectures designed specifically for the project.
- Initialize all trainable weights randomly from explicit mathematical distributions (`he_normal`, `xavier_uniform`, `orthogonal`) controlled by recorded seeds.
- Record initialization method, seed, architecture version, parameter count, and cryptographic weight-origin verification.
- Strictly avoid pretrained model weights, pretrained embeddings, or hidden external services.

Baseline environment inspection confirms that `numpy 2.4.2`, `scipy 1.17.1`, and `scikit-learn 1.9.1` are installed in Python 3.13, while PyTorch/TensorFlow are not installed and network access is disabled in the sandbox.

## Decision

1. **Custom Analytical Neural Networks in NumPy/SciPy:** We implement project-designed neural architectures with explicit analytical forward and backward propagation, AdamW optimization, gradient clipping, and serialization:
   - **Approach 1 (OCR Intelligence — `CustomOCRConvNet`):** A custom 2D Convolutional + ReLU + MaxPool + Two-Layer MLP neural classifier initialized via He Normal random weights (`seed`-controlled) operating on 16x16 character/token image patches extracted with bounding box coordinates, producing probability distributions, Shannon entropy (aleatoric uncertainty), and margin/dispersion metrics.
   - **Approach 2 (Medical NLP Intelligence — `CustomMedicalSequenceNet`):** A custom randomly initialized Character-Ngram + Contextual Token Embedding + Bi-Directional Recurrent Context Layer + Multi-Head Linear Classifier (`entity_head`, `assertion_head`, `temporality_head`) trained end-to-end from random weights via analytical backpropagation, paired with a documented deterministic clinical relation & modifier verifier.
2. **Cryptographic Weight-Origin Verification:** At initialization time, before any training step, the exact tensor shapes, random seed, initialization scheme, parameter count, and SHA-256 digest of initial weights (`initial_weights_sha256`) are recorded in the model manifest and verified upon registration in the Model Registry.

## Consequences

- **Positive:** Complete mathematical transparency; zero risk of accidental pretrained weight contamination; fast CPU training on the Intel Core i7-9750H; fully self-contained reproducibility.
- **Negative / Limitations:** Pure NumPy CPU models are intentionally compact and evaluated on controlled synthetic/authorized datasets; they are research baselines and not clinically validated diagnostic models.
