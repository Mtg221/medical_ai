# ADR-0010: PyTorch CPU (Random Initialization) for Dermatology Vision Models with Graceful Optional Extra

- **Status:** Accepted
- **Date:** 2026-11-16
- **Reconciles / Extends:** `ADR-0005`, `ADR-0008`, `ADR-0009`
- **References:** `docs/IMAGE_PROMPT.md`, `medresearch_platform/vision/`

---

## 1. Context

The platform is expanding to analyze dermatology skin images (`clinical`, `dermoscopic`, and `tbp_tile` modalities from `derm-corpus`) alongside clinical text documents within a unified patient record:
1. **Strict From-Scratch Constraint:** All primary models must be trained 100% from random initialization (`pretrained_weights_used = False`, zero ImageNet/torchvision pretrained weights).
2. **Multi-Task 2D Vision Requirements:** Image analysis requires a 5-class picture triage router (`skin_clinical`, `skin_dermoscopic`, `skin_tbp_tile`, `document_scan`, `other`), a multi-head 2D ResNet-style CNN (`malignancy` binary risk, harmonised `diagnosis` top-3 distribution, and `modality` head) on `128–160 px` RGB images, temperature calibration, and spatial Grad-CAM explanations on the final convolutional block.
3. **Intel macOS (`x86_64`) Wheel Constraint:** On the owner's workstation (`macOS Darwin x86_64`, Intel Core i7-9750H), PyTorch's last official `x86_64` macOS wheel is `torch==2.2.2` for Python `<= 3.12`, whereas the base text environment also runs on Python `3.11–3.13`.

---

## 2. Decision

1. **PyTorch CPU from Random Initialization for Vision CNNs:**
   - The multi-task ResNet-style skin classifier and picture router use PyTorch CPU (`torch.nn`) initialized explicitly from seeded Kaiming/Xavier random distributions with SHA-256 initial-weight manifests (`pretrained_weights_used: False`).
   - `torchvision` pretrained weights (`weights=...` or `pretrained=True`) are strictly forbidden.
2. **Text Models Remain Pure Vectorized NumPy (`ADR-0009`):**
   - All text NER, assertion, relation, and temporal models remain in pure NumPy/scikit-learn/CRFsuite so the core text pipeline runs with zero PyTorch dependency.
3. **Classical & Pure-NumPy Vision Baselines Retained:**
   - Every vision benchmark compares three from-scratch model families on identical lesion-grouped splits:
     1. **Classical Vision Baseline:** Colour histograms (`RGB`/`HSV`/`CIE-Lab`), texture (`LBP`, `GLCM`/Haralick), and Otsu lesion border/asymmetry geometry with calibrated Logistic Regression / SVM / HistGradientBoosting.
     2. **Pure-NumPy Toy CNN (`RandomInitDermConvNet`):** Hand-derived 2D convolution + pooling + metadata fusion with finite-difference gradient verification (`< 1e-4`).
     3. **PyTorch CPU Multi-Task ResNet (`DermResNetMultiTask`):** 4-stage residual CNN (`16 -> 32 -> 64 -> 128` channels, BatchNorm, ReLU, global pooling) with Grad-CAM and temperature scaling.
4. **Optional `.[vision]` Packaging & Graceful Runtime Degradation:**
   - `pyproject.toml` defines an optional `[project.optional-dependencies] vision` extra pinning `torch==2.2.2` on `darwin`/`x86_64` (`python_version <= "3.12"`) and `torch>=2.2.2` on Linux/Windows/ARM64.
   - For Intel macOS, a dedicated `.venv-vision` (Python 3.12 + `pip install -e ".[vision,dev]"`) is supported alongside the base environment.
   - If `torch` is not installed in the active interpreter, the FastAPI server, CLI, and test suite still start cleanly: classical and NumPy vision models remain available, and PyTorch-specific endpoints/CLI commands report `"vision PyTorch backend not installed (install with: pip install -e '.[vision]' on Python <=3.12)"`.

---

## 3. Consequences

- **Positive:** Enables true 2D residual CNN training on `128–160 px` dermatology images and exact Grad-CAM spatial heat maps on CPU while preserving 100% from-scratch scientific integrity and keeping the base platform runnable on any Python `>= 3.11` interpreter.
- **Negative / Operational:** On Intel macOS (`x86_64`), running the PyTorch CNN requires Python `3.12` (`.venv-vision`) due to upstream PyTorch wheel availability; this is documented in `README.md`.
