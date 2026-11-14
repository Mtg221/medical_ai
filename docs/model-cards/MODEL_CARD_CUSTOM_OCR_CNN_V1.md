# Model Card: CustomOCRConvNet v1.0.0

- **Model ID:** `mdl_ocr_convnet_v1`
- **Approach Category:** `DOCUMENT_OCR` (Approach 1)
- **Research Track:** `PRIMARY_RANDOM_INIT` (Zero pretrained weights)
- **Lifecycle Status:** `RESEARCH_ONLY` (**Experimentally validated** on synthetic benchmark; **Not clinically validated**)

## 1. Architecture & Random Initialization Provenance
- **Architecture:** 16x16 Character Patch 2D Spatial Projection (`16x8` filters across 4x4 grid = 128 spatial features) concatenated with 256 raw pixel features $\rightarrow$ `Linear(384, 96)` + ReLU $\rightarrow$ `Linear(96, 45)` softmax over 45 uppercase alphanumeric & clinical punctuation symbols.
- **Parameter Count:** `41,461` trainable float32 parameters
- **Initialization Method:** `he_normal` (biases initialized to `zeros`)
- **Random Seed:** `20261112`
- **Initial Untrained Weights SHA-256:** `1d58c59e6192df7cd72753733590ce1f8014b2357730c2d09aeed74fe03ac25d`
- **Trained Weights SHA-256:** `20ebb835ab7404eae0d9181c11c51187d6758f07aaffbc371b7fbbcce332d75d`
- **Optimizer:** Analytical backpropagation with AdamW ($\text{lr}=0.012$, $\beta_1=0.9$, $\beta_2=0.999$, weight decay $10^{-4}$, 140 epochs)

## 2. Evaluation & Uncertainty Quantification
- **Clean Synthetic Scans ($\sigma_{\text{noise}} = 0.01$):** `CER = 0.0000`, `WER = 0.0000` (`100.0%` character accuracy)
- **Robustness Under Heavy Gaussian Noise ($\sigma_{\text{noise}} = 0.14$):** `CER = 0.1895` (`81.05%` character accuracy)
- **Uncertainty Outputs:** Per-block and per-character top-1 confidence, normalized Shannon entropy (aleatoric uncertainty), and top-2 probability margin dispersion (epistemic uncertainty proxy).

## 3. Intended Use, Prohibited Use & Limitations
- **Intended Use:** Research evaluation of custom random-init optical character recognition and uncertainty propagation on synthetic or authorized clinical documents.
- **Prohibited Use:** Unreviewed clinical transcription, autonomous diagnosis, or replacing original source documents.
- **Limitations:** Trained on 16x16 synthetic monospace bitmap glyphs; handwritten text and arbitrary rotation require dedicated spatial transformer / CRNN extensions.
