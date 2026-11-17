# Model Card: `PyTorchDermResNet4Stage-v1.0.0` & `PyTorchPictureTriageCNN-v1.0.0`

- **Skin Model (`PyTorchDermResNet4Stage-v1.0.0`):** From-Scratch 4-Stage Residual CNN (`32 → 64 → 128 → 256` channels, `BatchNorm2d`, `ReLU`, `AdaptiveAvgPool2d`, `1,229,845` parameters, `96×96` RGB input, all layers trained)
- **Router (`PyTorchPictureTriageCNN-v1.0.0`):** From-Scratch 4-Block PyTorch CNN (`16 → 32 → 64 → 128`, `98,565` parameters, `96×96` RGB input, 5 classes)
- **Initialization:** 100% Random Initialization (`pretrained_weights_used = false`, `initial_weights_sha256` + `trained_weights_sha256` recorded in `manifest.json`)
- **Execution Profile:** CPU-Only Native (`torch==2.2.2` on Python 3.12 Intel macOS `x86_64`; `torch>=2.2.2` on Linux/Windows/ARM64)
- **Artefact Locations:** `models/derm_cnn_seed1/` (`manifest.json` + `weights_torch.pt`), `models/vision_router_v1/` (`manifest.json` + `router_torch.pt`), `configs/vision/abstention_v2.json`

> **Clinical & Scientific Safety Notice (EN / FR)**
> - **EN:** Research tool — not a diagnosis. A dermatologist must examine the lesion.
> - **FR :** Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion.

---

## 1. Intended Use & Out-of-Scope / Not-Intended Use

- **Intended Use:**
  - Research benchmarking of from-scratch 2D dermatology risk estimation, picture modality triage (`skin_clinical`, `skin_dermoscopic`, `skin_tbp_tile`, `document_scan`, `other`), bilingual image-quality gating, and spatial Grad-CAM explanations on `derm-corpus`.
  - Human-in-the-loop research review inside an authenticated institutional workspace (`review_status = "needs_review"` on every AI-generated image finding until confirmed by a reviewer).
- **Not-Intended / Prohibited Use:**
  - **NOT** a medical device and **NOT** cleared or validated for autonomous clinical diagnosis, triage, biopsy deferral, or treatment decisions.
  - Must **never** be deployed without mandatory dermatologist examination of the lesion.

---

## 2. Training Data, Splits & Licences (`derm-corpus` v2)

- **Metadata Catalog:** `573,024` records (`305,964` eligible with `quality_ok == true` and `research_allowed == true`).
- **Real Stratified Subset ([DERM_CORPUS_CARD_V2.md](../data/DERM_CORPUS_CARD_V2.md)):**
  - `train`: `26,474` real thumbnails (`18,251` benign, `5,804` malignant, `2,419` masked unknown/indet; `4,519` clinical, `20,449` dermoscopic, `1,506` TBP tile).
  - `val`: `3,000` real thumbnails (`150` per Fitzpatrick tone `I–VI`).
  - `locked_test_v2` ([locked_test_v2.txt](../../configs/vision/locked_test_v2.txt)): **`2,022`** frozen real thumbnails (`1,853` labelled `mg`: `1,064` malignant, `789` benign; `169` masked; `978` cross-split `dHash` near-duplicates dropped from test, `0` dropped from train).
  - **Zero** lesion `group_key` overlap across splits and **zero** synthetic images in any benchmark split.
- **Router Real Dataset:** Real skin thumbnails (`train` split) + rendered SYNTH-MED-V3.1 notes (`document_scan`) + `346` Wikimedia Commons CC0/PD everyday photos (`other`, recorded in `data/router_real/SOURCES.csv`) + `50` CC0/PD OOD check images (`cars`, `landscapes`, `hands`, `wounds`, `tattoos`).

---

## 3. Locked-Test Performance (`locked_test_v2`, `n = 1,853` Labelled)

- **PyTorch 4-Stage ResNet (`seed=1`):**
  - Malignancy AUROC: **`0.9398`** (`95% CI [0.9296, 0.9508]`)
  - 3-Seed Ensemble AUROC: **`0.9421`** (`[0.9325, 0.9528]`) · No-Metadata Ablation AUROC: **`0.9461`** (`[0.9369, 0.9555]`)
  - Sensitivity @ 90% Specificity: `0.8186` · Sensitivity @ 95% Specificity: `0.6786` · Balanced Accuracy: `0.8389`
  - Calibration: ECE `0.1554`, Brier `0.1334` · Abstention Rate (`min_confidence = 0.65` selected on `val`): `12.84%`
  - Grad-CAM Adebayo Weight-Randomisation Sanity: `mean L1 shift = 0.3546` (`"Where the model looked — not proof of a diagnosis."`)
- **Baselines on Identical `locked_test_v2`:**
  - `classical_logreg` (64 handcrafted features): AUROC `0.7321` `[0.7073, 0.7586]`, **wins on 9-class diagnosis Macro-F1 (`0.7447`)**
  - `numpy_frozen_random_features`: AUROC `0.7034` `[0.6797, 0.7298]`
  - `metadata_only_logreg` (no pixels): AUROC `0.6364` `[0.6097, 0.6651]`
- **Subgroup Performance & Limitations:**
  - **Modality:** `dermoscopic` (`n=1,160`) AUROC `0.9783` `[0.9710, 0.9857]` (ECE `0.0481`); `clinical` (`n=399`) AUROC `0.9029` `[0.8522, 0.9340]`; `tbp_tile` (`n=294`, `288` mal / `6` ben) AUROC `0.7095` `[0.4811, 0.9070]` (ECE `0.6503`).
  - **Fitzpatrick Tone:** `IV` (`n=44`, `30` mal / `14` ben) AUROC `0.9286` `[0.8387, 1.0000]`; `V` (`n=27`) is **too few to conclude (`< 30`)**; `I` (`128`), `II` (`124`), `III` (`129`), and `VI` (`38`) each contain a single malignancy class in `locked_test_v2` (AUROC undefined).

---

## 4. Privacy & Safety Controls

- **Fail-Loud Serving (503 / 404):** Missing/tampered models return HTTP `503` (no random weights); unknown/missing `subject_id` returns `404`/`422` (no cross-patient fallback); missing capture date stays `null` (`captured_date_source = "unknown"`).
- **EXIF & GPS Stripping:** All uploaded images have EXIF, GPS, XMP, and ICC metadata stripped before storage (`exif_stripped = 1`).
- **Abstention Gate (`configs/vision/abstention_v2.json`):** `min_confidence = 0.65` selected on `val` (`val_coverage = 86.81%`, `val_selective_error = 10.20%`); non-skin images (`other`) never receive a malignancy risk score.
