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

## 2. Training Data, Splits & Licences (`derm-corpus` v3)

- **Metadata Catalog:** `573,024` records across `isic`, `fitzpatrick17k`, `scin`, `skindisnet` ([DERM_FIELDS.md](../data/DERM_FIELDS.md)).
- **Lesion-Similarity Grouped Corpus v3 ([DERM_CORPUS_CARD_V3.md](../data/DERM_CORPUS_CARD_V3.md)):**
  - `train`: `31,565` real thumbnails (`20,630` benign, `8,125` malignant, `2,810` masked `mg`; includes `2,613` owner-approved labelled `fitzpatrick17k` clinical atlas images).
  - `val`: `1,497` real thumbnails (`900` benign, `597` malignant), filled after `locked_test_v3`.
  - `locked_test_v3` ([locked_test_v3.txt](../../configs/vision/locked_test_v3.txt)): **`1,624`** labelled thumbnails (`700` dermoscopic, `574` clinical, `350` TBP tile; Tone `I–IV` `75/75+`, Tone `V` `75/60`, Tone `VI` `75/27`). **0** lesion-group overlap (`lesion_groups_v3.json`, `24,877` groups).
  - *Superseded:* `locked_test_v2` (`n=1,853`) is retained as historical only because 380 similarity groups spanned >1 `v2` split (W1).
- **Router Real Dataset:** Real skin thumbnails + phone-photo-augmented rendered SYNTH-MED notes + `334` CC0/PD Wikimedia document scan photos (`217` train / `34` val / `83` held-out eval; `0` owner-taken photos) + `346` Wikimedia CC0/PD everyday photos (`other`) + `50` OOD check images (`data/router_real/SOURCES.csv`), paired with pre-skin `detect_dense_text_lines()` (`document_scan -> skin = 0 / 83`).

---

## 3. Locked-Test Performance (`locked_test_v3`, `n = 1,624` Labelled)

- **Lead Result — Served-Path End-to-End (`POST /api/v1/images/upload` with `models/vision_router_v1` + `models/derm_cnn_seed1` [`full_body_site_dropout_40`], `n = 1,624`):**
  - Router routed `89/1,624` (`5.5%`) to `other` (abstained) and `0` to `document_scan`; `skin_scored = 934`, `skin_abstained = 601`.
  - `dermoscopic` (`n=700`): AUROC **`0.8849`** `[0.8630, 0.9099]`, Sens@90%Spec `0.6167`, BalAcc `0.7929`, ECE `0.0611`, Abstain `26.3%`
  - `clinical` (`n=574`): AUROC **`0.7195`** `[0.6739, 0.7613]`, Sens@90%Spec **`0.2161`**, BalAcc `0.6930`, ECE `0.1189`, Abstain `29.8%`
  - `tbp_tile` (`n=350`): AUROC **`0.4819`** `[0.4458, 0.5131]`, Abstain `95.7%` (automatic TBP abstention)
  - Overall served-path (`n=1,624`): AUROC **`0.7842`** `[0.7646, 0.8057]`, BalAcc `0.6744`, ECE `0.0555`, Abstain `42.5%`, High-Risk Prec `0.7972` (corpus-prevalence-reweighted `3.28%` prev: `PPV=0.3837`, `NPV=0.9864`).
- **Offline Per-Modality (Direct 3-Seed `PyTorchDermResNet4Stage-v1.0.0` Ensemble Without Router):**
  - `dermoscopic` (`n=700`): AUROC **`0.9551`** `[0.9397, 0.9690]`, ECE `0.1033`, BalAcc `0.8729` (reweighted `9.35%` prev: `PPV=0.5457`, `NPV=0.9957`)
  - `clinical` (`n=574`): AUROC **`0.7803`** `[0.7380, 0.8199]`, ECE `0.0813`, BalAcc `0.7288`; specific `dx` (`n=280`): Macro-F1 `0.3036`, Top-3 `0.7250`
  - `tbp_tile` (`n=350`): AUROC **`0.7923`** `[0.7386, 0.8388]`, ECE `0.0589`, BalAcc `0.6500` — **at corpus prevalence (`0.14%`), `PPV = 0.0327` (`3.3%`), `ECE = 0.2082` -> serving abstains automatically on `tbp_tile` inputs.**
- **Overall (`n = 1,624`; mixes modalities with different base rates):**
  - 3-Seed Ensemble: AUROC **`0.8855`** `[0.8705, 0.9012]` · `seed1`: `0.8275` (`seed2` best val `0.8833`) · `no_meta`: `0.7772` · `classical_logreg`: `0.7880` · `numpy_frozen_random_features`: `0.7343` · `metadata_only_logreg`: `0.6554`
- **Grad-CAM Sanity (`n = 220` dermoscopic `val_v3` images, threshold `>= 0.30`):** `adebayo_mean_l1_shift = 0.2270` (**`adebayo_passed = false`**), `dermoscopic_center_heat_share = 0.2316`. While failing this sanity check, the Grad-CAM overlay is **off by default in the UI** (hidden behind an `"Experimental (failed sanity check)"` toggle) and **omitted from exported patient reports**.
- **Clinical (Phone) Photo Modality Caveat:** Served `skin_clinical` sensitivity at 90% specificity is `0.2161 < 0.50`, so the UI and reports **suppress risk bands** on `skin_clinical` photos and direct users to consult a dermatologist.
- **Tone × Collection Shortcut Caveat:** Tones `IV–VI` malignant images come largely from `fitzpatrick17k` vs benign from `isic`; a pixel-free `tone + dataset` classifier achieves `0.9733` AUROC on Tone `VI`, so dark-skin AUROC (`0.9694`) reflects dataset-source confounding.

---

## 4. Privacy & Safety Controls

- **Fail-Loud Serving (503 / 404):** Missing/tampered models return HTTP `503` (no random weights); unknown/missing `subject_id` returns `404`/`422` (no cross-patient fallback); missing capture date stays `null` (`captured_date_source = "unknown"`).
- **EXIF & GPS Stripping:** All uploaded images have EXIF, GPS, XMP, and ICC metadata stripped before storage (`exif_stripped = 1`).
- **Abstention Gate (`configs/vision/abstention_v3.json`):** `min_confidence = 0.61` selected on `val_v3` (`val_coverage = 76.15%`, `val_selective_error = 18.95%`); `tbp_tile` and non-skin images (`other`) never receive a malignancy risk score.
