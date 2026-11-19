# Kaggle GPU Training Notebooks (100% From Scratch)

These notebooks run our PyTorch training pipelines on **Kaggle's free NVIDIA T4 / P100 GPUs** (up to 30 hours/week) with **`pretrained_weights_used = False`** (strictly random initialization, zero external pretrained weights).

> **Hospital Privacy & Compliance Rule:** ONLY train on public / open-license / synthetic datasets on Kaggle. **Never upload real hospital patient notes, clinical photos, or PHI/PII** to Kaggle. Train on public data here, download the signed `.pt` checkpoints + SHA-256 manifests, and run inference (or private fine-tuning) strictly on-premise inside the hospital.

---

## Notebooks

1. **`kaggle_gpu_nlp_hospital.ipynb`** — Trains a **3-seed Unified Hospital NLP Model** (`PyTorchCharCNNBiLSTMCRF` with Exact Partial-Label Marginal CRF, `hidden_dim=128`, `epochs=20`) jointly on:
   - `SYNTH-MED-V3.1` (EN + FR, 14 clinical entity types)
   - `MACCROBAT2020` (EN real case reports, `CC-BY-4.0`)
   - `PxCorpus` (FR real clinical prescriptions, `CC-BY-4.0`)
   - `E3C Layer 1` (EN + FR clinical narratives, `CC-BY-NC-4.0` — pass `--permissive-only` to omit for commercial builds)
   - `BC5CDR` (EN biomedical Disease + Chemical, `Public-Domain`)
   - `NCBI Disease` (EN biomedical Disease, `Public-Domain`)
   - From-scratch E3C Layer 3 self-supervised EN/FR semantic clusters (`models/e3c_self_supervised_en_fr.npz`)
   - Output artifact: `/kaggle/working/hospital_nlp_models_gpu.zip`

2. **`kaggle_gpu_vision_hospital.ipynb`** — Trains a **3-seed `224x224` Dermatology Vision Ensemble** (`PyTorchDermResNet`, CUDA FP16 mixed precision `AMP`, `body_site_dropout=0.40`, `epochs=25`) using the locked patient/lesion-grouped v3 splits (`configs/vision/val_v3.txt`, `configs/vision/locked_test_v3.txt`), fits `val_v3`-only per-modality calibration temperatures, and runs the biopsy-confirmed (`histopathology`) audit.
   - Output artifact: `/kaggle/working/hospital_vision_models_gpu.zip`

---

## How to Run on Kaggle (4 Steps)

1. Go to [https://www.kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → **File → Import Notebook** → upload either `.ipynb` from `notebooks/`.
2. In the right-hand **Notebook options** panel:
   - **Accelerator:** select **GPU T4 x2** or **GPU P100**.
   - **Internet:** turn **ON** (needed to clone `Mtg221/medical_ai` and download open corpora).
3. *(Vision notebook only)* In **Add-ons → Secrets**, add two secrets (never hardcode credentials in cells):
   - `DERM_CLOUDFRONT_USER`
   - `DERM_CLOUDFRONT_PASS`
4. Click **Run All** (or **Save & Run All (Commit)** to run in the background). When finished, download the `.zip` from the **Output** tab and extract it into your local repo root:

```bash
unzip -o hospital_nlp_models_gpu.zip -d .
unzip -o hospital_vision_models_gpu.zip -d .
```

The live server automatically detects all 3 NLP seeds (`models/hospital_char_cnn_bilstm_crf_seed{1,2,3}`), all 3 Vision seeds (`models/derm_cnn_bsdrop40_seed1`, `models/derm_cnn_seed2`, `models/derm_cnn_seed3`), and the calibrated temperatures (`models/derm_ensemble_calibration.json`) on CPU startup:

```bash
MEDRESEARCH_DEFAULT_APPROACH=hospital_ensemble MEDRESEARCH_VISION_ENSEMBLE=1 \
  .venv-vision/bin/uvicorn medresearch_platform.interfaces.api:create_app --factory --host 127.0.0.1 --port 8000
```
