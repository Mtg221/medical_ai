# Google Colab & Kaggle GPU Training Notebooks (100% From Scratch)

These notebooks run our PyTorch training pipelines on **Google Colab (free T4 GPU)** or **Kaggle (free T4 x2 / P100 GPU)** with **`pretrained_weights_used = False`** (strictly random initialization, zero external pretrained weights). Both notebooks automatically detect whether they are running in Google Colab (`/content`) or Kaggle (`/kaggle/working`) and trigger automatic `.zip` download in Colab when finished.

> **Hospital Privacy & Compliance Rule:** ONLY train on public / open-license / synthetic datasets in the cloud. **Never upload real hospital patient notes, clinical photos, or PHI/PII** to Colab or Kaggle. Train on public data here, download the signed `.pt` checkpoints + SHA-256 manifests, and run inference (or private fine-tuning) strictly on-premise inside the hospital.

---

## Notebooks

1. **`kaggle_gpu_nlp_hospital.ipynb`** — Trains a **3-seed Unified Hospital NLP Model** (`PyTorchCharCNNBiLSTMCRF` with Exact Partial-Label Marginal CRF, `hidden_dim=128`, `epochs=20`) jointly on:
   - `SYNTH-MED-V3.1` (EN + FR, 14 clinical entity types)
   - `MACCROBAT2020` (EN real case reports, `CC-BY-4.0`)
   - `PxCorpus` (FR real clinical prescriptions, `CC-BY-4.0`)
   - `E3C Layer 1` (EN + FR clinical narratives, `CC-BY-NC-4.0` — set `PERMISSIVE_ONLY = True` to omit for commercial builds)
   - `BC5CDR` (EN biomedical Disease + Chemical, `Public-Domain`)
   - `NCBI Disease` (EN biomedical Disease, `Public-Domain`)
   - From-scratch E3C Layer 3 self-supervised EN/FR semantic clusters (`models/e3c_self_supervised_en_fr.npz`)
   - Output artifact: `hospital_nlp_models_gpu.zip`

2. **`kaggle_gpu_vision_hospital.ipynb`** — Trains a **3-seed `224x224` Dermatology Vision Ensemble** (`PyTorchDermResNet`, CUDA FP16 mixed precision `AMP`, `body_site_dropout=0.40`, `epochs=25`) using the locked patient/lesion-grouped v3 splits (`configs/vision/val_v3.txt`, `configs/vision/locked_test_v3.txt`), fits `val_v3`-only per-modality calibration temperatures, and runs the biopsy-confirmed (`histopathology`) audit.
   - Output artifact: `hospital_vision_models_gpu.zip`

---

## How to Run on Google Colab (Recommended, No SMS Needed)

1. Go to [https://colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → select `notebooks/kaggle_gpu_nlp_hospital.ipynb` (or `notebooks/kaggle_gpu_vision_hospital.ipynb`).
2. Click **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**.
3. *(Vision notebook only)* Either add `DERM_CLOUDFRONT_USER` and `DERM_CLOUDFRONT_PASS` under the **Secrets (key icon)** in Colab's left sidebar (and toggle Notebook access ON), or enter them in the hidden prompt when Cell 3 runs.
4. Click **Runtime → Run all**. When finished, Colab automatically downloads `hospital_nlp_models_gpu.zip` / `hospital_vision_models_gpu.zip`. Extract into your local repo root:

```bash
unzip -o hospital_nlp_models_gpu.zip -d .
unzip -o hospital_vision_models_gpu.zip -d .
```

## How to Run on Kaggle (30 hrs/week T4 x2 / P100)

1. Verify your phone number once in [Kaggle Account Settings](https://www.kaggle.com/settings) to unlock **GPU T4 x2 / P100** and **Internet ON**.
2. Import either `.ipynb` from `notebooks/`, enable **Accelerator → GPU T4 x2** and **Internet → ON** (and add `DERM_CLOUDFRONT_USER` / `DERM_CLOUDFRONT_PASS` under **Add-ons → Secrets** for Vision), then click **Run All**.

---

## Serving the GPU-Trained Checkpoints Locally on CPU

The live server automatically detects all 3 NLP seeds (`models/hospital_char_cnn_bilstm_crf_seed{1,2,3}`), all 3 Vision seeds (`models/derm_cnn_bsdrop40_seed1`, `models/derm_cnn_seed2`, `models/derm_cnn_seed3`), and the calibrated temperatures (`models/derm_ensemble_calibration.json`) on CPU startup:

```bash
MEDRESEARCH_DEFAULT_APPROACH=hospital_ensemble MEDRESEARCH_VISION_ENSEMBLE=1 \
  .venv-vision/bin/uvicorn medresearch_platform.interfaces.api:create_app --factory --host 127.0.0.1 --port 8000
```
