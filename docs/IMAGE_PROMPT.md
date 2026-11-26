# IMAGE PROMPT — Make the platform understand skin pictures (derm-corpus)

> All rules of `docs/MASTER_PROMPT.md` and `MEDICAL_AI_FINAL_PROMPT.md` still apply (from scratch, local, EN + FR,
> honesty, locked test sets, privacy, test before fix, never delete history).
> **Order:** finish Phase 1 (product safety) and Phase 2 (privacy) of the final prompt first — image uploads reuse
> those guards — then do this prompt. **Do only what this file asks.**

---

## PART A — What exists today and what the owner decided

### A.1 The data: the private "derm-corpus" (`https://d38rablk9v12ns.cloudfront.net/`)
- Protected with HTTP Basic auth (realm `derm-corpus`). **The owner puts the credentials in `.env` himself:**
  `DERM_CLOUDFRONT_USER=…` / `DERM_CLOUDFRONT_PASS=…` (`.env` is git-ignored). The agent **never** asks for, prints,
  logs, commits or copies them, and never puts them in URLs.
- From `extras/dermatology/catalog.py`, each image comes with metadata (gzip/plain JSON pages):
  `i` image id · `d` source dataset · `mg` benign/malignant · `dx` raw diagnosis · `st` Fitzpatrick skin tone I-VI ·
  `bs` body site · `cf` how the diagnosis was confirmed · `ok` quality flag · `lc` licence · `mk` case/lesion key ·
  `t` thumbnail path · `f` full-size path. Modalities: `clinical` (phone/camera photo), `dermoscopic`, `tbp_tile`
  (total-body-photography tile).
- **First job of the agent (Step 1):** explore it (with the owner's env credentials) and write
  `docs/data/DERM_CORPUS_CARD.md`: how many images per modality / source dataset / licence / benign-malignant /
  diagnosis / skin tone / body site / confirmation; image sizes; duplicates; how many lesions and patients (`mk`);
  which licences allow research use and which forbid redistribution or commercial use.

### A.2 The current code (`extras/dermatology/`)
- `vision_classifier.py`: a toy pure-NumPy CNN (one 3×3 conv layer, 6 channels, tanh, pooling) for benign vs malignant.
- `pipeline.py` still imports `medresearch_platform.dermatology.*`, which no longer exists after the move to `extras/`
  (broken imports). Default paths `/tmp/derm_meta`, `var/cache/derm_thumbs`.
- Not connected to the API, the frontend, the patient record or the reports.

### A.3 The owner's decisions (2026-11-16)
1. **Output = full analysis** of a skin picture: benign-vs-malignant risk, top-3 likely diagnoses, picture type
   (clinical / dermoscopic / body tile / **not a skin picture**), image-quality check, a heat map showing where the
   model looked, and "needs review" whenever unsure.
2. **Library = PyTorch CPU, random initialisation** for the image models (still 100 % from scratch: no pretrained
   weights, no ImageNet, no torchvision `weights=`). The text models stay NumPy. Record it in
   `DECISIONS/ADR-0010-pytorch-cpu-for-vision.md`.
   - Owner's PC is an **Intel Mac (x86_64)**: PyTorch's last wheel for it is **torch 2.2.2 on Python ≤ 3.12**. Use a
     Python 3.12 environment for the project (or a separate `.venv-vision`), pin `torch==2.2.2` there, newer CPU wheels
     on Windows/Linux. Put it in an optional extra `pip install -e ".[vision]"`; the platform must still start without it
     (image features then show "vision module not installed"). Document both setups in the README.
3. **One patient record:** image findings live next to the text facts in the same workspace, timeline and EN/FR report.

---

## PART B — Rules specific to images

1. **Safety wording.** The model gives a *research risk estimate*, never "diagnosis", "cancer detected" or advice.
   Every result shows: "Research tool — not a diagnosis. A dermatologist must examine the lesion." (EN/FR).
   High-risk or uncertain results are always `needs_review`.
2. **Abstain instead of guessing:** not a skin picture, poor quality (blur, too dark, too small, overexposed), or
   out-of-distribution → no risk score, a clear reason instead.
3. **Licences and privacy:** never commit corpus images, thumbnails or the metadata dump to git (`var/cache/derm/`,
   git-ignored). Respect each image's `lc`: exclude images whose licence forbids the use; keep the licence in every
   record. Screenshots and docs use only CC0/CC-BY images with attribution, or synthetic images. Clinical photos may
   show faces/tattoos: never show a full-size image outside the authenticated app; thumbnails only in lists.
4. **No leakage:** split by lesion/patient (`mk`), never by image — the same lesion photographed twice must not be in
   train and test. Deduplicate near-identical images (perceptual hash) across splits. Create a **locked test** split once,
   log every use in `docs/results/TEST_SET_LOG.md`.
5. **Fairness is a result, not a footnote:** every metric is reported per Fitzpatrick skin tone, modality, source dataset
   and body site, with the number of images; small groups (< 30) are flagged "too few to conclude".
6. **Honesty:** report what the model cannot do (e.g. few dark-skin malignant examples). Never tune on the locked test.

---

## PART C — Steps (in order)

- **Step 1**: Data access, card, splits (`medextract derm sync`, `docs/data/DERM_CORPUS_CARD.md`, `configs/vision/dx_taxonomy.yaml`, leak-free split audit)
- **Step 2**: Picture triage router & image-quality check (`skin_clinical`, `skin_dermoscopic`, `skin_tbp_tile`, `document_scan`, `other`)
- **Step 3**: Skin models (Classical baseline, Pure-NumPy CNN baseline, PyTorch CPU multi-task ResNet CNN from random init, temperature calibration, Grad-CAM heat maps, abstention/OOD)
- **Step 4**: Output contract (`ImageFinding`), `SAME_LESION` text-image linking, and human review persistence
- **Step 5**: FastAPI endpoints (`/api/v1/images/upload`, `/api/v1/images/{id}`, `/api/v1/images/{id}/heatmap`, `/api/v1/patients/{id}/record`, `/api/v1/results/vision/*`)
- **Step 6**: React/TS bilingual frontend (Unified drop zone, Image viewer with heat-map slider, Patient timeline, EN/FR reports, Vision benchmark & subgroup fairness tabs)
- **Step 7**: Research report chapter, model card (`docs/model-cards/MODEL_CARD_DERM_CNN_V1.md`), and acceptance test suite
