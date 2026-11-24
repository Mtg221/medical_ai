# Journal de Travail — Mission Complétion, Validation & Reproductibilité (`chore/completion-audit`)

**Branche :** `chore/completion-audit` (créée depuis `main` @ `6b8bd3cfb80ae452c0222c3cb10ba8031f393e0c`)  
**Environnement matériel :** macOS Darwin `x86_64`, Intel Core i7-9750H @ 2.60GHz, 32 Go RAM, CPU-only, Python 3.12.16 (`.venv-vision`), Node v24.18.0.  
**Règles absolues :** 100 % from-scratch, aucun nouveau modèle sans preuve préalable, splits historiques gelés, aucune fabrication de données humaines, aucun `git push`.

---

## Phase 0 — Inventaire & Baseline (`2026-11-24T19:09Z–19:24Z`)

### 1. Commandes exécutées & résultats bruts

| Vérification | Commande | Sortie brute | Résultat | Statut |
|---|---|---|---|---|
| Environnement & Git | `git status -sb && git rev-parse HEAD && uname -a && .venv-vision/bin/python --version` | `docs/results/raw/phase0_env.txt` | `main` propre @ `6b8bd3c`, i7-9750H, 32 Go RAM, Python 3.12.16, Node v24.18.0 | **Passed** |
| Lint (`ruff`) | `.venv-vision/bin/ruff check medresearch_platform scripts tests` | `docs/results/raw/phase0_static_checks.txt` | `All checks passed!` (`RUFF_EXIT=0`) | **Passed** |
| Typage (`mypy`) | `.venv-vision/bin/mypy medresearch_platform` | `docs/results/raw/phase0_static_checks.txt` | `10 errors in 6 files (checked 99 source files)` (`MYPY_EXIT=1`) | **Failed** |
| Tests Python (`pytest`) | `.venv-vision/bin/pytest -q` | `docs/results/raw/phase0_pytest.txt` | `1 failed, 416 passed, 5 xfailed, 4 xpassed, 20 warnings in 699.00s` (`PYTEST_EXIT=1`) | **Failed** |
| Build + Tests Frontend | `npm --prefix frontend run build && npm --prefix frontend test -- --run` | `docs/results/raw/phase0_frontend.txt` | Build Vite `BUILD_EXIT=0`; Vitest `2 files, 13 passed` (`TEST_EXIT=0`) | **Passed** |

### 2. Traçage réel des flux de données dans le code

1. **Flux Document (upload → stockage → OCR → NLP → DB → export) :**
   - `POST /api/v1/documents/upload` (`interfaces/routers/documents.py`) → `MedicalResearchPlatformService.ingest_document` (`workflows/pipeline.py:274`).
   - Validation taille/format/sécurité : `validate_upload_payload` + `inspect_untrusted_bytes` (`documents/parsers.py`, `security/safety.py`).
   - Stockage immuable du binaire original dans `storage_dir/originals/<sha256>.<ext>` + déduplication par `(institution_id, content_sha256)`.
   - Routage image : si `PNG/JPEG/TIFF/WEBP`, `vision_engine.router.triage_image` peut router vers `skin_vision_pipeline` ; sinon `parse_document_bytes` → `ocr_engine` (Tesseract + prétraitement binaire `{0, 255}` CLAHE si faible contraste).
   - Extraction NLP : `ExtractionService.extract` (`serving/extractor.py`) → `hybrid_v2` (5 étapes : sectioning, NER, ConText 3 axes + relations, temporalité, validation) ou `hospital_ensemble` (18 modèles from-scratch + CRF v2).
   - Persistance SQLite/Postgres : `documents`, `document_pages`, `document_sections`, `extracted_entities` (avec `certainty`/`temporality`/`experiencer`), `extracted_relations`, `longitudinal_events`, `audit_logs` (chaîne SHA-256).
   - Export : `reports/generator.py` (MD/HTML/PDF) et `workflows/review.py` (JSONL/BRAT).
2. **Flux PHI (détection → masquage → stockage) :**
   - `deidentify_clinical_text` (`security/privacy.py:120`) appelle `detect_person_spans` (`security/person_age_detector.py:287`) puis `_PII_PATTERNS` (`DOB`, `MEDICAL_RECORD_NUMBER`, `FRENCH_NIR`, `SSN`, `EMAIL`, `PHONE`, `ADDRESS`), remplace les spans non chevauchants par `[<TYPE>_REDACTED]`, et applique un décalage de dates ISO (`date_shift_days`).
   - **Constat critique #1 (`workflows/pipeline.py:489`) :** `deidentify_clinical_text(combined_text[:4000], date_shift_days=shift_days)` ne traite que les 4 000 premiers caractères pour `deidentified_preview`, alors que `document_pages.raw_text` stocke le texte brut non masqué (comportement à documenter explicitement dans la gouvernance sécurité et vérifier qu'aucune fuite PHI ne passe dans les logs/erreurs/rapports).
   - **Constat critique #2 (`security/privacy.py:12`) :** `detect_age_spans` (`person_age_detector.py:351`) n'est **pas** connecté à `deidentify_clinical_text` (uniquement utilisé en pré-masquage NER dans `stage2_ner.py`).
   - **Constat critique #3 (`scripts/eval_redaction_human.py:130`) :** la précision est calculée comme `tp / (tp + fp_non_phi)` — mélange d'un comptage **par entité PHI** (`tp`) et d'un comptage **par phrase non-PHI** (`fp_non_phi`), ignorant totalement les faux positifs intra-phrase.

### 3. Liste priorisée des défauts confirmés (à traiter dans les phases 1–11)

1. **[P0 — Phase 2] Outillage 80 notes humaines incomplet (`scripts/import_label_studio_realistic.py`, `scripts/eval_realistic.py`) :**
   - Défauts silencieux sur `language="en"`, `doc_type="consultation"`, `document_date="2025-01-01"`.
   - `annotations[0]` pris comme gold même en cas de désaccord entre annotateurs (aucune adjudication exigée).
   - Aucune validation stricte des quotas (80 docs uniques, 40 EN / 40 FR, 6 types de documents, >= 16 double-annotés >= 20 %), ni manifeste de gel SHA-256 / provenance.
2. **[P0 — Phase 3] Bug de métrique précision PHI + détection d'âge non câblée (`scripts/eval_redaction_human.py`, `security/privacy.py`) :**
   - Remplacer par des métriques span-level exactes (P/R/F1 par langue et par catégorie) distinctes du taux de faux positifs par phrase.
   - Connecter proprement la détection d'âge HIPAA (`>89 ans` / optionnellement tous âges) sans faux positifs sur durées (`14 days`, `7 jours`), dates ou mesures.
3. **[P1 — Phase 4/8] Échec baseline `test_conftest_helper_builds_valid_loadable_models` (`serving/extractor.py`) :**
   - Quand des répertoires isolés `model_dir` / `classical_crf_dir` sont passés à `ExtractionService`, le chargeur charge quand même `models/hospital_char_cnn_bilstm_crf_seed1..18` depuis la racine du dépôt (`len(loaded) == 3 != 2`).
4. **[P1 — Phase 4] Audit NLP complet sur `val.jsonl` + 7 phrases d'assertion EN/FR + 9 types de relations + séparation extraction/normalisation temporelle.**
5. **[P2 — Phase 9] 10 erreurs `mypy` dans 6 fichiers (`ocr_engine.py`, `neural_seq.py`, `classical_ml.py`, `stage3_context_relations.py`, `stage2_ner.py`, `pipeline.py`) + warnings PyTorch/tarfile.**
6. **[P2 — Phase 10] Finalisation de la GitHub Release `v0.2.0-research-artifacts` (vérifier les assets uploadés après le timeout HTTP 408 précédent).**

**Décision Phase 0 :** Baseline complète enregistrée. Passage immédiat à la **Phase 1 — Intégrité des données**.

---

## Phase 1 — Intégrité des données (`2026-11-24T19:24Z–19:26Z`)

### Commandes & sortie brute
- Commande : audit Python lecture seule des hash SHA-256 (`LOCK_MANIFEST.json`), fuites inter-splits (`SYNTH-MED-V3.1`, `open_clinical`, `open_clinical_plus_biomed`), `TEST_SET_LOG.md`, `v3_leakage_audit.json`, `v4_dedup_audit.json`, et taxonomie de provenance.
- Sortie brute : `docs/results/raw/phase1_data_integrity.txt`
- Statut : **Passed** (splits historiques non modifiés)

### Constats (fichiers exacts)
1. **`data/locked/synth_med_v3_1/LOCK_MANIFEST.json` (`5f5e3fe`) :** 4/4 splits gelés (`val`, `test_iid`, `test_unseen_terms`, `test_unseen_templates`) vérifiés SHA-256 (`match=True`) en `locked/` et `processed/v3_1/`. `0` chevauchement `doc_id`, `0` `patient_id`, `0` texte normalisé sur les 10 paires de splits.
2. **`data/processed/open_clinical/` et `open_clinical_plus_biomed/` (contamination documentée, splits gelés non touchés) :** `0` chevauchement `doc_id`/`patient_id`, MAIS **73 / 443 lignes de `open_clinical/test.jsonl`** (64 textes normalisés uniques), **100 % issus de `pxcorpus`** (prescriptions orales FR identiques prononcées par différents locuteurs), apparaissent textuellement dans `train.jsonl` (`0` fuite sur `maccrobat2020`, `e3c_layer1`, `bc5cdr`, `ncbi_disease`). **Protocole de remplacement recommandé (sans modifier l'existant) :** grouper par `(patient_id, sha256(normalized_text))` lors de la création d'un éventuel `open_clinical_v2`.
3. **`docs/results/TEST_SET_LOG.md` :** 51 évaluations historiques enregistrées ; splits historiques figés comme historique de développement.
4. **`docs/results/vision/v3_leakage_audit.json` & `v4_dedup_audit.json` & `configs/vision/locked_test_v4.txt` :** `0` chevauchement ID/lésion ; 13 doublons SHA-256 exacts retirés dans `locked_test_v4` (`n=1,611`, `0` doublon exact). Proximité `dHash <= 4` = `25.49 %` (`42.57 %` dermoscopie, `19.69 %` clinique, `0.86 %` TBP) — limite connue et documentée.
5. **Taxonomie de provenance vérifiée :**
   - Synthétique : `SYNTH-MED-V3.1` (`synthetic_v3_1`), `v1` (`synthetic_v2`), `v3` (`synthetic_v3`).
   - Proxy LLM (non humain) : `data/processed/realistic_llm/eval.jsonl` (`llm_authored_free_text_single_annotator`, `owner_written: false`).
   - Benchmarks publics : `MACCROBAT2020`, `PxCorpus`, `E3C`, `BC5CDR`, `NCBI Disease`, `ISIC/SLICE-3D/HAM10000`.
   - Humain écrit par le propriétaire (80 notes + 60 phrases PHI) : **Absent / En attente du propriétaire** (jamais fabriqué).

**Décision Phase 1 :** Intégrité vérifiée sans altérer aucun split gelé. Passage à la **Phase 2 — Outillage d'évaluation humaine (80 notes)**.

---

## Phase 2 — Outillage d'évaluation humaine (80 notes) (`2026-11-24T19:25Z–19:29Z`)

### Constats initiaux (avant correctif)
- `scripts/import_label_studio_realistic.py:41-43` : défauts silencieux (`language="en"`, `doc_type="consultation"`, `document_date="2025-01-01"`).
- `scripts/import_label_studio_realistic.py:188-194` : `annotations[0]` utilisé silencieusement comme gold même en cas de désaccord entre annotateurs.
- `scripts/import_label_studio_realistic.py` & `scripts/eval_realistic.py` : aucune vérification stricte des quotas (80 docs uniques, 40 EN / 40 FR, 6 `DOC_TYPES`, >= 16 double-annotés >= 20 %), ni manifeste SHA-256 de gel.

### Décision & modifications
1. **`scripts/import_label_studio_realistic.py` :**
   - Ajout de la validation explicite obligatoire des métadonnées (`doc_id`, `language in {"en","fr"}`, `doc_type in DOC_TYPES`, `document_date` ISO valide) sans défaut silencieux en mode strict (actif par défaut en CLI).
   - Ajout de `_select_gold_annotation_for_task` : refuse tout désaccord non adjudiqué (`ValueError`) et n'accepte que le consensus exact ou une annotation explicitement marquée `ground_truth=true` / `adjudicated=true`. L'IAA reste calculé sur les annotations indépendantes.
   - Ajout de `validate_human_realistic_protocol` (80 docs uniques, 0 doublon `doc_id`/texte normalisé, 40 EN / 40 FR, 6 `DOC_TYPES`, >= 16 double-annotés) et `build_freeze_manifest` (manifeste SHA-256 + provenance + version).
2. **`scripts/eval_realistic.py` :**
   - Ajout de `verify_owner_freeze_manifest` (vérifie SHA-256, `protocol_validated`, quotas 80/40/40/6/>=16) obligatoire lorsque `--owner-written` est passé ; enregistrement explicite de `git_commit`, `manifest_sha256`, `dataset_version`, `post_hoc`.
3. **`tests/test_human_realistic_protocol_strict.py` :** 8 nouveaux tests (paramétrés) prouvant le rejet des métadonnées manquantes, désaccords non adjudiqués, sets incomplets/déséquilibrés, doublons, et altération post-gel.

### Commande & résultat brut
- Commande : `.venv-vision/bin/pytest -v tests/test_human_realistic_protocol_strict.py tests/test_iaa_and_realistic_relations.py tests/test_realistic_import_and_labels.py`
- Sortie brute : `docs/results/raw/phase2_human_eval_tooling.txt`
- Résultat : **`14 passed in 0.22s`** (`EXIT=0`)
- Statut : **Passed** (notes humaines réelles : toujours en attente du propriétaire, jamais fabriquées)

---

## Phase 3 — Audit et correctifs PHI (`2026-11-24T19:29Z–19:33Z`)

### Constats initiaux (avant correctif)
1. **`scripts/eval_redaction_human.py:130` :** bug de précision confirmé — `tp / (tp + fp_non_phi)` mélangeait un décompte d'entités PHI (`tp`) avec un décompte de phrases non-PHI (`fp_non_phi`), rendant invisibles les faux positifs intra-phrase.
2. **`medresearch_platform/security/privacy.py:12` :** `detect_age_spans` n'était pas importé ni appelé par `deidentify_clinical_text`.
3. **`medresearch_platform/security/person_age_detector.py:347` :** `_DURATION_PREFIX_RE` ne couvrait pas les préfixes `after`, `après`, `every`, `tous les`, `toutes les`, `within`, `sous` et ne filtrait pas les valeurs `> 125` ans.

### Décision & modifications
1. **`medresearch_platform/security/person_age_detector.py` :** ajout de `detect_age_spans_with_values` avec garde élargie contre les durées/fréquences (`for 14 days`, `pendant 7 jours`, `after 2 years`, `après 3 ans`, `every 5 years`, `tous les 2 ans`) et rejet des âges impossibles (`> 125`).
2. **`medresearch_platform/security/privacy.py` :** câblage direct de `detect_age_spans_with_values` dans `deidentify_clinical_text` (`redact_ages_over_89=True` par défaut selon HIPAA Safe Harbor §164.514(b)(2)(i)(C) → `[AGE_OVER_89_REDACTED]`, et option `redact_all_ages=False` → `[AGE_REDACTED]`).
3. **`scripts/eval_redaction_human.py` :** calcul rigoureux de `precision` / `recall` / `f1` au niveau **span** (overlap + exact-span), `strict_full_token_recall`, détection des `false_positive_spans` y compris intra-phrase, et taux `non_phi_sentence_fp_rate` strictement séparé, ventilés par langue (`en`/`fr`) et par catégorie.
4. **`tests/test_phase3_phi_metrics_and_age.py` :** tests prouvant le masquage HIPAA `>89 ans`, l'absence de faux positifs sur durées/dates/mesures/éponymes, et la détection des FP intra-phrase.

### Commande & résultat brut
- Commande : `.venv-vision/bin/pytest -v tests/test_phase3_phi_metrics_and_age.py tests/test_step7_w10_p10_redaction.py tests/test_security_privacy_audit.py tests/test_p19_step4_y6_text_generalization.py tests/test_step1_p10_q1_to_q7.py tests/test_iaa_and_realistic_relations.py`
- Sortie brute : `docs/results/raw/phase3_phi_eval.txt`
- Résultat : **`19 passed in 21.75s`** (`PYTEST_EXIT=0`)
- Statut : **Passed** (jeu humain 60 phrases PHI : en attente du propriétaire)

---

## Phase 4 — Audit NLP (entités, relations, assertions, temporalité — sans nouvelle architecture) (`2026-11-24T19:33Z–19:42Z`)

### Constats (erreurs reproductibles avant correctif)
1. **`medresearch_platform/serving/extractor.py:101-120` :** lors d'un appel isolé `ExtractionService(model_dir=..., classical_crf_dir=...)`, `_discover_torch_ner_dirs` scannait encore `models/` à la racine du dépôt (`len(loaded) == 3 != 2`, cause de l'échec baseline `test_conftest_helper_builds_valid_loadable_models`).
2. **`medresearch_platform/approaches/context_negex.py:76-102` :** sur les 7 phrases d'assertion EN + 7 FR, `"Le scanner actuel ne montre pas la lésion."` échouait (`confirmed` au lieu de `absent`) faute de cues FR `"ne montre pas "` / `"ne retrouve pas "` / `"n'objective pas "` (`docs/results/raw/phase4_nlp_audit_before.txt`).
3. **`medresearch_platform/hybrid/stage3_context_relations.py:404-437` :** `PiecewiseCNNRelationClassifier.predict_relations` ne vérifiait ni les frontières de phrases (`. ` / `; `) ni la présence d'une tête intermédiaire de même type, produisant des liaisons croisées multi-médicaments (`metformin -> 10 mg`, `lisinopril -> BID`) sur EN et FR (`docs/results/raw/phase4_nlp_audit_before.txt`).
4. **Limitation connue documentée (sans nouveau modèle) :** sur phrases libres courtes hors domaine (`"Previous scan showed a lesion."`, `"Current scan does not show the lesion."`), les assertions 3 axes sont exactes (14/14 EN+FR), mais le typage NER de surface sur `hybrid_v2` seul peut encore mal typer des mots génériques (`lesion`, `scan does`) ; à suivre sur le futur jeu humain gelé de 80 notes.

### Décision & modifications (aucune nouvelle architecture)
1. **`medresearch_platform/serving/extractor.py` :** ancrage de `self.torch_ner_dir` sur `self.model_dir.parent` quand `model_dir` est fourni sans `MEDRESEARCH_TORCH_NER_DIR`.
2. **`medresearch_platform/approaches/context_negex.py` :** ajout des cues de négation clinique FR (`ne montre pas `, `ne retrouve pas `, `n'objective pas `, `ne met pas en évidence `, `ne révèle pas `, `n'a pas montré `, `ne présente pas `, `ne montre aucun`, `ne retrouve aucun`).
3. **`medresearch_platform/hybrid/stage3_context_relations.py` :** garde frontière de phrase (`[.!?;]\s+`), garde tête intermédiaire de même type (`_has_intervening_same_type_head`), et déduplication au plus proche prédécesseur pour les 5 relations fonctionnelles (`HAS_DOSAGE`, `HAS_FREQUENCY`, `HAS_ROUTE`, `HAS_VALUE`, `HAS_UNIT`).
4. **`tests/test_phase4_nlp_assertions_relations_temporal.py` :** 16 tests de régression (14 phrases d'assertion EN+FR, non-croisement multi-entités/frontières de phrases, ambiguïté locale temporelle, dates impossibles, drapeau `ocr_repaired`).

### Résultats avant / après sur `val.jsonl` (dev uniquement, `n=160`)
- **Relations (dev `val.jsonl`) :**
  - Avant (`docs/results/raw/phase4_dev_before.txt`) : P `0.9400`, R `0.9474`, **F1 `0.9437`** (`TP=972, FP=62, FN=54`)
  - Après (`docs/results/raw/phase4_dev_after.txt`) : P `0.9765` (**+3.65 pt**), R `0.9308`, **F1 `0.9531`** (**+0.94 pt**), FP réduits de **63 %** (`62 → 23`)
- **Assertions (dev `val.jsonl`) :** Accuracy `0.9488`, Macro F1 `0.7988` (préservé) + **14/14** phrases cibles EN+FR exactes.
- **Temporalité (dev `val.jsonl`) :** Extraction F1 `0.9937`, Normalisation Accuracy `1.0000` (mesurées séparément) ; `03/04/2025` EN=`2025-03-04` vs FR=`2025-04-03` (`ambiguous=True, needs_review=True`), `2025-02-30` → `None`, `2O25-03-1O` → `ocr_repaired=True, needs_review=True`.
- **Tests Phase 4 (`docs/results/raw/phase4_pytest.txt`) :** **`53 passed in 18.80s`** (`PYTEST_EXIT=0`), incluant `test_conftest_helper_builds_valid_loadable_models`.
- Statut : **Passed**

---

## Phase 5 — Audit OCR & propagation NLP (`2026-11-24T19:42Z–19:44Z`)

### Commandes & sorties brutes
- Tests OCR : `.venv-vision/bin/pytest -v tests/*ocr*.py` → `docs/results/raw/phase5_ocr_pytest.txt` (**`17 passed in 30.08s`**, `PYTEST_EXIT=0`)
- Synthèse benchmark 120 pages (`SYNTH-MED-V3.1`, `test_iid`) : `docs/results/raw/phase5_ocr_eval.txt` (lu depuis `docs/results/v3_1_ocr_full_benchmark.json`).

### Constats vérifiés
1. **Prétraitement & robustesse (`ocr_engine.py`, `test_phase5_ocr_robustness.py`, `test_ocr_layout_segmentation.py`) :** invariant binaire `{0, 255}` préservé, sauvetage CLAHE faible contraste (`p99 - p0.1 < 110.0`), séparation 2 colonnes (`detect_column_split`), alignement span OCR→référence 100 % exact sur texte identique et robuste au bruit.
2. **Métriques 120 pages (`v3_1_ocr_full_benchmark.json`) :**
   - Global : CER `0.0862`, WER `0.2112`, `classical_crf` Strict F1 `0.9978 → 0.9128` (`-0.0850`), `hybrid` `0.9937 → 0.8799` (`-0.1139`).
   - Par langue : **EN** CER `0.0824` / WER `0.1969` (`hybrid` `-0.0942`), **FR** CER `0.0899` / WER `0.2255` (`hybrid` `-0.1341`).
   - **Contradiction doc détectée (à corriger en Phase 11) :** `PROJECT_STATUS.md:24` citait `en 0.0725, fr 0.0999` au lieu de `en 0.0824, fr 0.0899` dans `v3_1_ocr_full_benchmark.json`.
   - Dégradation réaliste par palier : Tier 1 (`0.0046` CER, `-0.0064` F1) jusqu'à Tier 5 tableaux/multi-polices (`0.3673` CER, `-0.2442` F1) et Tier 6 flou 150 DPI (`0.1027` CER, `-0.3015` F1).
- Statut : **Passed**

---

## Phase 6 — Audit Vision dermatologie (piste séparée) (`2026-11-24T19:44Z–19:49Z`)

### Commandes & sortie brute
- Commande : `.venv-vision/bin/pytest -v tests/*vision*.py tests/test_phase3_reproducibility_and_lazy_vision.py` + inspection `vision_9seed_locked_test_v4_benchmark.json` et gouvernance des jeux externes.
- Sortie brute : `docs/results/raw/phase6_vision_audit.txt`
- Résultat : **`23 passed in 271.07s`** (`PYTEST_EXIT=0`)
- Statut : **Passed**

### Constats vérifiés
1. **Métriques par modalité sur `locked_test_v4` (`n=1,611`, `0` doublon SHA-256 exact vs train, `pretrained_weights_used: false`) :**
   - Global calibré : AUROC `0.9348`, Sens@90%Spec `0.7962`, ECE `0.0333`, Brier `0.1015`.
   - Dermoscopie (`n=690`) : AUROC `0.9727`, Sens@90%Spec `0.9291`, ECE `0.0403`, Brier `0.0662`.
   - Clinique smartphone (`n=571`) : AUROC `0.8805`, Sens@90%Spec `0.5826`, ECE `0.0339`, Brier `0.1295`.
   - TBP tile (`n=350`, auto-abstenu en inférence clinique) : AUROC `0.8768`, Sens@90%Spec `0.7000`, ECE `0.1017`, Brier `0.1256`.
   - Sous-ensemble basse incertitude (`std <= 0.12`, `n=1,263`) : AUROC `0.9483`, Sens@90%Spec `0.8347`, ECE `0.0322`, Brier `0.0874`.
2. **Sécurité & intégrité vision (`test_vision_step0_safety.py`, `test_vision_step2_data.py`, `test_vision_step3.py`) :** rejet 503 sur modèle manquant/corrompu, aucun fallback cross-patient, date de capture manquante marquée `null` + avertissement, stripping EXIF, Grad-CAM testé (avec limites Adebayo documentées dans la model card `docs/MODEL_CARD_DERM_CNN_V1.md`).
3. **Jeux externes (PAD-UFES-20, Fitzpatrick17k, DDI) :** outillage d'ingestion sans fuite (`scripts/ingest_external_clinical_derm.py`) et protocole shadow-mode (`docs/SHADOW_MODE_CLINICAL_PROTOCOL.md`, `scripts/eval_shadow_mode.py`) prêts ; **aucun téléchargement non autorisé effectué** (licences/accès institutionnels requis).

---

## Phase 7 — Sécurité & vie privée (`2026-11-24T19:49Z–19:51Z`)

### Commandes & sortie brute
- Commande : `.venv-vision/bin/pytest -v tests/test_security_privacy_audit.py tests/test_phase1_p1_to_p9.py tests/test_api_and_cli.py` + vérification `SQLiteLoginRateLimiter` et en-têtes CSP/`nosniff` dans `interfaces/`.
- Sortie brute : `docs/results/raw/phase7_security_audit.txt`
- Résultat : **`19 passed in 96.30s`** (`PYTEST_EXIT=0`)
- Statut : **Passed**

### Constats vérifiés
1. **Auth, RBAC, Isolation, Révocation & Rate-limit :** Argon2id + HMAC-SHA256 `jti`, `401` sur toutes les routes `/api/v1/*` protégées (`test_p8_all_api_v1_routes_require_bearer_token_401`), isolation inter-institutions `403` (`test_full_api_lifecycle_rbac_and_institution_isolation`), révocation `POST /api/v1/auth/logout`, throttling brute-force `429` persistant via `SQLiteLoginRateLimiter` (`routers/system.py:30`, zéro service externe requis).
2. **Uploads & Sécurité web :** allowlist MIME/extension (`415` sur `.exe`), limites de taille (`413`), garde path-traversal/zip-bomb (`inspect_untrusted_bytes`), en-têtes `Content-Security-Policy` (`frame-ancestors 'none'; base-uri 'none'`), `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` (`interfaces/api.py:90-122`).
3. **Chaîne d'audit & désérialisation :** chaîne SHA-256 infalsifiable (`verify_audit_chain`), zéro `pickle.load`/`joblib.load`, `weights_only=True` sur tous les `torch.load` (`test_no_unsafe_deserialization_calls_in_platform`).

---

## Phase 8 — Backend, Frontend & DB (`2026-11-24T19:51Z–19:52Z`)

### Commandes & sortie brute
- Commande : migration SQLite depuis base vide + idempotence, `pytest` jobs/reviews/reports, et scan mock-data frontend.
- Sortie brute : `docs/results/raw/phase8_backend_frontend_db.txt`
- Résultat : migration `001_initial_schema` idempotente (26 tables créées, colonnes 3 axes `certainty`/`temporality`/`experiencer` présentes), **`5 passed in 11.57s`** (`PYTEST_EXIT=0`), `ZERO_MOCK_CONSTANTS_IN_PROD_UI`.
- Statut : **Passed**

---

## Phase 9 — Qualité code, dépendances & performance CPU (`2026-11-24T19:52Z–19:54Z`)

### Modifications
1. **Correction des 10 erreurs `mypy` dans 6 fichiers :**
   - `medresearch_platform/documents/ocr_engine.py:177, 339, 340` : `value=(255,)` pour correspondre à l'overload `Sequence[float]` de `cv2.copyMakeBorder`.
   - `neural_seq.py`, `classical_ml.py`, `stage3_context_relations.py`, `stage2_ner.py`, `pipeline.py` : remplacement de `rng.shuffle(list[...])` par permutation indexée typée `rng.permutation(len(...))`.
2. **Correction de 2 warnings :**
   - `medresearch_platform/vision/cnn_torch.py:115-118` : `np.array(..., copy=True)` avant `torch.from_numpy` (supprime le warning `non-writable array`).
   - `scripts/package_release_artifacts.py:150` : `tf.extractall(path=dest_resolved, filter="data")` (supprime le `DeprecationWarning` Python 3.14).

### Commandes & résultats bruts (`docs/results/raw/phase9_static_and_cpu.txt`)
- `ruff check medresearch_platform scripts tests` → **`All checks passed!`** (`RUFF_EXIT=0`)
- `mypy medresearch_platform` → **`Success: no issues found in 99 source files`** (`MYPY_EXIT=0`, `10 → 0` erreurs)
- Scan zéro pré-entraîné / API externe : **`ZERO_PRETRAINED_OR_EXTERNAL_API_CALLS`**
- Profilage CPU (Intel Core i7-9750H, 32 Go RAM, CPU-only) :
  - RSS baseline : `144.5 Mo`
  - Chargement `hybrid_v2` : **`33.0 ms`**, RSS `148.5 Mo` (`+4.0 Mo`)
  - Inférence 1 note `hybrid_v2` : **`29.2 ms`**, RSS `149.8 Mo`
- Dépendances (`pyproject.toml`) : aucune nouvelle dépendance ajoutée ; contraintes `numpy<2` + `torch==2.2.2` pour macOS Intel `x86_64` intactes.
- Statut : **Passed**
