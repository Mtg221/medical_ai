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
