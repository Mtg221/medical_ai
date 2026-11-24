# Rapport Final d'Ingénierie, de Validation & de Reproductibilité (`medical_ai`)

**Date (UTC) :** `2026-11-24`  
**Branche de travail :** `chore/completion-audit` (créée depuis `main` @ `6b8bd3cfb80ae452c0222c3cb10ba8031f393e0c`)  
**Push distant :** **Non effectué** (strictement interdit sans autorisation explicite)  
**Plateforme matérielle vérifiée :** macOS Darwin `x86_64`, Intel Core i7-9750H @ 2.60 GHz, 32 Go RAM, **CPU-only**, Python `3.12.16` (`.venv-vision`), Node `v24.18.0`.

---

## 1. Résumé exécutif & maturité scientifique honnête

La plateforme `medical_ai` est un **prototype de recherche bilingue (EN/FR) 100 % entraîné from-scratch** (zéro poids pré-entraîné, zéro API LLM/inférence externe). L'audit complet en 12 phases (Phases 0 à 11) a permis d'identifier et corriger des défauts concrets d'implémentation, de métriques et de protocole **sans introduire aucune nouvelle architecture de modèle**, **sans modifier aucun split historique gelé**, et **sans fabriquer aucune donnée humaine**.

**Niveau de maturité honnête :**
- **Prêt pour la recherche reproductible locale et l'évaluation aveugle contrôlée** (CI, typage `mypy` 0 erreur, `ruff` 0 erreur, 30 checkpoints locaux SHA-256 vérifiés, outillage d'import/adjudication/gel 80 notes humaines durci, métriques PHI span-level exactes, sécurité RBAC/SQLite/CSP/audit vérifiée).
- **Non validé cliniquement / non certifié dispositif médical :** l'évaluation indépendante sur **80 notes cliniques écrites et double-annotées par un humain** (`post_hoc: false`) et sur **>= 50 phrases PHI humaines inédites + >= 10 phrases non-PHI pièges** reste **en attente des textes du propriétaire**. De même, la piste vision dermatologie n'a pas encore été évaluée sur des cohortes cliniques smartphone externes restreintes (`PAD-UFES-20`, `Fitzpatrick17k`, `DDI`) ni en mode shadow prospectif hospitalier.

---

## 2. Commits sur `chore/completion-audit` & fichiers modifiés

| Commit | Portée | Résumé |
|---|---|---|
| `269cce1` | Phases 0–2 | Journal `docs/WORK_LOG.md`, sorties brutes Phases 0–2, durcissement protocole 80 notes humaines (`scripts/import_label_studio_realistic.py`, `scripts/eval_realistic.py`, `tests/test_human_realistic_protocol_strict.py`). |
| `cb1d2f6` | Phase 3 | Correction du bug de précision PHI (span-level P/R/F1 séparé du taux FP par phrase, détection FP intra-phrase) + câblage HIPAA `>89 ans` (`medresearch_platform/security/person_age_detector.py`, `medresearch_platform/security/privacy.py`, `scripts/eval_redaction_human.py`, `tests/test_phase3_phi_metrics_and_age.py`). |
| `33339ce` | Phase 4 | Isolation des répertoires dans `ExtractionService` (`serving/extractor.py`), négation FR `ne montre pas` (`approaches/context_negex.py`), garde frontière de phrase et déduplication multi-entités des relations (`hybrid/stage3_context_relations.py`, `tests/test_phase4_nlp_assertions_relations_temporal.py`). |
| `24678be` | Phases 5–9 | Résolution des 10 erreurs `mypy` dans 6 fichiers (`ocr_engine.py`, `neural_seq.py`, `classical_ml.py`, `stage3_context_relations.py`, `stage2_ner.py`, `pipeline.py`), suppression des warnings PyTorch non-writable (`vision/cnn_torch.py`) et `tarfile.extractall(filter="data")` (`scripts/package_release_artifacts.py`), sorties brutes Phases 5–9. |
| *Commit final Phase 11* | Phases 10–11 | Réconciliation `PROJECT_STATUS.md:24` (CER OCR EN/FR), sorties brutes Phases 10–11, `docs/WORK_LOG.md`, `docs/FINAL_ENGINEERING_AND_RESEARCH_COMPLETION_REPORT.md`. |

---

## 3. Causes racines identifiées & correctifs appliqués

1. **Outillage 80 notes humaines non conforme au protocole (`scripts/import_label_studio_realistic.py`, `scripts/eval_realistic.py`) :**
   - *Cause racine :* défauts silencieux (`language="en"`, `doc_type="consultation"`, `document_date="2025-01-01"`), sélection silencieuse de `annotations[0]` comme gold même en cas de désaccord, absence de vérification des quotas (80 docs uniques, 40 EN / 40 FR, 6 `DOC_TYPES`, `>= 16` double-annotés `>= 20 %`) et de manifeste de gel SHA-256.
   - *Correctif :* validation explicite obligatoire des métadonnées, rejet explicite de tout désaccord non adjudiqué (`ground_truth=true`/`adjudicated=true` requis), calcul IAA sur les annotateurs indépendants, génération d'un manifeste SHA-256 vérifié avant toute exécution `--owner-written` dans `eval_realistic.py`.
2. **Bug de métrique précision PHI + détection d'âge non câblée (`scripts/eval_redaction_human.py`, `security/privacy.py`, `security/person_age_detector.py`) :**
   - *Cause racine :* `precision = tp / (tp + fp_non_phi)` mélangeait entités PHI (`tp`) et phrases non-PHI (`fp_non_phi`), rendant invisibles les faux positifs intra-phrase ; `detect_age_spans` n'était pas appelé dans `deidentify_clinical_text`.
   - *Correctif :* calcul span-level exact et overlap (P/R/F1, `strict_full_token_recall`, `false_positive_spans` intra-phrase) strictement séparé de `non_phi_sentence_fp_rate`, ventilé par langue (`en`/`fr`) et par catégorie ; câblage HIPAA `>89 ans` (`[AGE_OVER_89_REDACTED]`, option `redact_all_ages=False`) avec garde renforcée contre durées (`for 14 days`, `pendant 7 jours`, `after 2 years`, `après 3 ans`, `every 5 years`), dates et mesures.
3. **Échec baseline `test_conftest_helper_builds_valid_loadable_models` (`serving/extractor.py:101-120`) :**
   - *Cause racine :* `ExtractionService(model_dir=..., classical_crf_dir=...)` scannait malgré tout `models/hospital_char_cnn_bilstm_crf_seed*` à la racine du dépôt (`len(loaded) == 3 != 2`).
   - *Correctif :* ancrage de `self.torch_ner_dir` sur `self.model_dir.parent` lorsque `model_dir` est explicitement passé.
4. **Négation clinique FR manquante & liaisons croisées multi-entités (`approaches/context_negex.py`, `hybrid/stage3_context_relations.py`) :**
   - *Cause racine :* absence des cues FR (`ne montre pas `, `ne retrouve pas `, `n'objective pas `, etc.) et absence de garde frontière de phrase / tête intermédiaire / déduplication dans `PiecewiseCNNRelationClassifier.predict_relations`.
   - *Correctif :* ajout des cues FR et des gardes structurelles pour les 5 relations fonctionnelles (`HAS_DOSAGE`, `HAS_FREQUENCY`, `HAS_ROUTE`, `HAS_VALUE`, `HAS_UNIT`).
5. **10 erreurs `mypy` + 2 warnings runtime (`ocr_engine.py`, `neural_seq.py`, `classical_ml.py`, `stage3_context_relations.py`, `stage2_ner.py`, `pipeline.py`, `cnn_torch.py`, `package_release_artifacts.py`) :**
   - *Correctif :* `value=(255,)` pour `cv2.copyMakeBorder`, `rng.permutation` typé au lieu de `rng.shuffle(list)`, `np.array(..., copy=True)` avant `torch.from_numpy`, `tarfile.extractall(..., filter="data")`.
6. **Contradiction documentaire OCR (`PROJECT_STATUS.md:24`) :**
   - *Cause racine :* `PROJECT_STATUS.md` citait `en 0.0725, fr 0.0999` alors que `docs/results/v3_1_ocr_full_benchmark.json` mesure `en 0.0824, fr 0.0899`. Réconcilié.

---

## 4. Commandes exécutées, sorties brutes & résultats vérifiés

Toutes les métriques ci-dessous proviennent de sorties brutes sauvegardées dans `docs/results/raw/` :

| Phase / Contrôle | Sortie brute | Résultat vérifié | Statut |
|---|---|---|---|
| Phase 0 — Baseline `pytest` complet | `docs/results/raw/phase0_pytest.txt` | `1 failed, 416 passed, 5 xfailed, 4 xpassed` (échec `test_conftest_helper_builds_valid_loadable_models` corrigé en Phase 4) | **Fixed** |
| Phase 1 — Intégrité des données & fuites | `docs/results/raw/phase1_data_integrity.txt` | `SYNTH-MED-V3.1` `4/4` SHA-256 OK (`0` fuite) ; contamination `PxCorpus` (`73/443`) documentée sans altérer les splits | **Passed** |
| Phase 2 — Outillage 80 notes humaines | `docs/results/raw/phase2_human_eval_tooling.txt` | `14 passed in 0.22s` (`EXIT=0`) | **Passed** |
| Phase 3 — PHI span-level & âge HIPAA | `docs/results/raw/phase3_phi_eval.txt` | `19 passed in 21.75s` (`PYTEST_EXIT=0`) | **Passed** |
| Phase 4 — NLP avant/après (`val.jsonl` dev) | `docs/results/raw/phase4_dev_before.txt`, `phase4_dev_after.txt`, `phase4_pytest.txt` | Relations F1 `0.9437 → 0.9531` (`+0.94 pt`), Précision `0.9400 → 0.9765` (`-63 %` FP), `14/14` assertions EN+FR, `53 passed` | **Passed** |
| Phase 5 — OCR & propagation NLP | `docs/results/raw/phase5_ocr_pytest.txt`, `phase5_ocr_eval.txt` | `17 passed in 30.08s` ; CER `0.0862` (`en 0.0824`, `fr 0.0899`), WER `0.2112` | **Passed** |
| Phase 6 — Vision dermatologie (piste séparée) | `docs/results/raw/phase6_vision_audit.txt` | `23 passed in 271.07s` ; `locked_test_v4` (`n=1,611`, `0` doublon SHA-256) AUROC `0.9348`, dermoscopie `0.9727`, clinique `0.8805`, TBP auto-abstenu `0.8768` | **Passed** |
| Phase 7 — Sécurité, RBAC, isolation, audit | `docs/results/raw/phase7_security_audit.txt` | `19 passed in 96.30s` ; `SQLiteLoginRateLimiter`, CSP/`nosniff`/`no-store`, Argon2id, `jti` revocation, chaîne SHA-256 | **Passed** |
| Phase 8 — Backend, Frontend & DB | `docs/results/raw/phase8_backend_frontend_db.txt` | Migration SQLite vide idempotente (26 tables, 3 axes présents), `5 passed`, `ZERO_MOCK_CONSTANTS_IN_PROD_UI` | **Passed** |
| Phase 9 — Qualité code & profil CPU | `docs/results/raw/phase9_static_and_cpu.txt` | `ruff` `0` erreur, `mypy` `0` erreur (`99` fichiers), `ZERO_PRETRAINED_OR_EXTERNAL_API_CALLS`, `hybrid_v2` `33.0 ms` load / `29.2 ms` inférence (`149.8 Mo` RSS) | **Passed** |
| Phase 10 — Artefacts & Release | `docs/results/raw/phase10_artifacts_release.txt` | `30/30` checkpoints vérifiés (`344.6 Mo`, `0` invalide, `0` pré-entraîné), `checksums.sha256` OK ; Release GitHub non publiée sans autorisation | **Blocked (remote auth)** |
| Phase 11 — Vérification finale globale | `docs/results/raw/phase11_final_verify.txt` | `ruff` OK, `mypy` OK (`99` fichiers), build Vite OK, Vitest `13/13` OK, régression Python `92 passed in 67.75s` (`PYTEST_EXIT=0`) | **Passed** |

---

## 5. Intégrité des données & changements de comportement des modèles

### Intégrité des données (splits historiques gelés et intacts)
- **`SYNTH-MED-V3.1` (`LOCK_MANIFEST.json`, `5f5e3fe`) :** `4/4` empreintes SHA-256 conformes (`val`, `test_iid`, `test_unseen_terms`, `test_unseen_templates`), `0` chevauchement `doc_id` / `patient_id` / texte normalisé.
- **`open_clinical` (`train` vs `test`) — Contamination documentée (sans modifier le split) :** `73/443` lignes de `open_clinical/test.jsonl` (64 textes normalisés uniques), **100 % issues de `pxcorpus`**, sont textuellement identiques à `train.jsonl` (`0` fuite sur `MACCROBAT2020`, `E3C`, `BC5CDR`, `NCBI Disease`). Recommandation pour un futur `open_clinical_v2` : grouper par `(patient_id, sha256(normalized_text))` avant split.
- **`TEST_SET_LOG.md` :** 51 évaluations historiques consignées (`post_hoc: true`).
- **Vision `locked_test_v3` → `locked_test_v4` :** `0` chevauchement ID/lésion ; 13 doublons SHA-256 exacts retirés (`n=1,611`, `0` doublon exact) ; proximité `dHash <= 4` de `25.49 %` documentée.

### Changements de comportement des modèles (dev `val.jsonl`, `n=160`)
- **Relations `hybrid_v2` :** Précision **`0.9400 → 0.9765`** (`+3.65 pt`), Rappel `0.9474 → 0.9308`, **F1 `0.9437 → 0.9531`** (`+0.94 pt`), faux positifs divisés par `2.7` (`62 → 23`), zéro liaison croisée sur les phrases multi-médicaments EN et FR.
- **Assertions 3 axes (`certainty`/`temporality`/`experiencer`) :** `14/14` phrases cliniques cibles (7 EN + 7 FR) exactes, dont `"Le scanner actuel ne montre pas la lésion."` désormais `absent`.
- **Temporalité :** Extraction F1 `0.9937` et Normalisation Accuracy `1.0000` mesurées séparément ; dates ambiguës (`03/04/2025`) et réparations OCR (`2O25-03-1O`) explicitement marquées `needs_review=True` ; dates impossibles (`2025-02-30`) rejetées (`None`).
- **Limite connue NLP (non masquée) :** sur phrases libres très courtes hors domaine (`"Previous scan showed a lesion."`), les 3 axes d'assertion sont exacts mais le typage NER de surface sur `hybrid_v2` seul peut encore confondre certains noms génériques (`lesion`, `scan does`) ; à évaluer sans tuning sur le futur jeu aveugle de 80 notes humaines.

---

## 6. Éléments en attente / bloqués (actions requises du propriétaire)

1. **80 notes cliniques écrites et double-annotées par un humain (`docs/REALISTIC_SET_OWNER_CHECKLIST.md`) :** 40 EN + 40 FR couvrant les 6 `DOC_TYPES`, `>= 16` double-annotées (`>= 20 %`) avec adjudication explicite des désaccords, importées via `scripts/import_label_studio_realistic.py` puis évaluées **une seule fois** avec `scripts/eval_realistic.py --owner-written` (`post_hoc: false`).
2. **Jeu PHI humain inédit (`docs/PHI_REDACTION_OWNER_TEMPLATE.jsonl`) :** `>= 50` phrases PHI EN/FR + `>= 10` phrases non-PHI pièges évaluées une fois avec `scripts/eval_redaction_human.py --owner-written`.
3. **Datasets dermatologie externes (`PAD-UFES-20`, `Fitzpatrick17k`, `DDI`) & protocole shadow-mode :** téléchargement soumis à licences/accès institutionnels (`scripts/ingest_external_clinical_derm.py`, `scripts/eval_shadow_mode.py`).
4. **Push Git & publication GitHub Release `v0.2.0-research-artifacts` :** la branche `chore/completion-audit` est prête localement ; après revue/merge et autorisation explicite, le propriétaire peut publier les archives vérifiées (`dist/release/models_nlp_18seed.tar.gz`, `dist/release/models_vision_9seed.tar.gz`, `dist/release/checksums.sha256`) via :
   ```bash
   gh release create v0.2.0-research-artifacts \
     dist/release/models_nlp_18seed.tar.gz \
     dist/release/models_vision_9seed.tar.gz \
     dist/release/checksums.sha256 \
     --title "v0.2.0-research-artifacts (30 From-Scratch Verified Checkpoints)" \
     --notes "SHA-256 verified from-scratch NLP (18+2) and Vision (9+1) checkpoints."
   ```

---

## 7. Commandes reproductibles (CPU-only, macOS i7-9750H)

### Dev rapide (~60–90 s)
```bash
.venv-vision/bin/ruff check medresearch_platform scripts tests
.venv-vision/bin/mypy medresearch_platform
.venv-vision/bin/pytest -q tests/test_human_realistic_protocol_strict.py tests/test_phase3_phi_metrics_and_age.py tests/test_phase4_nlp_assertions_relations_temporal.py tests/test_conftest_fresh_clone.py
npm --prefix frontend run build && npm --prefix frontend test -- --run
```

### Évaluation & vérification complète (~12 min, 1 job lourd à la fois)
```bash
.venv-vision/bin/python scripts/package_release_artifacts.py verify
.venv-vision/bin/pytest -q
```
