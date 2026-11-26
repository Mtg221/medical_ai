# Model Card: CustomMedicalNLPNet v1.0.0

- **Model ID:** `mdl_nlp_multitask_v1`
- **Approach Category:** `MEDICAL_NLP` (Approach 2)
- **Research Track:** `PRIMARY_RANDOM_INIT` (Zero pretrained embeddings or weights)
- **Lifecycle Status:** `RESEARCH_ONLY` (**Experimentally validated** on synthetic bilingual benchmark; **Not clinically validated**)

## 1. Architecture & Random Initialization Provenance
- **Input Representation:** 96-dimensional deterministic feature vector combining 64 character 3-gram hashing buckets and 32 orthographic + bilingual clinical context indicator features (explicitly documented per Section 7).
- **Network Architecture:** Shared `Linear(96, 64)` + `tanh` hidden representation feeding three task heads:
  1. `entity_head`: `Linear(64, 12)` over 12 clinical entity types (`SYMPTOM`, `DIAGNOSIS`, `MEDICATION`, `DOSAGE`, `FREQUENCY`, `ROUTE`, `LAB_TEST`, `LAB_VALUE`, `LAB_UNIT`, `PROCEDURE`, `ANATOMY`, `TEMPORAL_EXPR`)
  2. `assertion_head`: `Linear(64, 6)` over 6 assertion states (`AFFIRMED`, `NEGATED`, `UNCERTAIN`, `HYPOTHETICAL`, `CONDITIONAL`, `FAMILY_HISTORY`)
  3. `temporal_head`: `Linear(64, 5)` over 5 temporal states (`CURRENT`, `HISTORICAL`, `PLANNED`, `RESOLVED`, `UNKNOWN`)
- **Parameter Count:** `7,703` trainable float32 parameters
- **Initialization Method:** `xavier_uniform` (biases initialized to `zeros`)
- **Random Seed:** `20261112`
- **Initial Untrained Weights SHA-256:** `b324edda246624b1848b67d804d0e0e45ebbbb9878d99a10f86c063d1a20ea4f`
- **Trained Weights SHA-256:** `4b64997f9ed7db295f5fbb592ec39ab1975a2ce29faaf997463068ac1745fb56`

## 2. Evaluation, Calibration & Fairness Summary (`SYNTH-MED-EN-FR v1.0.0`)
- **Entity Extraction (Exact Span):** `Precision = 0.9250`, `Recall = 1.0000`, `Exact F1 = 0.9610`
- **Assertion Status Accuracy:** `0.9730` (95% Bootstrap CI: `[0.9182, 1.0000]`)
- **Temporal Status Accuracy:** `0.9730` (95% Bootstrap CI: `[0.9189, 1.0000]`)
- **Calibration:** `Brier Score = 0.0725`, `Expected Calibration Error (ECE) = 0.1454`
- **Selective Prediction Risk:** Error rate decreases from `0.0750` at 100% coverage to `0.0278` at 90% coverage and `0.0000` at 80% coverage.
- **Subgroup Fairness (Language):** English `Exact F1 = 0.9714`, French `Exact F1 = 0.9091`, Mixed EN/FR `Exact F1 = 1.0000` (`max_exact_f1_disparity = 0.0909`).

## 3. Intended Use, Prohibited Use & Limitations
- **Intended Use:** Bilingual (EN/FR) medical entity, negation/uncertainty, temporality, and relation extraction research with mandatory human review capability.
- **Prohibited Use:** Autonomous diagnosis, treatment modification, or unreviewed clinical decision making.
