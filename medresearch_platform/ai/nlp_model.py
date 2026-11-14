"""Approach 2: Custom Random-Initialized Medical NLP Entity, Assertion, Temporality & Relation Model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from medresearch_platform.ai.random_init import (
    InitMethod,
    RandomInitManifest,
    build_random_init_manifest,
    compute_weights_sha256,
    initialize_tensor,
)
from medresearch_platform.ai.terminology import BilingualTerminologyRegistry

ENTITY_TYPES: tuple[str, ...] = (
    "SYMPTOM",
    "DIAGNOSIS",
    "MEDICATION",
    "DOSAGE",
    "FREQUENCY",
    "ROUTE",
    "LAB_TEST",
    "LAB_VALUE",
    "LAB_UNIT",
    "PROCEDURE",
    "ANATOMY",
    "TEMPORAL_EXPR",
)

ASSERTION_STATUSES: tuple[str, ...] = (
    "AFFIRMED",
    "NEGATED",
    "UNCERTAIN",
    "HYPOTHETICAL",
    "CONDITIONAL",
    "FAMILY_HISTORY",
)

TEMPORAL_STATUSES: tuple[str, ...] = (
    "CURRENT",
    "HISTORICAL",
    "PLANNED",
    "RESOLVED",
    "UNKNOWN",
)

MEDICATION_STATUSES: tuple[str, ...] = (
    "NOT_APPLICABLE",
    "PRESCRIBED",
    "DISPENSED",
    "TAKEN",
    "ADMINISTERED",
    "DISCONTINUED",
    "UNKNOWN",
)


@dataclass(frozen=True)
class ExtractedEntityRecord:
    entity_type: str
    source_text: str
    normalized_text: str
    terminology_concept_id: str | None
    mapping_match_type: str
    assertion_status: str
    temporal_status: str
    medication_status: str
    page_number: int
    section_type: str
    char_start: int
    char_end: int
    bbox: dict[str, int]
    confidence: float
    uncertainty: float
    abstained: bool
    explanation_features: list[str]


@dataclass(frozen=True)
class ExtractedRelationRecord:
    source_idx: int
    target_idx: int
    relation_type: str
    derivation_type: str
    evidence_text: str
    confidence: float
    uncertainty: float


# Transparent, documented bilingual candidate span patterns (Section 7 requirement: explicitly documented)
_BILINGUAL_SPAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "DIAGNOSIS",
        re.compile(
            r"\b(essential hypertension|hypertension artérielle essentielle|hypertension artérielle|hypertension|"
            r"type 2 diabetes mellitus|diabète sucré de type 2|diabète de type 2|type 2 diabetes|"
            r"bacterial pneumonia|pneumopathie infectieuse|pneumonie|pneumonia|"
            r"pulmonary embolism|embolie pulmonaire|atrial fibrillation|fibrillation auriculaire)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "SYMPTOM",
        re.compile(
            r"\b(shortness of breath|dyspnea|dyspnée|essoufflement|"
            r"chest pain|douleur thoracique|douleurs thoraciques|"
            r"fever|fièvre|pyrexia|cough|toux sèche|toux)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MEDICATION",
        re.compile(
            r"\b(metformin|metformine|amlodipine|amoxicillin|amoxicilline|apixaban)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "DOSAGE",
        re.compile(r"\b(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ui|ml))\b", re.IGNORECASE),
    ),
    (
        "FREQUENCY",
        re.compile(
            r"\b(twice daily|once daily|three times daily|bid|qd|tid|"
            r"deux fois par jour|une fois par jour|trois fois par jour|matin et soir)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ROUTE",
        re.compile(r"\b(oral|orally|per os|po|intravenous|iv|voie orale|intraveineuse)\b", re.IGNORECASE),
    ),
    (
        "LAB_TEST",
        re.compile(
            r"\b(hemoglobin a1c|hémoglobine glyquée|hba1c|"
            r"c-reactive protein|protéine c-réactive|crp|"
            r"serum creatinine|créatininémie|créatinine|creatinine)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "LAB_VALUE",
        re.compile(r"(?:hba1c|crp|creatinine|créatinine|créatininémie|glyquée)\s*(?:is|est\s*à|de|:|=)?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    ),
    (
        "LAB_UNIT",
        re.compile(r"\b(mg/l|mg/dl|mmol/l|µmol/l|umol/l|%)\b", re.IGNORECASE),
    ),
    (
        "PROCEDURE",
        re.compile(
            r"\b(computed tomography of chest|ct pulmonary angiogram|chest ct|"
            r"angioscanner thoracique|scanner thoracique|tdm thoracique)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ANATOMY",
        re.compile(r"\b(thorax|chest|lungs|pulmonary|poitrine|poumons|pulmonaire|thoracique)\b", re.IGNORECASE),
    ),
    (
        "TEMPORAL_EXPR",
        re.compile(
            r"\b((?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])|"
            r"for\s+\d+\s+days|since\s+\d+\s+days|depuis\s+\d+\s+jours|il\s+y\s+a\s+\d+\s+ans|in\s+20\d{2}|en\s+20\d{2})\b",
            re.IGNORECASE,
        ),
    ),
)

# Contextual cue lexicons (EN + FR) used as transparent input features into the custom neural classifier
_NEGATION_CUES = (
    "no ", "not ", "denies ", "without ", "ruled out", "rules out", "negative for", "free of",
    "pas de ", "sans ", "absence de ", "aucun ", "aucune ", "écarté", "ecarte", "exclue", "exclu", "ni ",
)
_UNCERTAINTY_CUES = (
    "suspected", "possible", "probable", "cannot rule out", "questionable", "evaluate for",
    "suspecté", "suspecte", "suspicion de", "probable", "possible", "hypothèse de", "a éliminer",
)
_HYPOTHETICAL_CUES = (
    "if ", "should ", "in case of", "return if", "call if",
    "en cas de", "si apparition", "si douleur", "si fièvre",
)
_FAMILY_CUES = (
    "family history", "mother had", "father had", "mother with", "father with", "sister with",
    "antécédent familial", "antecedent familial", "mère atteinte", "père atteint", "histoire familiale",
)
_HISTORICAL_CUES = (
    "history of", "past medical history", "previously", "resolved", "former", "in 201", "in 2020", "in 2021",
    "antécédent de", "antecedents de", "ancien", "ancienne", "résolu", "guéri", "il y a",
)
_PLANNED_CUES = (
    "scheduled", "plan to", "will order", "recommend", "follow-up",
    "prévu", "programmer", "à réaliser", "planifié",
)
_DISCONTINUED_MED_CUES = (
    "discontinued", "stopped", "held", "ceased",
    "arrêté", "arrete", "suspendu", "interrompu",
)
_TAKEN_MED_CUES = (
    "takes ", "taking ", "reported taking", "compliant with",
    "prend ", "prise de", "observant sous",
)
_PRESCRIBED_MED_CUES = (
    "prescribed", "start ", "initiated on", "discharge on",
    "prescrit", "débuter", "instauration de", "ordonnance",
)


def encode_span_and_context_features(
    span_text: str,
    left_context: str,
    right_context: str,
    feature_dim: int = 96,
) -> tuple[np.ndarray, list[str]]:
    """Featurize a candidate span and its surrounding bilingual clinical context into a fixed vector.
    Uses character 3-gram hashing + orthographic + contextual indicator features (zero pretrained embeddings).
    """
    vec = np.zeros(feature_dim, dtype=np.float32)
    explanations: list[str] = []
    clean_span = span_text.strip().lower()
    left_low = left_context.lower()
    right_low = right_context.lower()
    window_low = f"{left_low} {clean_span} {right_low}"

    # buckets 0..63: deterministic character 3-gram hashing of span_text
    padded = f"<{clean_span}>"
    for i in range(max(1, len(padded) - 2)):
        trigram = padded[i : i + 3]
        h = sum((ord(c) * (31 ** idx)) for idx, c in enumerate(trigram)) % 64
        vec[h] += 1.0
    norm = float(np.linalg.norm(vec[:64]))
    if norm > 0:
        vec[:64] /= norm

    # buckets 64..95: interpretable orthographic & clinical context features
    has_digit = any(c.isdigit() for c in clean_span)
    vec[64] = 1.0 if has_digit else 0.0
    vec[65] = 1.0 if any(u in clean_span for u in ("mg", "g", "ml", "mcg", "%", "mmol", "µmol")) else 0.0
    vec[66] = 1.0 if any(f in clean_span for f in ("daily", "jour", "bid", "tid", "qd", "fois", "matin")) else 0.0
    vec[67] = 1.0 if any(r in clean_span for r in ("oral", "iv", "intravenous", "voie", "per os", "po")) else 0.0
    vec[68] = 1.0 if any(d in clean_span for d in ("hypertension", "diabetes", "diabète", "pneumonia", "pneumopathie", "pneumonie", "embolism", "embolie", "fibrillation")) else 0.0
    vec[69] = 1.0 if any(s in clean_span for s in ("dyspnea", "dyspnée", "essoufflement", "pain", "douleur", "fever", "fièvre", "cough", "toux")) else 0.0
    vec[70] = 1.0 if any(m in clean_span for m in ("metformin", "metformine", "amlodipine", "amoxicillin", "amoxicilline", "apixaban")) else 0.0
    vec[71] = 1.0 if any(lab in clean_span for lab in ("hba1c", "glyquée", "crp", "reactive", "réactive", "creatinine", "créatinine", "créatininémie")) else 0.0
    vec[72] = 1.0 if any(p in clean_span for p in ("computed tomography", "chest ct", "angioscanner", "scanner", "tdm")) else 0.0
    vec[73] = 1.0 if any(a in clean_span for a in ("thorax", "chest", "lungs", "poitrine", "poumons")) else 0.0
    vec[74] = 1.0 if any(t in clean_span for t in ("days", "jours", "ans", "202", "201", "depuis", "since")) else 0.0
    vec[75] = 1.0 if (has_digit and not vec[65] and not vec[71] and not vec[74]) else 0.0

    # Context cues for assertion & temporality
    if any(cue in left_low or cue in right_low[:25] for cue in _NEGATION_CUES):
        vec[80] = 1.0
        explanations.append("context_cue:negation_trigger")
    if any(cue in left_low or cue in right_low[:25] for cue in _UNCERTAINTY_CUES):
        vec[81] = 1.0
        explanations.append("context_cue:uncertainty_trigger")
    if any(cue in left_low for cue in _HYPOTHETICAL_CUES):
        vec[82] = 1.0
        explanations.append("context_cue:hypothetical_trigger")
    if any(cue in left_low for cue in _FAMILY_CUES):
        vec[83] = 1.0
        explanations.append("context_cue:family_history_trigger")
    if any(cue in window_low for cue in _HISTORICAL_CUES):
        vec[84] = 1.0
        explanations.append("context_cue:historical_trigger")
    if any(cue in window_low for cue in _PLANNED_CUES):
        vec[85] = 1.0
        explanations.append("context_cue:planned_trigger")
    if any(cue in window_low for cue in _DISCONTINUED_MED_CUES):
        vec[86] = 1.0
        explanations.append("context_cue:medication_discontinued")
    if any(cue in window_low for cue in _TAKEN_MED_CUES):
        vec[87] = 1.0
        explanations.append("context_cue:medication_taken")
    if any(cue in window_low for cue in _PRESCRIBED_MED_CUES):
        vec[88] = 1.0
        explanations.append("context_cue:medication_prescribed")

    if not explanations:
        explanations.append("context_cue:default_affirmed_current")

    return vec, explanations


def _build_nlp_weights(seed: int, init_method: InitMethod = "xavier_uniform") -> dict[str, np.ndarray]:
    """Build randomly initialized multi-head neural weights for CustomMedicalNLPNet-v1."""
    rng = np.random.default_rng(seed)
    return {
        "shared_w": initialize_tensor(rng, (96, 64), init_method),
        "shared_b": initialize_tensor(rng, (64,), "zeros"),
        "entity_w": initialize_tensor(rng, (64, len(ENTITY_TYPES)), init_method),
        "entity_b": initialize_tensor(rng, (len(ENTITY_TYPES),), "zeros"),
        "assertion_w": initialize_tensor(rng, (64, len(ASSERTION_STATUSES)), init_method),
        "assertion_b": initialize_tensor(rng, (len(ASSERTION_STATUSES),), "zeros"),
        "temporal_w": initialize_tensor(rng, (64, len(TEMPORAL_STATUSES)), init_method),
        "temporal_b": initialize_tensor(rng, (len(TEMPORAL_STATUSES),), "zeros"),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / np.sum(exp_v, axis=1, keepdims=True)


class CustomMedicalNLPNet:
    """Custom randomly initialized Multi-Head Clinical NLP Neural Network (Entity + Assertion + Temporality)."""

    architecture_name = "CustomMedicalNLPNet"
    architecture_version = "1.0.0"

    def __init__(
        self,
        seed: int = 20261112,
        init_method: InitMethod = "xavier_uniform",
        abstention_uncertainty_threshold: float = 0.65,
    ) -> None:
        self.seed = seed
        self.init_method: InitMethod = init_method
        self.abstention_uncertainty_threshold = abstention_uncertainty_threshold
        self.params = _build_nlp_weights(seed, init_method)
        self.init_manifest: RandomInitManifest = build_random_init_manifest(
            architecture_name=self.architecture_name,
            architecture_version=self.architecture_version,
            init_method=init_method,
            seed=seed,
            parameters=self.params,
        )
        self.terminology = BilingualTerminologyRegistry()

    def forward(self, x_batch: np.ndarray) -> dict[str, np.ndarray]:
        h_pre = np.matmul(x_batch, self.params["shared_w"]) + self.params["shared_b"]
        h_act = np.tanh(h_pre)

        ent_logits = np.matmul(h_act, self.params["entity_w"]) + self.params["entity_b"]
        ass_logits = np.matmul(h_act, self.params["assertion_w"]) + self.params["assertion_b"]
        tmp_logits = np.matmul(h_act, self.params["temporal_w"]) + self.params["temporal_b"]

        return {
            "h_pre": h_pre,
            "h_act": h_act,
            "entity_probs": _softmax(ent_logits),
            "assertion_probs": _softmax(ass_logits),
            "temporal_probs": _softmax(tmp_logits),
        }

    def train_on_labeled_spans(
        self,
        training_samples: list[dict[str, Any]],
        *,
        epochs: int = 80,
        learning_rate: float = 0.12,
        weight_decay: float = 1e-4,
    ) -> dict[str, Any]:
        """Train all three heads jointly from random initialization via analytical backpropagation."""
        x_rows: list[np.ndarray] = []
        y_ent: list[int] = []
        y_ass: list[int] = []
        y_tmp: list[int] = []

        for sample in training_samples:
            vec, _ = encode_span_and_context_features(
                sample["span_text"],
                sample["left_context"],
                sample["right_context"],
            )
            x_rows.append(vec)
            y_ent.append(ENTITY_TYPES.index(sample["entity_type"]))
            y_ass.append(ASSERTION_STATUSES.index(sample["assertion_status"]))
            y_tmp.append(TEMPORAL_STATUSES.index(sample["temporal_status"]))

        x_mat = np.stack(x_rows, axis=0)
        y_ent_arr = np.array(y_ent, dtype=np.int64)
        y_ass_arr = np.array(y_ass, dtype=np.int64)
        y_tmp_arr = np.array(y_tmp, dtype=np.int64)
        n = x_mat.shape[0]

        loss_history: list[float] = []
        for _epoch in range(epochs):
            out = self.forward(x_mat)
            p_ent = out["entity_probs"]
            p_ass = out["assertion_probs"]
            p_tmp = out["temporal_probs"]

            loss_ent = -np.mean(np.log(np.clip(p_ent[np.arange(n), y_ent_arr], 1e-8, 1.0)))
            loss_ass = -np.mean(np.log(np.clip(p_ass[np.arange(n), y_ass_arr], 1e-8, 1.0)))
            loss_tmp = -np.mean(np.log(np.clip(p_tmp[np.arange(n), y_tmp_arr], 1e-8, 1.0)))
            total_loss = float(loss_ent + loss_ass + loss_tmp)
            loss_history.append(round(total_loss, 5))

            dent = p_ent.copy()
            dent[np.arange(n), y_ent_arr] -= 1.0
            dent /= float(n)

            dass = p_ass.copy()
            dass[np.arange(n), y_ass_arr] -= 1.0
            dass /= float(n)

            dtmp = p_tmp.copy()
            dtmp[np.arange(n), y_tmp_arr] -= 1.0
            dtmp /= float(n)

            h_act = out["h_act"]
            dentity_w = np.matmul(h_act.T, dent) + weight_decay * self.params["entity_w"]
            dentity_b = np.sum(dent, axis=0)

            dassertion_w = np.matmul(h_act.T, dass) + weight_decay * self.params["assertion_w"]
            dassertion_b = np.sum(dass, axis=0)

            dtemporal_w = np.matmul(h_act.T, dtmp) + weight_decay * self.params["temporal_w"]
            dtemporal_b = np.sum(dtmp, axis=0)

            dh_act = (
                np.matmul(dent, self.params["entity_w"].T)
                + np.matmul(dass, self.params["assertion_w"].T)
                + np.matmul(dtmp, self.params["temporal_w"].T)
            )
            dh_pre = dh_act * (1.0 - h_act**2)
            dshared_w = np.matmul(x_mat.T, dh_pre) + weight_decay * self.params["shared_w"]
            dshared_b = np.sum(dh_pre, axis=0)

            self.params["entity_w"] -= learning_rate * dentity_w
            self.params["entity_b"] -= learning_rate * dentity_b
            self.params["assertion_w"] -= learning_rate * dassertion_w
            self.params["assertion_b"] -= learning_rate * dassertion_b
            self.params["temporal_w"] -= learning_rate * dtemporal_w
            self.params["temporal_b"] -= learning_rate * dtemporal_b
            self.params["shared_w"] -= learning_rate * dshared_w
            self.params["shared_b"] -= learning_rate * dshared_b

        final_out = self.forward(x_mat)
        ent_acc = float(np.mean(np.argmax(final_out["entity_probs"], axis=1) == y_ent_arr))
        ass_acc = float(np.mean(np.argmax(final_out["assertion_probs"], axis=1) == y_ass_arr))
        tmp_acc = float(np.mean(np.argmax(final_out["temporal_probs"], axis=1) == y_tmp_arr))

        return {
            "epochs": epochs,
            "initial_loss": loss_history[0] if loss_history else 0.0,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "entity_train_accuracy": round(ent_acc, 4),
            "assertion_train_accuracy": round(ass_acc, 4),
            "temporal_train_accuracy": round(tmp_acc, 4),
            "trained_weights_sha256": compute_weights_sha256(self.params),
        }

    def extract_from_text(
        self,
        text: str,
        *,
        page_number: int = 1,
        section_type: str = "BODY",
    ) -> tuple[list[ExtractedEntityRecord], list[ExtractedRelationRecord]]:
        """Extract medical entities, assertions, temporality, terminology codes, and relations."""
        raw_candidates: list[tuple[int, int, str, str]] = []
        for pattern_type, regex in _BILINGUAL_SPAN_PATTERNS:
            for match in regex.finditer(text):
                grp = 1 if match.lastindex and match.lastindex >= 1 else 0
                s, e = match.span(grp)
                surface = match.group(grp)
                raw_candidates.append((s, e, surface, pattern_type))

        # Deduplicate overlapping candidate spans preferring longer spans
        raw_candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        filtered_candidates: list[tuple[int, int, str, str]] = []
        last_end = -1
        for s, e, surface, pattern_type in raw_candidates:
            if s >= last_end:
                filtered_candidates.append((s, e, surface, pattern_type))
                last_end = e

        entities: list[ExtractedEntityRecord] = []
        for s, e, surface, pattern_hint in filtered_candidates:
            # Determine local clause boundaries (bounded by newline, period, or semicolon)
            clause_start = max(
                text.rfind("\n", 0, s),
                text.rfind(".", 0, s),
                text.rfind(";", 0, s),
                0,
            )
            clause_end_candidates = [
                pos for pos in (text.find("\n", e), text.find(".", e), text.find(";", e)) if pos != -1
            ]
            clause_end = min(clause_end_candidates) if clause_end_candidates else len(text)

            left_ctx = text[max(clause_start, s - 65) : s]
            right_ctx = text[e : min(clause_end, e + 45)]

            feat_vec, explanations = encode_span_and_context_features(surface, left_ctx, right_ctx)
            out = self.forward(feat_vec.reshape(1, -1))

            ent_probs = out["entity_probs"][0]
            ass_probs = out["assertion_probs"][0]
            tmp_probs = out["temporal_probs"][0]

            pred_ent_idx = int(np.argmax(ent_probs))
            pred_ass_idx = int(np.argmax(ass_probs))
            pred_tmp_idx = int(np.argmax(tmp_probs))

            # Combine neural prediction with pattern type validation
            entity_type = ENTITY_TYPES[pred_ent_idx] if float(ent_probs[pred_ent_idx]) >= 0.45 else pattern_hint
            if pattern_hint in ("DOSAGE", "FREQUENCY", "ROUTE", "LAB_VALUE", "LAB_UNIT", "TEMPORAL_EXPR"):
                entity_type = pattern_hint

            assertion_status = ASSERTION_STATUSES[pred_ass_idx]
            temporal_status = TEMPORAL_STATUSES[pred_tmp_idx]

            # Explicit rule-verified safety check for critical clinical distinctions (Section 10)
            if feat_vec[83] > 0.5:
                assertion_status = "FAMILY_HISTORY"
            elif feat_vec[82] > 0.5:
                assertion_status = "HYPOTHETICAL"
            elif feat_vec[81] > 0.5:
                assertion_status = "UNCERTAIN"
            elif feat_vec[80] > 0.5:
                assertion_status = "NEGATED"

            if feat_vec[84] > 0.5:
                temporal_status = "HISTORICAL"
            elif feat_vec[85] > 0.5:
                temporal_status = "PLANNED"

            medication_status = "NOT_APPLICABLE"
            if entity_type == "MEDICATION":
                if feat_vec[86] > 0.5:
                    medication_status = "DISCONTINUED"
                elif feat_vec[87] > 0.5:
                    medication_status = "TAKEN"
                elif feat_vec[88] > 0.5:
                    medication_status = "PRESCRIBED"
                else:
                    medication_status = "UNKNOWN"

            # Uncertainty propagation: average normalized entropy across heads
            ent_entropy = float(
                -np.sum(ent_probs * np.log(np.clip(ent_probs, 1e-9, 1.0))) / np.log(len(ENTITY_TYPES))
            )
            ass_entropy = float(
                -np.sum(ass_probs * np.log(np.clip(ass_probs, 1e-9, 1.0))) / np.log(len(ASSERTION_STATUSES))
            )
            combined_uncertainty = round(0.5 * (ent_entropy + ass_entropy), 4)
            combined_confidence = round(
                float(0.5 * ( np.max(ent_probs) + np.max(ass_probs) )), 4
            )
            abstained = bool(combined_uncertainty > self.abstention_uncertainty_threshold)

            term_map = self.terminology.map_mention(
                surface,
                category_hint=entity_type if entity_type in ("DIAGNOSIS", "SYMPTOM", "MEDICATION", "LAB_TEST", "PROCEDURE", "ANATOMY") else None,
            )

            line_number = text.count("\n", 0, s)
            bbox = {
                "page": page_number,
                "x": (s - text.rfind("\n", 0, s)) * 8,
                "y": 40 + line_number * 22,
                "width": max(16, (e - s) * 8),
                "height": 18,
            }

            entities.append(
                ExtractedEntityRecord(
                    entity_type=entity_type,
                    source_text=surface,
                    normalized_text=term_map.normalized_text or surface.lower(),
                    terminology_concept_id=term_map.concept_id,
                    mapping_match_type=term_map.match_type,
                    assertion_status=assertion_status,
                    temporal_status=temporal_status,
                    medication_status=medication_status,
                    page_number=page_number,
                    section_type=section_type,
                    char_start=s,
                    char_end=e,
                    bbox=bbox,
                    confidence=combined_confidence,
                    uncertainty=combined_uncertainty,
                    abstained=abstained,
                    explanation_features=explanations,
                )
            )

        relations = self._extract_same_clause_relations(text, entities)
        return entities, relations

    @staticmethod
    def _extract_same_clause_relations(
        text: str,
        entities: list[ExtractedEntityRecord],
    ) -> list[ExtractedRelationRecord]:
        """Link medications to dosage/frequency/route and lab tests to values/units within the same sentence."""
        relations: list[ExtractedRelationRecord] = []
        for i, src in enumerate(entities):
            for j, dst in enumerate(entities):
                if i == j:
                    continue
                dist = abs(dst.char_start - src.char_end) if dst.char_start >= src.char_end else abs(src.char_start - dst.char_end)
                if dist > 85:
                    continue
                span_lo = min(src.char_start, dst.char_start)
                span_hi = max(src.char_end, dst.char_end)
                between = text[span_lo:span_hi]
                if "\n\n" in between:
                    continue

                rel_type: str | None = None
                if src.entity_type == "MEDICATION" and dst.entity_type == "DOSAGE":
                    rel_type = "HAS_DOSAGE"
                elif src.entity_type == "MEDICATION" and dst.entity_type == "FREQUENCY":
                    rel_type = "HAS_FREQUENCY"
                elif src.entity_type == "MEDICATION" and dst.entity_type == "ROUTE":
                    rel_type = "HAS_ROUTE"
                elif src.entity_type == "LAB_TEST" and dst.entity_type == "LAB_VALUE":
                    rel_type = "HAS_LAB_VALUE"
                elif src.entity_type == "LAB_TEST" and dst.entity_type == "LAB_UNIT":
                    rel_type = "HAS_LAB_UNIT"
                elif src.entity_type == "SYMPTOM" and dst.entity_type == "ANATOMY":
                    rel_type = "LOCATED_IN_ANATOMY"
                elif src.entity_type in ("SYMPTOM", "DIAGNOSIS", "PROCEDURE") and dst.entity_type == "TEMPORAL_EXPR":
                    rel_type = "TEMPORALLY_ANCHORED_TO"

                if rel_type:
                    conf = round(min(src.confidence, dst.confidence), 4)
                    unc = round(max(src.uncertainty, dst.uncertainty), 4)
                    relations.append(
                        ExtractedRelationRecord(
                            source_idx=i,
                            target_idx=j,
                            relation_type=rel_type,
                            derivation_type="EXTRACTED",
                            evidence_text=between.strip(),
                            confidence=conf,
                            uncertainty=unc,
                        )
                    )
        return relations

    def save_checkpoint(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **cast(Any, self.params))
        return compute_weights_sha256(self.params)

    def load_checkpoint(self, path: Path) -> None:
        with np.load(str(path)) as data:
            for key in self.params:
                self.params[key] = data[key].astype(np.float32)
