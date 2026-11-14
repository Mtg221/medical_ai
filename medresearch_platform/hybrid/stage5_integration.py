"""Stage 5: Transformer Candidate Reranker, Temperature Scaling, Consistency Checker, and Missing Info."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import numpy as np

from medresearch_platform.approaches.neural_seq import _AdamW, _sigmoid, _softmax
from medresearch_platform.data.schema import ENTITY_TYPES, AnnotatedDocument
from medresearch_platform.eval_harness.metrics import (
    PredictedEntity,
    PredictedRelation,
    PredictedTimex,
)
from medresearch_platform.hybrid.stage2_ner import CandidateEntity

_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(ENTITY_TYPES)}


class TransformerCandidateReranker:
    """Random-init self-attention reranker over candidate entities in a document."""

    def __init__(self, *, d_model: int = 24, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # Input features per candidate: 14 entity-type one-hot + 3 source flags + raw_conf + norm_len = 19
        in_dim = len(ENTITY_TYPES) + 5
        self.in_dim = in_dim
        self.d_model = d_model
        self.params: dict[str, np.ndarray] = {
            "W_in": rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, d_model)),
            "W_q": rng.normal(0.0, np.sqrt(1.0 / d_model), size=(d_model, d_model)),
            "W_k": rng.normal(0.0, np.sqrt(1.0 / d_model), size=(d_model, d_model)),
            "W_v": rng.normal(0.0, np.sqrt(1.0 / d_model), size=(d_model, d_model)),
            "W_out": rng.normal(0.0, np.sqrt(1.0 / d_model), size=(d_model, 1)),
            "b_out": np.zeros(1, dtype=np.float64),
        }

    def _featurize(self, candidates: list[CandidateEntity]) -> np.ndarray:
        mat = np.zeros((len(candidates), self.in_dim), dtype=np.float64)
        base = len(ENTITY_TYPES)
        for i, c in enumerate(candidates):
            t_idx = _TYPE_TO_IDX.get(c.entity.type)
            if t_idx is not None:
                mat[i, t_idx] = 1.0
            mat[i, base + 0] = 1.0 if "bilstm_crf" in c.source_stage else 0.0
            mat[i, base + 1] = 1.0 if "rule_layer" in c.source_stage else 0.0
            mat[i, base + 2] = 1.0 if c.source_stage == "bilstm_crf+rule_layer" else 0.0
            mat[i, base + 3] = float(c.entity.confidence)
            mat[i, base + 4] = min(len(c.entity.text), 40) / 40.0
        return mat

    def forward(self, x_raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        x = x_raw @ self.params["W_in"]
        scale = np.sqrt(float(self.d_model))
        q = x @ self.params["W_q"]
        k = x @ self.params["W_k"]
        v = x @ self.params["W_v"]
        attn = _softmax((q @ k.T) / scale)
        h = np.tanh(x + attn @ v)
        logits = (h @ self.params["W_out"] + self.params["b_out"]).squeeze(-1)
        return logits, {"x_raw": x_raw, "x": x, "q": q, "k": k, "v": v, "attn": attn, "h": h, "scale": scale}

    def step_doc(self, candidates: list[CandidateEntity], labels: np.ndarray, opt: _AdamW) -> float:
        x_raw = self._featurize(candidates)
        logits, cache = self.forward(x_raw)
        probs = _sigmoid(logits)
        n = len(labels)
        loss = float(-np.mean(labels * np.log(np.maximum(probs, 1e-12)) + (1.0 - labels) * np.log(np.maximum(1.0 - probs, 1e-12))))

        d_logits = (probs - labels) / max(1, n)
        grads: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.params.items()}
        grads["W_out"] = cache["h"].T @ d_logits[:, None]
        grads["b_out"] = np.array([np.sum(d_logits)], dtype=np.float64)

        dh = d_logits[:, None] @ self.params["W_out"].T
        d_pre_h = dh * (1.0 - cache["h"] ** 2)
        d_attn = d_pre_h @ cache["v"].T
        dv = cache["attn"].T @ d_pre_h
        dot = np.sum(d_attn * cache["attn"], axis=-1, keepdims=True)
        d_scores = cache["attn"] * (d_attn - dot) / cache["scale"]
        dq = d_scores @ cache["k"]
        dk = d_scores.T @ cache["q"]

        x = cache["x"]
        grads["W_q"] = x.T @ dq
        grads["W_k"] = x.T @ dk
        grads["W_v"] = x.T @ dv
        dx = d_pre_h + dq @ self.params["W_q"].T + dk @ self.params["W_k"].T + dv @ self.params["W_v"].T
        grads["W_in"] = cache["x_raw"].T @ dx
        opt.step(self.params, grads)
        return loss

    def fit_on_candidates(
        self,
        doc_candidates: list[tuple[AnnotatedDocument, list[CandidateEntity]]],
        *,
        epochs: int = 8,
        seed: int = 42,
    ) -> TransformerCandidateReranker:
        rng = np.random.default_rng(seed)
        opt = _AdamW(self.params, lr=0.015)
        items = [(d, c) for d, c in doc_candidates if c]
        for _ in range(epochs):
            for idx in rng.permutation(len(items)):
                doc, cands = items[int(idx)]
                gold_keys = {(e.type, e.start, e.end) for e in doc.entities}
                y = np.array(
                    [1.0 if (c.entity.type, c.entity.start, c.entity.end) in gold_keys else 0.0 for c in cands],
                    dtype=np.float64,
                )
                self.step_doc(cands, y, opt)
        return self

    def rerank(self, candidates: list[CandidateEntity], *, threshold: float = 0.25) -> list[CandidateEntity]:
        if not candidates:
            return []
        logits, _ = self.forward(self._featurize(candidates))
        scores = _sigmoid(logits)
        kept: list[CandidateEntity] = []
        for c, sc in zip(candidates, scores, strict=True):
            # Keep high-agreement rule/neural spans or candidates scoring >= threshold
            if sc >= threshold or c.source_stage == "bilstm_crf+rule_layer":
                blended_conf = round(float(0.5 * c.entity.confidence + 0.5 * sc), 4)
                kept.append(
                    CandidateEntity(
                        entity=PredictedEntity(
                            type=c.entity.type,
                            start=c.entity.start,
                            end=c.entity.end,
                            text=c.entity.text,
                            assertion=c.entity.assertion,
                            confidence=blended_conf,
                        ),
                        source_stage=f"{c.source_stage}->transformer_fusion",
                    )
                )
        return kept


class TemperatureScaler:
    """Fits temperature T > 0 on validation confidences to calibrate probabilities."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, confidences: list[float], correct_flags: list[int]) -> float:
        if not confidences:
            return 1.0
        best_t = 1.0
        best_nll = float("inf")
        p_arr = np.clip(np.array(confidences, dtype=np.float64), 1e-4, 1.0 - 1e-4)
        logits = np.log(p_arr / (1.0 - p_arr))
        y = np.array(correct_flags, dtype=np.float64)
        for t in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0):
            cal_p = _sigmoid(logits / t)
            nll = float(-np.mean(y * np.log(np.maximum(cal_p, 1e-12)) + (1.0 - y) * np.log(np.maximum(1.0 - cal_p, 1e-12))))
            if nll < best_nll:
                best_nll = nll
                best_t = t
        self.temperature = best_t
        return best_t

    def calibrate(self, conf: float) -> float:
        p = max(1e-4, min(1.0 - 1e-4, float(conf)))
        logit = float(np.log(p / (1.0 - p)))
        return round(float(_sigmoid(np.array([logit / self.temperature]))[0]), 4)


_LAB_RANGES: dict[str, tuple[float, float, str]] = {
    "hba1c": (3.0, 18.0, "%"),
    "crp": (0.0, 500.0, "mg/L"),
    "hemoglobin": (3.0, 22.0, "g/dL"),
    "hémoglobine": (3.0, 22.0, "g/dL"),
}


def check_consistency(
    entities: list[PredictedEntity],
    relations: list[PredictedRelation],
    timex: list[PredictedTimex],
    document_date: str,
) -> list[str]:
    """Run clinical consistency checks and return human-readable warnings."""
    warnings: list[str] = []

    # 1. Dose without medication
    meds = [e for e in entities if e.type == "MEDICATION"]
    for d in [e for e in entities if e.type == "DOSAGE"]:
        has_near_med = any(abs(d.start - m.end) <= 45 for m in meds)
        if not has_near_med:
            warnings.append(f"dose without medication at char {d.start} ({d.text})")

    # 2. Negated + affirmed same concept in same document
    by_norm: dict[str, set[str]] = {}
    for e in entities:
        if e.type in ("DIAGNOSIS", "SYMPTOM"):
            by_norm.setdefault(e.text.lower(), set()).add(e.assertion)
    for term, asserts in by_norm.items():
        if "present" in asserts and "absent" in asserts:
            warnings.append(f"conflicting assertions (present + absent) for '{term}'")

    # 3. Date after document_date
    try:
        doc_d = date.fromisoformat(document_date)
        for tx in timex:
            if tx.normalized and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tx.normalized):
                if date.fromisoformat(tx.normalized) > doc_d:
                    warnings.append(f"date {tx.normalized} is after document_date {document_date}")
    except ValueError:
        pass

    # 4. Lab value plausible range & unit mismatch
    ent_by_span = {(e.start, e.end): e for e in entities}
    for rel in relations:
        if rel.type == "HAS_VALUE":
            lab_e = ent_by_span.get((rel.head_start, rel.head_end))
            val_e = ent_by_span.get((rel.tail_start, rel.tail_end))
            if lab_e and val_e:
                spec = _LAB_RANGES.get(lab_e.text.lower())
                if spec:
                    lo, hi, expected_unit = spec
                    try:
                        v = float(val_e.text.replace(",", "."))
                        if v < lo or v > hi:
                            warnings.append(f"lab value {lab_e.text}={v} outside plausible range [{lo}, {hi}]")
                    except ValueError:
                        pass
                    for r2 in relations:
                        if r2.type == "HAS_UNIT" and (r2.head_start, r2.head_end) == (val_e.start, val_e.end):
                            unit_e = ent_by_span.get((r2.tail_start, r2.tail_end))
                            if unit_e and unit_e.text != expected_unit:
                                warnings.append(
                                    f"unit mismatch for {lab_e.text}: got '{unit_e.text}', expected '{expected_unit}'"
                                )
    return warnings


def detect_missing_sections(text: str, language: str) -> list[str]:
    """Identify expected clinical sections that are absent from the document."""
    low = text.lower()
    expected = (
        ("allergies", ("allerg",)),
        ("vital signs", ("vital", "blood pressure", "heart rate", "constantes", "tension")),
        ("medications", ("medication", "traitement", "prescription")),
        ("laboratory results", ("laboratory", "biologie", "hba1c", "crp")),
    )
    missing: list[str] = []
    for name, cues in expected:
        if not any(c in low for c in cues):
            missing.append(name)
    return missing
