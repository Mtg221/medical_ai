"""Stage 3: Learned BiLSTM Assertion Classifier + Piecewise 1D CNN Relation Classifier (pure NumPy)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from medresearch_platform.approaches.neural_seq import (
    _AdamW,
    _lstm_backward_seq,
    _lstm_forward_seq,
    _softmax,
)
from medresearch_platform.approaches.tokenizer import TrainOnlyVocabulary, tokenize_with_offsets
from medresearch_platform.data.schema import (
    ASSERTION_CLASSES,
    RELATION_TYPES,
    AnnotatedDocument,
)
from medresearch_platform.eval_harness.metrics import PredictedEntity, PredictedRelation

REL_LABELS: tuple[str, ...] = ("NO_REL",) + RELATION_TYPES
REL_TO_ID: dict[str, int] = {r: i for i, r in enumerate(REL_LABELS)}
ASSERT_TO_ID: dict[str, int] = {a: i for i, a in enumerate(ASSERTION_CLASSES)}

_PAIR_TO_REL_TYPES: dict[tuple[str, str], tuple[str, ...]] = {
    ("MEDICATION", "DOSAGE"): ("HAS_DOSAGE",),
    ("MEDICATION", "FREQUENCY"): ("HAS_FREQUENCY",),
    ("MEDICATION", "ROUTE"): ("HAS_ROUTE",),
    ("MEDICATION", "DURATION"): ("TREATS",),
    ("MEDICATION", "DIAGNOSIS"): ("TREATS",),
    ("LAB_TEST", "LAB_VALUE"): ("HAS_VALUE",),
    ("LAB_TEST", "LAB_UNIT"): ("HAS_UNIT",),
    ("LAB_VALUE", "LAB_UNIT"): ("HAS_UNIT",),
    ("PROCEDURE", "DIAGNOSIS"): ("REVEALS",),
    ("DIAGNOSIS", "PROCEDURE"): ("REVEALS",),
    ("DIAGNOSIS", "ANATOMY"): ("LOCATED_IN",),
    ("SYMPTOM", "ANATOMY"): ("LOCATED_IN",),
    ("DIAGNOSIS", "SYMPTOM"): ("CAUSES",),
}
_REL_ALLOWED_PAIRS: set[tuple[str, str]] = set(_PAIR_TO_REL_TYPES.keys())
_FUNCTIONAL_TAIL_REL_TYPES: frozenset[str] = frozenset(
    {"HAS_DOSAGE", "HAS_FREQUENCY", "HAS_ROUTE", "HAS_VALUE", "HAS_UNIT"}
)
_SENTENCE_BREAK_RE = re.compile(r"[.!?;]\s+")


def _has_intervening_same_type_head(
    entities: list[PredictedEntity],
    head: PredictedEntity,
    tail: PredictedEntity,
) -> bool:
    """Return True if another entity of `head.type` sits strictly between `head` and `tail`."""
    lo = min(head.end, tail.end)
    hi = max(head.start, tail.start)
    for other in entities:
        if other.start == head.start and other.end == head.end:
            continue
        if other.type == head.type and other.start >= lo and other.end <= hi:
            return True
    return False


class BiLSTMAssertionClassifier:
    """BiLSTM over entity-marked token windows predicting the 6 assertion classes."""

    def __init__(self, vocab_size: int, *, emb_dim: int = 20, hidden_dim: int = 24, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # +2 special marker tokens (<ENT>, </ENT>), +1 binary marker channel
        in_dim = emb_dim + 2
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        # Per-class calibrated probability thresholds (class 0 = present; >1.0 means abstain/defer to ConText)
        self.class_thresholds: dict[str, float] = {cls_name: 0.85 for cls_name in ASSERTION_CLASSES}
        self.params: dict[str, np.ndarray] = {
            "E": rng.normal(0.0, 0.1, size=(vocab_size + 2, emb_dim)),
            "Wf_x": rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, 4 * hidden_dim)),
            "Wf_h": rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, 4 * hidden_dim)),
            "bf": np.zeros(4 * hidden_dim, dtype=np.float64),
            "Wb_x": rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, 4 * hidden_dim)),
            "Wb_h": rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, 4 * hidden_dim)),
            "bb": np.zeros(4 * hidden_dim, dtype=np.float64),
            "W_cls": rng.normal(0.0, np.sqrt(1.0 / (2 * hidden_dim)), size=(2 * hidden_dim, len(ASSERTION_CLASSES))),
            "b_cls": np.zeros(len(ASSERTION_CLASSES), dtype=np.float64),
        }

    def _window_inputs(
        self,
        text: str,
        start: int,
        end: int,
        vocab: TrainOnlyVocabulary,
        toks: list[Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if toks is None:
            toks = tokenize_with_offsets(text)
        ent_idxs = [i for i, t in enumerate(toks) if max(t.start, start) < min(t.end, end)]
        if not ent_idxs:
            return np.array([1], dtype=np.int64), np.ones((1, 2), dtype=np.float64), 0
        lo = max(0, ent_idxs[0] - 12)
        hi = min(len(toks), ent_idxs[-1] + 6)
        w_ids: list[int] = []
        flags: list[list[float]] = []
        ent_start_pos = 0
        for i in range(lo, hi):
            if i == ent_idxs[0]:
                w_ids.append(self.vocab_size)  # <ENT>
                flags.append([1.0, 0.0])
                ent_start_pos = len(w_ids)
            w_ids.append(min(self.vocab_size - 1, vocab.encode_token(toks[i].text)))
            in_ent = 1.0 if i in ent_idxs else 0.0
            flags.append([in_ent, 1.0 if i < ent_idxs[0] else 0.0])
            if i == ent_idxs[-1]:
                w_ids.append(self.vocab_size + 1)  # </ENT>
                flags.append([1.0, 0.0])
        return np.array(w_ids, dtype=np.int64), np.array(flags, dtype=np.float64), ent_start_pos

    def forward(self, w_ids: np.ndarray, flags: np.ndarray, pool_idx: int) -> tuple[np.ndarray, dict[str, Any]]:
        x = np.concatenate([self.params["E"][w_ids], flags], axis=1)
        hf, cf = _lstm_forward_seq(x, self.params["Wf_x"], self.params["Wf_h"], self.params["bf"], reverse=False)
        hb, cb = _lstm_forward_seq(x, self.params["Wb_x"], self.params["Wb_h"], self.params["bb"], reverse=True)
        h_seq = np.concatenate([hf, hb], axis=1)
        # Pool entity-marker + mean context
        pooled = 0.7 * h_seq[min(pool_idx, len(h_seq) - 1)] + 0.3 * np.mean(h_seq, axis=0)
        logits = pooled @ self.params["W_cls"] + self.params["b_cls"]
        return logits, {"w_ids": w_ids, "h_seq": h_seq, "pooled": pooled, "pool_idx": min(pool_idx, len(h_seq) - 1), "cf": cf, "cb": cb}

    def loss_and_grads(
        self, w_ids: np.ndarray, flags: np.ndarray, pool_idx: int, target_id: int, weight: float
    ) -> tuple[float, dict[str, np.ndarray]]:
        logits, cache = self.forward(w_ids, flags, pool_idx)
        probs = _softmax(logits)
        loss = float(-weight * np.log(max(probs[target_id], 1e-12)))

        dlogits = probs.copy()
        dlogits[target_id] -= 1.0
        dlogits *= weight

        grads: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.params.items()}
        grads["W_cls"] = np.outer(cache["pooled"], dlogits)
        grads["b_cls"] = dlogits
        d_pooled = dlogits @ self.params["W_cls"].T

        n = len(w_ids)
        dh = np.broadcast_to((0.3 / n) * d_pooled, cache["h_seq"].shape).copy()
        dh[cache["pool_idx"]] += 0.7 * d_pooled

        hd = self.hidden_dim
        dx_f, grads["Wf_x"], grads["Wf_h"], grads["bf"] = _lstm_backward_seq(
            dh[:, :hd], cache["cf"], self.params["Wf_x"], self.params["Wf_h"], reverse=False
        )
        dx_b, grads["Wb_x"], grads["Wb_h"], grads["bb"] = _lstm_backward_seq(
            dh[:, hd:], cache["cb"], self.params["Wb_x"], self.params["Wb_h"], reverse=True
        )
        np.add.at(grads["E"], w_ids, (dx_f + dx_b)[:, : self.emb_dim])
        return loss, grads

    def step_example(
        self, w_ids: np.ndarray, flags: np.ndarray, pool_idx: int, target_id: int, weight: float, opt: _AdamW
    ) -> float:
        loss, grads = self.loss_and_grads(w_ids, flags, pool_idx, target_id, weight)
        opt.step(self.params, grads)
        return loss

    def fit(
        self,
        train_docs: list[AnnotatedDocument],
        vocab: TrainOnlyVocabulary,
        *,
        epochs: int = 8,
        max_examples: int = 1400,
        seed: int = 42,
    ) -> BiLSTMAssertionClassifier:
        rng = np.random.default_rng(seed)
        non_present: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        present: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        docs_shuffled = [train_docs[int(i)] for i in rng.permutation(len(train_docs))]
        for d in docs_shuffled:
            toks = tokenize_with_offsets(d.text)
            for e in d.entities:
                y = ASSERT_TO_ID[e.assertion]
                if y == 0 and len(present) >= max_examples:
                    continue
                w_ids, flags, pidx = self._window_inputs(d.text, e.start, e.end, vocab, toks=toks)
                if y == 0:
                    present.append((w_ids, flags, pidx, y))
                else:
                    non_present.append((w_ids, flags, pidx, y))
        present = [present[int(i)] for i in rng.permutation(len(present))]
        keep_present = present[: max(200, min(len(present), max_examples - len(non_present)))]
        examples = non_present + keep_present
        examples = [examples[int(i)] for i in rng.permutation(len(examples))]

        # Hold out 15% of examples to calibrate per-class precision thresholds so Stage 3b never hurts ConText
        n_cal = max(1, int(0.15 * len(examples))) if len(examples) >= 20 else 0
        cal_examples = examples[:n_cal]
        tr_examples = examples[n_cal:] if n_cal > 0 else examples

        counts = np.ones(len(ASSERTION_CLASSES), dtype=np.float64)
        for _, _, _, y in tr_examples:
            counts[y] += 1.0

        weights = np.clip((len(tr_examples) / (len(ASSERTION_CLASSES) * counts)) ** 0.75, 0.5, 8.0)
        opt = _AdamW(self.params, lr=0.015)
        for _ in range(epochs):
            for idx in rng.permutation(len(tr_examples)):
                w_ids, flags, pidx, y = tr_examples[int(idx)]
                self.step_example(w_ids, flags, pidx, y, float(weights[y]), opt)

        if cal_examples:
            cal_probs = np.array(
                [_softmax(self.forward(w, f, p)[0]) for w, f, p, _ in cal_examples],
                dtype=np.float64,
            )
            cal_y = np.array([y for _, _, _, y in cal_examples], dtype=np.int64)
            new_thresholds: dict[str, float] = {"present": 0.50}
            for cls_idx, cls_name in enumerate(ASSERTION_CLASSES):
                if cls_idx == 0:
                    continue
                chosen = 1.01  # abstain unless validation precision >= 0.80 with >= 2 hits
                for th in (0.65, 0.75, 0.82, 0.88, 0.92):
                    pred_mask = (np.argmax(cal_probs, axis=1) == cls_idx) & (cal_probs[:, cls_idx] >= th)
                    n_pred = int(np.sum(pred_mask))
                    if n_pred >= 2:
                        prec = float(np.sum(cal_y[pred_mask] == cls_idx)) / n_pred
                        if prec >= 0.80:
                            chosen = th
                            break
                new_thresholds[cls_name] = chosen
            self.class_thresholds = new_thresholds
        return self

    def predict_calibrated(
        self,
        text: str,
        start: int,
        end: int,
        vocab: TrainOnlyVocabulary,
        *,
        fallback_label: str = "present",
        toks: list[Any] | None = None,
    ) -> str:
        """Predict an assertion label only when calibrated class precision passes threshold."""
        probs = self.predict_proba(text, start, end, vocab, toks=toks)
        best_idx = int(np.argmax(probs))
        best_cls = ASSERTION_CLASSES[best_idx]
        if best_cls == "present":
            return fallback_label
        th = float(self.class_thresholds.get(best_cls, 0.85))
        return best_cls if float(probs[best_idx]) >= th else fallback_label

    def predict_proba(
        self,
        text: str,
        start: int,
        end: int,
        vocab: TrainOnlyVocabulary,
        toks: list[Any] | None = None,
    ) -> np.ndarray:
        w_ids, flags, pidx = self._window_inputs(text, start, end, vocab, toks=toks)
        logits, _ = self.forward(w_ids, flags, pidx)
        return _softmax(logits)

    def predict(
        self,
        text: str,
        start: int,
        end: int,
        vocab: TrainOnlyVocabulary,
        toks: list[Any] | None = None,
    ) -> str:
        probs = self.predict_proba(text, start, end, vocab, toks=toks)
        return ASSERTION_CLASSES[int(np.argmax(probs))]


class PiecewiseCNNRelationClassifier:
    """1D CNN with entity-relative position embeddings over candidate entity pairs."""

    def __init__(self, vocab_size: int, *, emb_dim: int = 16, pos_dim: int = 6, n_filters: int = 24, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.emb_dim = emb_dim
        self.pos_dim = pos_dim
        self.n_filters = n_filters
        self.max_pos = 64
        in_dim = emb_dim + 2 * pos_dim
        self.params: dict[str, np.ndarray] = {
            "E": rng.normal(0.0, 0.1, size=(vocab_size, emb_dim)),
            "P1": rng.normal(0.0, 0.1, size=(2 * self.max_pos + 1, pos_dim)),
            "P2": rng.normal(0.0, 0.1, size=(2 * self.max_pos + 1, pos_dim)),
            "W_conv": rng.normal(0.0, np.sqrt(1.0 / (3 * in_dim)), size=(3 * in_dim, n_filters)),
            "b_conv": np.zeros(n_filters, dtype=np.float64),
            "W_cls": rng.normal(0.0, np.sqrt(1.0 / n_filters), size=(n_filters, len(REL_LABELS))),
            "b_cls": np.zeros(len(REL_LABELS), dtype=np.float64),
        }

    def _pair_inputs(
        self, text: str, h_start: int, h_end: int, t_start: int, t_end: int, vocab: TrainOnlyVocabulary
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        span_lo = max(0, min(h_start, t_start) - 24)
        span_hi = min(len(text), max(h_end, t_end) + 24)
        toks = tokenize_with_offsets(text[span_lo:span_hi])
        if len(toks) < 3:
            w_ids = np.array([1, 1, 1], dtype=np.int64)
            p1 = np.full(3, self.max_pos, dtype=np.int64)
            p2 = np.full(3, self.max_pos, dtype=np.int64)
            return w_ids, p1, p2

        h_idx = min(range(len(toks)), key=lambda i: abs((span_lo + toks[i].start) - h_start))
        t_idx = min(range(len(toks)), key=lambda i: abs((span_lo + toks[i].start) - t_start))
        w_ids = np.array([vocab.encode_token(t.text) for t in toks], dtype=np.int64)
        p1 = np.array([np.clip(i - h_idx + self.max_pos, 0, 2 * self.max_pos) for i in range(len(toks))], dtype=np.int64)
        p2 = np.array([np.clip(i - t_idx + self.max_pos, 0, 2 * self.max_pos) for i in range(len(toks))], dtype=np.int64)
        return w_ids, p1, p2

    def forward(self, w_ids: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        x = np.concatenate([self.params["E"][w_ids], self.params["P1"][p1], self.params["P2"][p2]], axis=1)
        # 1D conv with kernel=3 (valid)
        windows = np.concatenate([x[:-2], x[1:-1], x[2:]], axis=1)
        pre = windows @ self.params["W_conv"] + self.params["b_conv"]
        act = np.maximum(0.0, pre)
        argmax_pos = np.argmax(act, axis=0)
        pooled = act[argmax_pos, np.arange(self.n_filters)]
        logits = pooled @ self.params["W_cls"] + self.params["b_cls"]
        return logits, {
            "w_ids": w_ids,
            "p1": p1,
            "p2": p2,
            "windows": windows,
            "pre": pre,
            "argmax_pos": argmax_pos,
            "pooled": pooled,
        }

    def loss_and_grads(
        self, w_ids: np.ndarray, p1: np.ndarray, p2: np.ndarray, target_id: int, weight: float
    ) -> tuple[float, dict[str, np.ndarray]]:
        logits, cache = self.forward(w_ids, p1, p2)
        probs = _softmax(logits)
        loss = float(-weight * np.log(max(probs[target_id], 1e-12)))

        dlogits = probs.copy()
        dlogits[target_id] -= 1.0
        dlogits *= weight

        grads: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.params.items()}
        grads["W_cls"] = np.outer(cache["pooled"], dlogits)
        grads["b_cls"] = dlogits
        d_pooled = dlogits @ self.params["W_cls"].T

        d_act = np.zeros_like(cache["pre"])
        d_act[cache["argmax_pos"], np.arange(self.n_filters)] = d_pooled
        d_pre = d_act * (cache["pre"] > 0.0)
        grads["W_conv"] = cache["windows"].T @ d_pre
        grads["b_conv"] = np.sum(d_pre, axis=0)

        d_win = d_pre @ self.params["W_conv"].T
        in_dim = self.emb_dim + 2 * self.pos_dim
        dx = np.zeros((len(w_ids), in_dim), dtype=np.float64)
        dx[:-2] += d_win[:, :in_dim]
        dx[1:-1] += d_win[:, in_dim : 2 * in_dim]
        dx[2:] += d_win[:, 2 * in_dim :]

        np.add.at(grads["E"], w_ids, dx[:, : self.emb_dim])
        np.add.at(grads["P1"], p1, dx[:, self.emb_dim : self.emb_dim + self.pos_dim])
        np.add.at(grads["P2"], p2, dx[:, self.emb_dim + self.pos_dim :])
        return loss, grads

    def step_example(
        self, w_ids: np.ndarray, p1: np.ndarray, p2: np.ndarray, target_id: int, weight: float, opt: _AdamW
    ) -> float:
        loss, grads = self.loss_and_grads(w_ids, p1, p2, target_id, weight)
        opt.step(self.params, grads)
        return loss

    def fit(
        self,
        train_docs: list[AnnotatedDocument],
        vocab: TrainOnlyVocabulary,
        *,
        epochs: int = 8,
        max_examples: int = 900,
        seed: int = 42,
    ) -> PiecewiseCNNRelationClassifier:
        rng = np.random.default_rng(seed)
        examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        counts = np.ones(len(REL_LABELS), dtype=np.float64)

        for d in train_docs[:150]:
            gold_rel_map = {(r.head, r.tail): r.type for r in d.relations}
            for head in d.entities:
                for tail in d.entities:
                    if head.id == tail.id or abs(head.start - tail.start) > 140:
                        continue
                    between = d.text[min(head.end, tail.end) : max(head.start, tail.start)]
                    if "\n" in between:
                        continue
                    if (head.type, tail.type) not in _REL_ALLOWED_PAIRS and (head.id, tail.id) not in gold_rel_map:
                        continue
                    lbl = gold_rel_map.get((head.id, tail.id), "NO_REL")
                    if lbl == "NO_REL" and rng.random() > 0.35:
                        continue
                    w_ids, p1, p2 = self._pair_inputs(d.text, head.start, head.end, tail.start, tail.end, vocab)
                    y = REL_TO_ID[lbl]
                    examples.append((w_ids, p1, p2, y))
                    counts[y] += 1.0
                    if len(examples) >= max_examples:
                        break
                if len(examples) >= max_examples:
                    break
            if len(examples) >= max_examples:
                break

        weights = np.clip(np.sqrt(len(examples) / (len(REL_LABELS) * counts)), 0.5, 4.0)
        opt = _AdamW(self.params, lr=0.015)
        for _ in range(epochs):
            for idx in rng.permutation(len(examples)):
                w_ids, p1, p2, y = examples[int(idx)]
                self.step_example(w_ids, p1, p2, y, float(weights[y]), opt)
        return self

    def predict_relations(
        self, text: str, entities: list[PredictedEntity], vocab: TrainOnlyVocabulary
    ) -> list[PredictedRelation]:
        candidates: list[tuple[PredictedRelation, int, int]] = []
        for i, head in enumerate(entities):
            for j, tail in enumerate(entities):
                if i == j or abs(head.start - tail.start) > 140:
                    continue
                if (head.type, tail.type) not in _REL_ALLOWED_PAIRS:
                    continue
                between = text[min(head.end, tail.end) : max(head.start, tail.start)]
                if "\n" in between or _SENTENCE_BREAK_RE.search(between):
                    continue
                allowed_rels = _PAIR_TO_REL_TYPES[(head.type, tail.type)]
                if all(r in _FUNCTIONAL_TAIL_REL_TYPES for r in allowed_rels):
                    if _has_intervening_same_type_head(entities, head, tail):
                        continue
                w_ids, p1, p2 = self._pair_inputs(text, head.start, head.end, tail.start, tail.end, vocab)
                logits, _ = self.forward(w_ids, p1, p2)
                allowed_names = ("NO_REL",) + allowed_rels
                allowed_ids = [REL_TO_ID[n] for n in allowed_names]
                masked_logits = np.full_like(logits, -1e9)
                for aid in allowed_ids:
                    masked_logits[aid] = logits[aid]
                probs = _softmax(masked_logits)
                pred_id = int(np.argmax(probs))
                if pred_id != 0 and float(probs[pred_id]) >= 0.50:
                    rel_name = REL_LABELS[pred_id]
                    is_preceding = 0 if head.end <= tail.start else 1
                    char_gap = abs(tail.start - head.end) if head.end <= tail.start else abs(head.start - tail.end)
                    candidates.append(
                        (
                            PredictedRelation(
                                type=rel_name,
                                head_start=head.start,
                                head_end=head.end,
                                tail_start=tail.start,
                                tail_end=tail.end,
                                confidence=round(float(probs[pred_id]), 4),
                            ),
                            is_preceding,
                            char_gap,
                        )
                    )

        # Deduplicate functional modifier tails (each dosage/frequency/route/value/unit attaches to its nearest valid head)
        rels: list[PredictedRelation] = []
        best_by_tail: dict[tuple[str, int, int], tuple[int, int, float, PredictedRelation]] = {}
        for rel, is_prec, gap in candidates:
            if rel.type in _FUNCTIONAL_TAIL_REL_TYPES:
                key = (rel.type, rel.tail_start, rel.tail_end)
                rank = (is_prec, gap, -rel.confidence)
                prev = best_by_tail.get(key)
                if prev is None or rank < (prev[0], prev[1], prev[2]):
                    best_by_tail[key] = (rank[0], rank[1], rank[2], rel)
            else:
                rels.append(rel)
        rels.extend(entry[3] for entry in best_by_tail.values())
        return rels
