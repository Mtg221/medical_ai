"""Phase 3: Pure-NumPy random-initialized BiLSTM, BiLSTM-CRF, and Transformer sequence taggers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from medresearch_platform.approaches.context_negex import apply_context_and_relations
from medresearch_platform.approaches.tokenizer import (
    BIO_TAGS,
    ID_TO_TAG,
    TAG_TO_ID,
    Token,
    TrainOnlyVocabulary,
    align_entities_to_bio,
    decode_bio_to_entities,
    tokenize_with_offsets,
)
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import DocumentPrediction

NUM_TAGS = len(BIO_TAGS)
CHAR_FEAT_DIM = 32


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.maximum(np.sum(exps, axis=-1, keepdims=True), 1e-12)


def _ortho_features(tokens: list[Token]) -> np.ndarray:
    """32-dim deterministic character/orthographic + prefix/suffix morphology vector per token."""
    feats = np.zeros((len(tokens), CHAR_FEAT_DIM), dtype=np.float64)
    for i, tok in enumerate(tokens):
        w = tok.text
        wl = w.lower()
        feats[i, 0] = 1.0 if w.isupper() else 0.0
        feats[i, 1] = 1.0 if w.istitle() else 0.0
        feats[i, 2] = 1.0 if w.isdigit() else 0.0
        feats[i, 3] = 1.0 if any(c.isdigit() for c in w) else 0.0
        feats[i, 4] = 1.0 if "-" in w else 0.0
        feats[i, 5] = 1.0 if "/" in w or "%" in w else 0.0
        feats[i, 6] = min(len(w), 20) / 20.0
        feats[i, 7] = 1.0 if wl in ("mg", "mcg", "g", "ml", "l", "%", "ui", "mmol", "umol") else 0.0
        # Prefix n-gram hash buckets (dims 8..15)
        for n in (2, 3):
            if len(wl) >= n:
                pref = wl[:n]
                h_p = sum((idx + 1) * ord(ch) for idx, ch in enumerate(pref))
                feats[i, 8 + (h_p % 8)] += 0.5
        # Suffix n-gram hash buckets (dims 16..31) — captures medical suffixes (-itis, -statin, -émie, ...)
        for n in (2, 3, 4):
            if len(wl) >= n:
                suff = wl[-n:]
                h_s = sum((idx + 3) * ord(ch) for idx, ch in enumerate(suff))
                feats[i, 16 + (h_s % 16)] += 0.5
    return feats


def _compute_class_weights(encoded_seqs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    counts = np.ones(NUM_TAGS, dtype=np.float64)
    for _, _, y in encoded_seqs:
        for t in y:
            counts[t] += 1.0
    total = float(np.sum(counts))
    weights = np.sqrt(total / (NUM_TAGS * counts))
    weights[0] = min(weights[0], 0.35)  # Downweight dominant 'O' class
    weights[1:] = np.clip(weights[1:], 1.0, 4.0)
    return weights


class _AdamW:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 0.01, weight_decay: float = 1e-4) -> None:
        self.lr = lr
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            g_clip = np.clip(g, -5.0, 5.0)
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g_clip
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * (g_clip * g_clip)
            m_hat = self.m[k] / (1.0 - b1**self.t)
            v_hat = self.v[k] / (1.0 - b2**self.t)
            params[k] -= self.lr * (m_hat / (np.sqrt(v_hat) + eps) + self.wd * params[k])


def _encode_docs(
    docs: list[AnnotatedDocument], vocab: TrainOnlyVocabulary
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    seqs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for d in docs:
        toks = tokenize_with_offsets(d.text)
        if not toks:
            continue
        tags = align_entities_to_bio(toks, d.entities)
        w_ids = np.array([vocab.encode_token(t.text) for t in toks], dtype=np.int64)
        c_feats = _ortho_features(toks)
        y_ids = np.array([TAG_TO_ID[t] for t in tags], dtype=np.int64)
        seqs.append((w_ids, c_feats, y_ids))
    return seqs


def _lstm_forward_seq(
    x: np.ndarray, w_x: np.ndarray, w_h: np.ndarray, b: np.ndarray, reverse: bool = False
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    seq_len = x.shape[0]
    h_dim = w_h.shape[0]
    h = np.zeros(h_dim, dtype=np.float64)
    c = np.zeros(h_dim, dtype=np.float64)
    hs = np.zeros((seq_len, h_dim), dtype=np.float64)
    caches: list[dict[str, np.ndarray]] = []
    order = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

    for t in order:
        z = x[t] @ w_x + h @ w_h + b
        i = _sigmoid(z[:h_dim])
        f = _sigmoid(z[h_dim : 2 * h_dim])
        o = _sigmoid(z[2 * h_dim : 3 * h_dim])
        g = np.tanh(z[3 * h_dim :])
        c_next = f * c + i * g
        tanh_c = np.tanh(c_next)
        h_next = o * tanh_c
        hs[t] = h_next
        caches.append({"x": x[t], "h_prev": h, "c_prev": c, "i": i, "f": f, "o": o, "g": g, "tanh_c": tanh_c})
        h, c = h_next, c_next

    if reverse:
        caches.reverse()
    return hs, caches


def _lstm_backward_seq(
    d_hs: np.ndarray,
    caches: list[dict[str, np.ndarray]],
    w_x: np.ndarray,
    w_h: np.ndarray,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq_len = len(caches)
    h_dim = w_h.shape[0]
    dx = np.zeros((seq_len, w_x.shape[0]), dtype=np.float64)
    dw_x = np.zeros_like(w_x)
    dw_h = np.zeros_like(w_h)
    db = np.zeros(4 * h_dim, dtype=np.float64)
    dh_next = np.zeros(h_dim, dtype=np.float64)
    dc_next = np.zeros(h_dim, dtype=np.float64)

    order = range(seq_len) if reverse else range(seq_len - 1, -1, -1)
    for t in order:
        cache = caches[t]
        dh = d_hs[t] + dh_next
        do = dh * cache["tanh_c"] * cache["o"] * (1.0 - cache["o"])
        dc = dh * cache["o"] * (1.0 - cache["tanh_c"] ** 2) + dc_next
        di = dc * cache["g"] * cache["i"] * (1.0 - cache["i"])
        df = dc * cache["c_prev"] * cache["f"] * (1.0 - cache["f"])
        dg = dc * cache["i"] * (1.0 - cache["g"] ** 2)
        dz = np.concatenate([di, df, do, dg])
        dw_x += np.outer(cache["x"], dz)
        dw_h += np.outer(cache["h_prev"], dz)
        db += dz
        dx[t] = dz @ w_x.T
        dh_next = dz @ w_h.T
        dc_next = dc * cache["f"]
    return dx, dw_x, dw_h, db


class NeuralBiLSTMTagger:
    """Random-initialized PyTorch-style BiLSTM sequence tagger implemented in NumPy."""

    def __init__(self, vocab_size: int, *, emb_dim: int = 24, hidden_dim: int = 32, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        in_dim = emb_dim + CHAR_FEAT_DIM
        self.params: dict[str, np.ndarray] = {
            "E": rng.normal(0.0, 0.1, size=(vocab_size, emb_dim)),
            "Wf_x": rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, 4 * hidden_dim)),
            "Wf_h": rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, 4 * hidden_dim)),
            "bf": np.zeros(4 * hidden_dim, dtype=np.float64),
            "Wb_x": rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, 4 * hidden_dim)),
            "Wb_h": rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, 4 * hidden_dim)),
            "bb": np.zeros(4 * hidden_dim, dtype=np.float64),
            "W_out": rng.normal(0.0, np.sqrt(1.0 / (2 * hidden_dim)), size=(2 * hidden_dim, NUM_TAGS)),
            "b_out": np.zeros(NUM_TAGS, dtype=np.float64),
        }
        self.emb_dim = emb_dim

    def forward_logits(self, w_ids: np.ndarray, c_feats: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        emb = self.params["E"][w_ids]
        x = np.concatenate([emb, c_feats], axis=1)
        hf, cf = _lstm_forward_seq(x, self.params["Wf_x"], self.params["Wf_h"], self.params["bf"], reverse=False)
        hb, cb = _lstm_forward_seq(x, self.params["Wb_x"], self.params["Wb_h"], self.params["bb"], reverse=True)
        h = np.concatenate([hf, hb], axis=1)
        logits = h @ self.params["W_out"] + self.params["b_out"]
        return logits, {"w_ids": w_ids, "h": h, "cf": cf, "cb": cb, "h_dim": hf.shape[1]}

    def step_sequence(
        self,
        w_ids: np.ndarray,
        c_feats: np.ndarray,
        y_ids: np.ndarray,
        class_weights: np.ndarray,
        opt: _AdamW,
    ) -> float:
        logits, cache = self.forward_logits(w_ids, c_feats)
        probs = _softmax(logits)
        n = len(y_ids)
        cw = class_weights[y_ids]
        loss = float(-np.mean(cw * np.log(np.maximum(probs[np.arange(n), y_ids], 1e-12))))

        dlogits = probs.copy()
        dlogits[np.arange(n), y_ids] -= 1.0
        dlogits *= (cw[:, None] / max(1, n))

        grads: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.params.items()}
        grads["W_out"] = cache["h"].T @ dlogits
        grads["b_out"] = np.sum(dlogits, axis=0)
        dh = dlogits @ self.params["W_out"].T
        hd = cache["h_dim"]

        dx_f, grads["Wf_x"], grads["Wf_h"], grads["bf"] = _lstm_backward_seq(
            dh[:, :hd], cache["cf"], self.params["Wf_x"], self.params["Wf_h"], reverse=False
        )
        dx_b, grads["Wb_x"], grads["Wb_h"], grads["bb"] = _lstm_backward_seq(
            dh[:, hd:], cache["cb"], self.params["Wb_x"], self.params["Wb_h"], reverse=True
        )
        dx = dx_f + dx_b
        np.add.at(grads["E"], w_ids, dx[:, : self.emb_dim])
        opt.step(self.params, grads)
        return loss

    def predict(self, w_ids: np.ndarray, c_feats: np.ndarray) -> tuple[list[str], list[float]]:
        logits, _ = self.forward_logits(w_ids, c_feats)
        probs = _softmax(logits)
        pred_ids = np.argmax(probs, axis=-1)
        return [ID_TO_TAG[int(i)] for i in pred_ids], [float(probs[t, pred_ids[t]]) for t in range(len(pred_ids))]


class NeuralBiLSTMCRFTagger(NeuralBiLSTMTagger):
    """Random-initialized BiLSTM + linear-chain CRF with BIO constraints & Viterbi decoding."""

    def __init__(self, vocab_size: int, *, emb_dim: int = 24, hidden_dim: int = 32, seed: int = 42) -> None:
        super().__init__(vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim, seed=seed)
        rng = np.random.default_rng(seed + 1)
        trans = rng.normal(0.0, 0.1, size=(NUM_TAGS, NUM_TAGS))
        # Penalize illegal BIO transitions (O -> I-X, B-X -> I-Y, I-X -> I-Y)
        for i, prev_tag in enumerate(BIO_TAGS):
            for j, cur_tag in enumerate(BIO_TAGS):
                if cur_tag.startswith("I-"):
                    etype = cur_tag[2:]
                    if prev_tag not in (f"B-{etype}", f"I-{etype}"):
                        trans[i, j] = -6.0
        self.params["T"] = trans

    def _viterbi(self, emit: np.ndarray) -> list[int]:
        seq_len = emit.shape[0]
        trans = self.params["T"]
        dp = emit[0].copy()
        backptrs = np.zeros((seq_len, NUM_TAGS), dtype=np.int64)
        for t in range(1, seq_len):
            scores = dp[:, None] + trans + emit[t][None, :]
            backptrs[t] = np.argmax(scores, axis=0)
            dp = np.max(scores, axis=0)
        best = [int(np.argmax(dp))]
        for t in range(seq_len - 1, 0, -1):
            best.append(int(backptrs[t, best[-1]]))
        best.reverse()
        return best

    def step_sequence(
        self,
        w_ids: np.ndarray,
        c_feats: np.ndarray,
        y_ids: np.ndarray,
        class_weights: np.ndarray,
        opt: _AdamW,
    ) -> float:
        emit, cache = self.forward_logits(w_ids, c_feats)
        seq_len = emit.shape[0]
        trans = self.params["T"]

        # Forward-backward log-marginals for exact CRF NLL gradient
        alpha = np.zeros((seq_len, NUM_TAGS), dtype=np.float64)
        alpha[0] = emit[0]
        for t in range(1, seq_len):
            s = alpha[t - 1][:, None] + trans + emit[t][None, :]
            m = np.max(s, axis=0)
            alpha[t] = m + np.log(np.sum(np.exp(s - m), axis=0))
        m_last = np.max(alpha[-1])
        log_z = float(m_last + np.log(np.sum(np.exp(alpha[-1] - m_last))))

        beta = np.zeros((seq_len, NUM_TAGS), dtype=np.float64)
        for t in range(seq_len - 2, -1, -1):
            s = trans + emit[t + 1][None, :] + beta[t + 1][None, :]
            m = np.max(s, axis=1)
            beta[t] = m + np.log(np.sum(np.exp(s - m[:, None]), axis=1))

        gold_score = float(emit[0, y_ids[0]])
        for t in range(1, seq_len):
            gold_score += float(trans[y_ids[t - 1], y_ids[t]] + emit[t, y_ids[t]])
        nll = (log_z - gold_score) / max(1, seq_len)

        # Expected node marginals
        log_marg = alpha + beta - log_z
        marg = np.exp(np.clip(log_marg, -30.0, 0.0))
        d_emit = marg.copy()
        d_emit[np.arange(seq_len), y_ids] -= 1.0
        d_emit /= max(1, seq_len)

        d_trans = np.zeros_like(trans)
        for t in range(1, seq_len):
            pair_log = alpha[t - 1][:, None] + trans + emit[t][None, :] + beta[t][None, :] - log_z
            d_trans += np.exp(np.clip(pair_log, -30.0, 0.0))
            d_trans[y_ids[t - 1], y_ids[t]] -= 1.0
        d_trans /= max(1, seq_len)

        grads: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.params.items()}
        grads["T"] = d_trans
        grads["W_out"] = cache["h"].T @ d_emit
        grads["b_out"] = np.sum(d_emit, axis=0)
        dh = d_emit @ self.params["W_out"].T
        hd = cache["h_dim"]

        dx_f, grads["Wf_x"], grads["Wf_h"], grads["bf"] = _lstm_backward_seq(
            dh[:, :hd], cache["cf"], self.params["Wf_x"], self.params["Wf_h"], reverse=False
        )
        dx_b, grads["Wb_x"], grads["Wb_h"], grads["bb"] = _lstm_backward_seq(
            dh[:, hd:], cache["cb"], self.params["Wb_x"], self.params["Wb_h"], reverse=True
        )
        np.add.at(grads["E"], w_ids, (dx_f + dx_b)[:, : self.emb_dim])
        opt.step(self.params, grads)
        return nll

    def predict(self, w_ids: np.ndarray, c_feats: np.ndarray) -> tuple[list[str], list[float]]:
        emit, _ = self.forward_logits(w_ids, c_feats)
        probs = _softmax(emit)
        best_ids = self._viterbi(emit)
        return [ID_TO_TAG[i] for i in best_ids], [float(probs[t, best_ids[t]]) for t in range(len(best_ids))]


class NeuralTransformerTagger:
    """Random-initialized single-block Transformer encoder (self-attention + FFN) in NumPy."""

    def __init__(self, vocab_size: int, *, d_model: int = 32, max_len: int = 256, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        in_dim = d_model + CHAR_FEAT_DIM
        self.d_model = d_model
        self.max_len = max_len
        pos = np.zeros((max_len, d_model), dtype=np.float64)
        positions = np.arange(max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pos[:, 0::2] = np.sin(positions * div)
        pos[:, 1::2] = np.cos(positions * div)
        self.pos_enc = pos

        self.params: dict[str, np.ndarray] = {
            "E": rng.normal(0.0, 0.1, size=(vocab_size, d_model)),
            "W_in": rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(in_dim, d_model)),
            "W_q": rng.normal(0.0, np.sqrt(1.0 / d_model), size=(d_model, d_model)),
            "W_k": rng.normal(0.0, np.sqrt(1.0 / d_model), size=(d_model, d_model)),
            "W_v": rng.normal(0.0, np.sqrt(1.0 / d_model), size=(d_model, d_model)),
            "W_ff1": rng.normal(0.0, np.sqrt(1.0 / d_model), size=(d_model, 2 * d_model)),
            "b_ff1": np.zeros(2 * d_model, dtype=np.float64),
            "W_ff2": rng.normal(0.0, np.sqrt(1.0 / (2 * d_model)), size=(2 * d_model, d_model)),
            "b_ff2": np.zeros(d_model, dtype=np.float64),
            "W_out": rng.normal(0.0, np.sqrt(1.0 / d_model), size=(d_model, NUM_TAGS)),
            "b_out": np.zeros(NUM_TAGS, dtype=np.float64),
        }

    def forward_logits(self, w_ids: np.ndarray, c_feats: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        n = min(len(w_ids), self.max_len)
        w_sl = w_ids[:n]
        c_sl = c_feats[:n]
        x_raw = np.concatenate([self.params["E"][w_sl], c_sl], axis=1)
        x = x_raw @ self.params["W_in"] + self.pos_enc[:n]
        scale = np.sqrt(float(self.d_model))
        q = x @ self.params["W_q"]
        k = x @ self.params["W_k"]
        v = x @ self.params["W_v"]
        attn_scores = (q @ k.T) / scale
        attn = _softmax(attn_scores)
        h_attn = x + attn @ v
        ff_pre = h_attn @ self.params["W_ff1"] + self.params["b_ff1"]
        ff_act = np.maximum(0.0, ff_pre)
        h_out = h_attn + ff_act @ self.params["W_ff2"] + self.params["b_ff2"]
        logits = h_out @ self.params["W_out"] + self.params["b_out"]
        return logits, {
            "w_sl": w_sl,
            "x_raw": x_raw,
            "x": x,
            "q": q,
            "k": k,
            "v": v,
            "attn": attn,
            "h_attn": h_attn,
            "ff_pre": ff_pre,
            "ff_act": ff_act,
            "h_out": h_out,
            "scale": scale,
        }

    def step_sequence(
        self,
        w_ids: np.ndarray,
        c_feats: np.ndarray,
        y_ids: np.ndarray,
        class_weights: np.ndarray,
        opt: _AdamW,
    ) -> float:
        logits, cache = self.forward_logits(w_ids, c_feats)
        n = logits.shape[0]
        y_sl = y_ids[:n]
        probs = _softmax(logits)
        cw = class_weights[y_sl]
        loss = float(-np.mean(cw * np.log(np.maximum(probs[np.arange(n), y_sl], 1e-12))))

        dlogits = probs.copy()
        dlogits[np.arange(n), y_sl] -= 1.0
        dlogits *= (cw[:, None] / max(1, n))

        grads: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.params.items()}
        grads["W_out"] = cache["h_out"].T @ dlogits
        grads["b_out"] = np.sum(dlogits, axis=0)

        dh_out = dlogits @ self.params["W_out"].T
        dff_act = dh_out @ self.params["W_ff2"].T
        grads["W_ff2"] = cache["ff_act"].T @ dh_out
        grads["b_ff2"] = np.sum(dh_out, axis=0)

        dff_pre = dff_act * (cache["ff_pre"] > 0.0)
        grads["W_ff1"] = cache["h_attn"].T @ dff_pre
        grads["b_ff1"] = np.sum(dff_pre, axis=0)

        dh_attn = dh_out + dff_pre @ self.params["W_ff1"].T
        d_attn_v = dh_attn
        d_attn = d_attn_v @ cache["v"].T
        dv = cache["attn"].T @ dh_attn

        # Softmax Jacobian per row
        dot = np.sum(d_attn * cache["attn"], axis=-1, keepdims=True)
        d_scores = cache["attn"] * (d_attn - dot) / cache["scale"]
        dq = d_scores @ cache["k"]
        dk = d_scores.T @ cache["q"]

        x = cache["x"]
        grads["W_q"] = x.T @ dq
        grads["W_k"] = x.T @ dk
        grads["W_v"] = x.T @ dv
        dx = dh_attn + dq @ self.params["W_q"].T + dk @ self.params["W_k"].T + dv @ self.params["W_v"].T

        grads["W_in"] = cache["x_raw"].T @ dx
        dx_raw = dx @ self.params["W_in"].T
        np.add.at(grads["E"], cache["w_sl"], dx_raw[:, : self.d_model])
        opt.step(self.params, grads)
        return loss

    def predict(self, w_ids: np.ndarray, c_feats: np.ndarray) -> tuple[list[str], list[float]]:
        if len(w_ids) <= self.max_len:
            logits, _ = self.forward_logits(w_ids, c_feats)
            probs = _softmax(logits)
            pred_ids = np.argmax(probs, axis=-1)
            return [ID_TO_TAG[int(i)] for i in pred_ids], [float(probs[t, pred_ids[t]]) for t in range(len(pred_ids))]
        # Sliding non-overlapping windows for long documents
        all_tags: list[str] = []
        all_confs: list[float] = []
        for start in range(0, len(w_ids), self.max_len):
            logits, _ = self.forward_logits(w_ids[start : start + self.max_len], c_feats[start : start + self.max_len])
            probs = _softmax(logits)
            pred_ids = np.argmax(probs, axis=-1)
            all_tags.extend(ID_TO_TAG[int(i)] for i in pred_ids)
            all_confs.extend(float(probs[t, pred_ids[t]]) for t in range(len(pred_ids)))
        return all_tags, all_confs


@dataclass
class TrainConfig:
    epochs: int = 12
    lr: float = 0.015
    hidden_dim: int = 32
    max_train_docs: int = 120
    patience: int = 3


def _train_and_predict_neural(
    model_kind: str,
    train_docs: list[AnnotatedDocument],
    eval_docs: list[AnnotatedDocument],
    seed: int,
    cfg: dict[str, Any],
) -> list[DocumentPrediction]:
    vocab = TrainOnlyVocabulary(max_size=6000, min_freq=1).fit(train_docs)
    rng = np.random.default_rng(seed)

    # Balanced EN/FR subset from train_docs only (never touches test)
    shuffled_train = list(train_docs)
    rng.shuffle(shuffled_train)
    train_subset = shuffled_train[: int(cfg.get("max_train_docs", 160))]
    n_es = max(1, int(0.12 * len(train_subset)))
    es_docs = train_subset[:n_es]
    tr_docs = train_subset[n_es:] if len(train_subset) > n_es else train_subset

    tr_seqs = _encode_docs(tr_docs, vocab)
    es_seqs = _encode_docs(es_docs, vocab)
    cw = _compute_class_weights(tr_seqs)

    h_dim = int(cfg.get("hidden_dim", 32))
    lr = float(cfg.get("lr", 0.015))
    epochs = int(cfg.get("epochs", 10))
    patience = int(cfg.get("patience", 3))

    model: NeuralBiLSTMTagger | NeuralBiLSTMCRFTagger | NeuralTransformerTagger
    if model_kind == "bilstm":
        model = NeuralBiLSTMTagger(len(vocab), emb_dim=24, hidden_dim=h_dim, seed=seed)
    elif model_kind == "bilstm_crf":
        model = NeuralBiLSTMCRFTagger(len(vocab), emb_dim=24, hidden_dim=h_dim, seed=seed)
    elif model_kind == "transformer":
        model = NeuralTransformerTagger(len(vocab), d_model=h_dim, seed=seed)
    else:
        raise ValueError(f"Unknown neural model_kind {model_kind!r}")

    opt = _AdamW(model.params, lr=lr)
    best_es_loss = float("inf")
    best_params = {k: v.copy() for k, v in model.params.items()}
    stale = 0

    dropout_p = float(cfg.get("dropout", 0.15))
    for _epoch in range(epochs):
        order = rng.permutation(len(tr_seqs))
        for idx in order:
            w_ids, c_feats, y_ids = tr_seqs[int(idx)]
            if dropout_p > 0.0:
                # Seeded word-dropout (<UNK>=1) + inverted char-feature dropout for generalization
                keep_w = rng.random(len(w_ids)) >= (dropout_p * 0.5)
                w_step = np.where(keep_w, w_ids, 1)
                c_mask = (rng.random(c_feats.shape) >= dropout_p).astype(np.float64)
                c_step = (c_feats * c_mask) / max(1.0 - dropout_p, 1e-6)
            else:
                w_step, c_step = w_ids, c_feats
            model.step_sequence(w_step, c_step, y_ids, cw, opt)

        # Check early stopping on train-internal val slice
        es_losses: list[float] = []
        for w_ids, c_feats, y_ids in es_seqs:
            logits, _ = model.forward_logits(w_ids, c_feats)
            probs = _softmax(logits[: len(y_ids)])
            y_sl = y_ids[: len(probs)]
            es_losses.append(float(-np.mean(cw[y_sl] * np.log(np.maximum(probs[np.arange(len(y_sl)), y_sl], 1e-12)))))
        mean_es = sum(es_losses) / max(1, len(es_losses))
        if mean_es < best_es_loss - 1e-4:
            best_es_loss = mean_es
            best_params = {k: v.copy() for k, v in model.params.items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.params = best_params

    # Train-only document type classifier (never reads eval doc.doc_type)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    train_dtypes = [d.doc_type for d in train_docs]
    if len(set(train_dtypes)) >= 2:
        doc_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
        x_doc_tr = doc_vec.fit_transform([d.text for d in train_docs])
        x_doc_tr.indices = x_doc_tr.indices.astype(np.int32, copy=False)
        x_doc_tr.indptr = x_doc_tr.indptr.astype(np.int32, copy=False)
        doc_clf = LinearSVC(random_state=seed, class_weight="balanced").fit(x_doc_tr, train_dtypes)
        x_doc_ev = doc_vec.transform([d.text for d in eval_docs])
        x_doc_ev.indices = x_doc_ev.indices.astype(np.int32, copy=False)
        x_doc_ev.indptr = x_doc_ev.indptr.astype(np.int32, copy=False)
        eval_doc_preds = [str(p) for p in doc_clf.predict(x_doc_ev)]
    else:
        fallback_dt = train_dtypes[0] if train_dtypes else "discharge_summary"
        eval_doc_preds = [fallback_dt for _ in eval_docs]

    preds: list[DocumentPrediction] = []
    for d, pred_doc_type in zip(eval_docs, eval_doc_preds, strict=True):
        toks = tokenize_with_offsets(d.text)
        if not toks:
            preds.append(
                DocumentPrediction(
                    d.doc_id, d.language, d.doc_type, predicted_doc_type=pred_doc_type
                )
            )
            continue
        w_ids = np.array([vocab.encode_token(t.text) for t in toks], dtype=np.int64)
        c_feats = _ortho_features(toks)
        pred_tags, confs = model.predict(w_ids, c_feats)
        ents = decode_bio_to_entities(d.text, toks, pred_tags, confs)
        enriched, relations, timex = apply_context_and_relations(d, ents)
        preds.append(
            DocumentPrediction(
                doc_id=d.doc_id,
                language=d.language,
                doc_type=d.doc_type,
                predicted_doc_type=pred_doc_type,
                entities=enriched,
                relations=relations,
                timex=timex,
            )
        )
    return preds


def predict_bilstm(
    train_docs: list[AnnotatedDocument], eval_docs: list[AnnotatedDocument], seed: int, cfg: dict[str, Any]
) -> list[DocumentPrediction]:
    return _train_and_predict_neural("bilstm", train_docs, eval_docs, seed, cfg)


def predict_bilstm_crf(
    train_docs: list[AnnotatedDocument], eval_docs: list[AnnotatedDocument], seed: int, cfg: dict[str, Any]
) -> list[DocumentPrediction]:
    return _train_and_predict_neural("bilstm_crf", train_docs, eval_docs, seed, cfg)


def predict_transformer(
    train_docs: list[AnnotatedDocument], eval_docs: list[AnnotatedDocument], seed: int, cfg: dict[str, Any]
) -> list[DocumentPrediction]:
    return _train_and_predict_neural("transformer", train_docs, eval_docs, seed, cfg)
