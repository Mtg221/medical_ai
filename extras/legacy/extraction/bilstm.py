"""Approach 3: Pure-NumPy Randomly Initialized BiLSTM Sequence Tagger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from medresearch_platform.ai.random_init import (
    RandomInitManifest,
    build_random_init_manifest,
    initialize_tensor,
)
from medresearch_platform.ai.terminology import BilingualTerminologyRegistry
from medresearch_platform.extraction.contract import ExtractionResult, records_from_bio_tags
from extras.legacy.extraction.nn_utils import (
    AdamWOptimizer,
    masked_cross_entropy_and_grad,
    numerical_gradient_check,
    pad_sequence_batch,
    sigmoid,
    softmax,
)
from medresearch_platform.extraction.text import TAG_TO_ID, TAGS, Token, Vocabulary, spans_to_bio, tokenize


class BiLSTMEncoder:
    """Shared pure-NumPy randomly initialized Embedding + Bidirectional LSTM + Linear projection."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 32,
        num_tags: int = len(TAGS),
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {
            "E": initialize_tensor(rng, (vocab_size, embed_dim), method="xavier_uniform").astype(np.float64),
            "W_fwd": initialize_tensor(rng, (embed_dim + hidden_dim, 4 * hidden_dim), method="xavier_uniform").astype(
                np.float64
            ),
            "b_fwd": np.zeros((4 * hidden_dim,), dtype=np.float64),
            "W_bwd": initialize_tensor(rng, (embed_dim + hidden_dim, 4 * hidden_dim), method="xavier_uniform").astype(
                np.float64
            ),
            "b_bwd": np.zeros((4 * hidden_dim,), dtype=np.float64),
            "W_out": initialize_tensor(rng, (2 * hidden_dim, num_tags), method="xavier_uniform").astype(np.float64),
            "b_out": np.zeros((num_tags,), dtype=np.float64),
        }
        # Initialize forget-gate biases to 1.0 (standard random-init LSTM stability)
        self.params["b_fwd"][hidden_dim : 2 * hidden_dim] = 1.0
        self.params["b_bwd"][hidden_dim : 2 * hidden_dim] = 1.0

    @property
    def parameter_count(self) -> int:
        return int(sum(arr.size for arr in self.params.values()))

    def manifest(self, architecture_name: str, architecture_version: str = "1.0.0") -> RandomInitManifest:
        return build_random_init_manifest(
            architecture_name=architecture_name,
            architecture_version=architecture_version,
            init_method="xavier_uniform",
            seed=self.seed,
            parameters=self.params,
        )

    def _run_direction(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
        reverse: bool,
    ) -> tuple[np.ndarray, list[tuple[Any, ...]]]:
        bsz, seq_len, _ = x.shape
        h_dim = self.hidden_dim
        h_all = np.zeros((bsz, seq_len, h_dim), dtype=np.float64)
        h_prev = np.zeros((bsz, h_dim), dtype=np.float64)
        c_prev = np.zeros((bsz, h_dim), dtype=np.float64)
        cache: list[tuple[Any, ...]] = []

        steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        for t in steps:
            m_t = mask[:, t : t + 1]
            z_t = np.concatenate([x[:, t, :], h_prev], axis=1)
            gates = z_t @ W + b
            i_t = sigmoid(gates[:, 0:h_dim])
            f_t = sigmoid(gates[:, h_dim : 2 * h_dim])
            o_t = sigmoid(gates[:, 2 * h_dim : 3 * h_dim])
            g_t = np.tanh(gates[:, 3 * h_dim : 4 * h_dim])

            c_raw = f_t * c_prev + i_t * g_t
            tanh_c = np.tanh(c_raw)
            h_raw = o_t * tanh_c

            c_t = m_t * c_raw
            h_t = m_t * h_raw
            h_all[:, t, :] = h_t

            cache.append((t, m_t, z_t, c_prev, i_t, f_t, o_t, g_t, tanh_c))
            h_prev, c_prev = h_t, c_t

        return h_all, cache

    def _backprop_direction(
        self,
        dh_all: np.ndarray,
        W: np.ndarray,
        cache: list[tuple[Any, ...]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bsz, seq_len, h_dim = dh_all.shape
        x_dim = W.shape[0] - h_dim
        dx = np.zeros((bsz, seq_len, x_dim), dtype=np.float64)
        dW = np.zeros_like(W)
        db = np.zeros((4 * h_dim,), dtype=np.float64)

        dh_next = np.zeros((bsz, h_dim), dtype=np.float64)
        dc_next = np.zeros((bsz, h_dim), dtype=np.float64)

        for t, m_t, z_t, c_prev, i_t, f_t, o_t, g_t, tanh_c in reversed(cache):
            dh = dh_all[:, t, :] + dh_next
            dh_raw = m_t * dh
            do_t = dh_raw * tanh_c
            dc_raw = m_t * dc_next + dh_raw * o_t * (1.0 - tanh_c * tanh_c)

            df_t = dc_raw * c_prev
            di_t = dc_raw * g_t
            dg_t = dc_raw * i_t
            dc_prev = dc_raw * f_t

            d_gates = np.concatenate(
                [
                    di_t * i_t * (1.0 - i_t),
                    df_t * f_t * (1.0 - f_t),
                    do_t * o_t * (1.0 - o_t),
                    dg_t * (1.0 - g_t * g_t),
                ],
                axis=1,
            )
            dW += z_t.T @ d_gates
            db += np.sum(d_gates, axis=0)
            dz = d_gates @ W.T
            dx[:, t, :] = dz[:, :x_dim]
            dh_next = dz[:, x_dim:]
            dc_next = dc_prev

        return dx, dW, db

    def forward(self, token_ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        x = self.params["E"][token_ids] * mask[:, :, None]
        h_fwd, cache_fwd = self._run_direction(x, mask, self.params["W_fwd"], self.params["b_fwd"], reverse=False)
        h_bwd, cache_bwd = self._run_direction(x, mask, self.params["W_bwd"], self.params["b_bwd"], reverse=True)
        h_cat = np.concatenate([h_fwd, h_bwd], axis=-1)
        logits = h_cat @ self.params["W_out"] + self.params["b_out"]
        ctx = {
            "token_ids": token_ids,
            "mask": mask,
            "h_cat": h_cat,
            "cache_fwd": cache_fwd,
            "cache_bwd": cache_bwd,
        }
        return logits, ctx

    def backward(self, d_logits: np.ndarray, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
        token_ids: np.ndarray = ctx["token_ids"]
        mask: np.ndarray = ctx["mask"]
        h_cat: np.ndarray = ctx["h_cat"]

        bsz, seq_len, two_h = h_cat.shape
        h_dim = two_h // 2

        dW_out = h_cat.reshape(-1, two_h).T @ d_logits.reshape(-1, self.num_tags)
        db_out = np.sum(d_logits, axis=(0, 1))
        dh_cat = d_logits @ self.params["W_out"].T

        dx_fwd, dW_fwd, db_fwd = self._backprop_direction(
            dh_cat[:, :, :h_dim], self.params["W_fwd"], ctx["cache_fwd"]
        )
        dx_bwd, dW_bwd, db_bwd = self._backprop_direction(
            dh_cat[:, :, h_dim:], self.params["W_bwd"], ctx["cache_bwd"]
        )
        dx = (dx_fwd + dx_bwd) * mask[:, :, None]

        dE = np.zeros_like(self.params["E"])
        np.add.at(dE, token_ids, dx)

        return {
            "E": dE,
            "W_fwd": dW_fwd,
            "b_fwd": db_fwd,
            "W_bwd": dW_bwd,
            "b_bwd": db_bwd,
            "W_out": dW_out,
            "b_out": db_out,
        }


@dataclass
class BiLSTMExtractor:
    """Approach 3: Pure-NumPy randomly initialized BiLSTM medical sequence tagger."""

    embed_dim: int = 32
    hidden_dim: int = 32
    epochs: int = 25
    batch_size: int = 8
    lr: float = 0.015
    seed: int = 42
    model_name: str = "RandomInitBiLSTMExtractor"
    model_version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.vocab: Vocabulary | None = None
        self.encoder: BiLSTMEncoder | None = None
        self.terminology = BilingualTerminologyRegistry()
        self.loss_history: list[float] = []

    def _encode_corpus(
        self, corpus: list[tuple[str, list[tuple[int, int, str]]]]
    ) -> list[tuple[list[int], list[int]]]:
        assert self.vocab is not None
        encoded: list[tuple[list[int], list[int]]] = []
        for text, spans in corpus:
            toks = tokenize(text)
            if not toks:
                continue
            x_ids = self.vocab.encode([t.text for t in toks])
            y_ids = [TAG_TO_ID[tag] for tag in spans_to_bio(toks, spans)]
            encoded.append((x_ids, y_ids))
        return encoded

    def fit(self, corpus: list[tuple[str, list[tuple[int, int, str]]]]) -> dict[str, Any]:
        token_lists = [[t.text for t in tokenize(text)] for text, _ in corpus]
        self.vocab = Vocabulary.build(token_lists)
        self.encoder = BiLSTMEncoder(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_tags=len(TAGS),
            seed=self.seed,
        )
        encoded = self._encode_corpus(corpus)
        optimizer = AdamWOptimizer(self.encoder.params, lr=self.lr)
        rng = np.random.default_rng(self.seed)
        self.loss_history = []

        for _epoch in range(self.epochs):
            order = rng.permutation(len(encoded))
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(order), self.batch_size):
                idxs = order[start : start + self.batch_size]
                x_batch = [encoded[i][0] for i in idxs]
                y_batch = [encoded[i][1] for i in idxs]
                tok_arr, mask_arr, tag_arr = pad_sequence_batch(x_batch, y_batch)
                assert tag_arr is not None
                logits, ctx = self.encoder.forward(tok_arr, mask_arr)
                loss, d_logits = masked_cross_entropy_and_grad(logits, tag_arr, mask_arr)
                grads = self.encoder.backward(d_logits, ctx)
                optimizer.step(self.encoder.params, grads)
                epoch_loss += loss
                batches += 1
            self.loss_history.append(epoch_loss / max(1, batches))

        manifest = self.encoder.manifest(self.model_name, self.model_version)
        return {
            "initial_loss": round(self.loss_history[0], 4) if self.loss_history else None,
            "final_loss": round(self.loss_history[-1], 4) if self.loss_history else None,
            "parameter_count": self.encoder.parameter_count,
            "weights_sha256": manifest.initial_weights_sha256,
            "pretrained_weights_used": manifest.pretrained_weights_used,
        }

    def check_gradients(self) -> dict[str, float]:
        """Run central finite-difference gradient check on a tiny synthetic batch."""
        enc = self.encoder or BiLSTMEncoder(vocab_size=8, embed_dim=6, hidden_dim=5, num_tags=5, seed=self.seed)
        tok_arr, mask_arr, tag_arr = pad_sequence_batch(
            [[1, 2, 3, 4], [2, 5, 1]],
            [[0, 1, 2, 0], [3, 4, 0]],
        )
        assert tag_arr is not None

        def loss_and_grads() -> tuple[float, dict[str, np.ndarray]]:
            logits, ctx = enc.forward(tok_arr, mask_arr)
            loss, d_logits = masked_cross_entropy_and_grad(logits, tag_arr, mask_arr)
            grads = enc.backward(d_logits, ctx)
            return loss, grads

        return numerical_gradient_check(enc.params, loss_and_grads)

    def predict_bio(self, text: str) -> tuple[list[Token], list[str], list[float]]:
        if self.encoder is None or self.vocab is None:
            raise RuntimeError("BiLSTMExtractor must be fitted before prediction.")
        toks = tokenize(text)
        if not toks:
            return [], [], []
        x_ids = self.vocab.encode([t.text for t in toks])
        tok_arr, mask_arr, _ = pad_sequence_batch([x_ids])
        logits, _ = self.encoder.forward(tok_arr, mask_arr)
        probs = softmax(logits[0], axis=-1)
        best_idx = np.argmax(probs, axis=-1)
        pred_tags = [TAGS[int(i)] for i in best_idx]
        confs = [float(probs[t, best_idx[t]]) for t in range(len(toks))]
        return toks, pred_tags, confs

    def extract(
        self,
        text: str,
        *,
        source_record_id: str = "record-0",
        document_id: str = "doc-0",
        language: str = "en",
        page_number: int = 1,
    ) -> ExtractionResult:
        assert self.encoder is not None
        manifest = self.encoder.manifest(self.model_name, self.model_version)
        toks, tags, confs = self.predict_bio(text)
        return records_from_bio_tags(
            text=text,
            tokens=toks,
            tags=tags,
            token_confidences=confs,
            source_record_id=source_record_id,
            document_id=document_id,
            language=language,
            model_name=self.model_name,
            model_version=self.model_version,
            provenance={
                "approach": "bilstm_random_init",
                "parameter_count": self.encoder.parameter_count,
                "weights_sha256": manifest.initial_weights_sha256,
                "pretrained_weights_used": False,
            },
            terminology=self.terminology,
            page_number=page_number,
        )
