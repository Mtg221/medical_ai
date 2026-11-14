"""Approach 5: Pure-NumPy Randomly Initialized Transformer Encoder Sequence Tagger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from medresearch_platform.ai.random_init import build_random_init_manifest, initialize_tensor
from medresearch_platform.ai.terminology import BilingualTerminologyRegistry
from medresearch_platform.extraction.contract import ExtractionResult, records_from_bio_tags
from medresearch_platform.extraction.nn_utils import (
    AdamWOptimizer,
    masked_cross_entropy_and_grad,
    numerical_gradient_check,
    pad_sequence_batch,
    softmax,
)
from medresearch_platform.extraction.text import TAG_TO_ID, TAGS, Token, Vocabulary, spans_to_bio, tokenize


def _layer_norm_forward(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    out = gamma * x_hat + beta
    return out, (x_hat, inv_std, gamma)


def _layer_norm_backward(
    d_out: np.ndarray, cache: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_hat, inv_std, gamma = cache
    d_dim = x_hat.shape[-1]
    d_gamma = np.sum(d_out * x_hat, axis=(0, 1))
    d_beta = np.sum(d_out, axis=(0, 1))
    dx_hat = d_out * gamma
    dx = (inv_std / float(d_dim)) * (
        float(d_dim) * dx_hat
        - np.sum(dx_hat, axis=-1, keepdims=True)
        - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
    )
    return dx, d_gamma, d_beta


class TransformerEncoderModel:
    """Small CPU-compatible randomly initialized Transformer Encoder for BIO sequence tagging."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 128,
        d_model: int = 32,
        num_heads: int = 4,
        d_ff: int = 64,
        num_layers: int = 1,
        num_tags: int = len(TAGS),
        seed: int = 42,
    ) -> None:
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_tags = num_tags
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {
            "E_tok": initialize_tensor(rng, (vocab_size, d_model), method="xavier_uniform").astype(np.float64),
            "E_pos": initialize_tensor(rng, (max_len, d_model), method="xavier_uniform").astype(np.float64),
            "W_out": initialize_tensor(rng, (d_model, num_tags), method="xavier_uniform").astype(np.float64),
            "b_out": np.zeros((num_tags,), dtype=np.float64),
        }
        for layer in range(num_layers):
            p = f"L{layer}_"
            for proj in ("W_q", "W_k", "W_v", "W_o"):
                self.params[p + proj] = initialize_tensor(rng, (d_model, d_model), method="xavier_uniform").astype(
                    np.float64
                )
            for bias in ("b_q", "b_k", "b_v", "b_o"):
                self.params[p + bias] = np.zeros((d_model,), dtype=np.float64)
            self.params[p + "ln1_g"] = np.ones((d_model,), dtype=np.float64)
            self.params[p + "ln1_b"] = np.zeros((d_model,), dtype=np.float64)
            self.params[p + "W_ff1"] = initialize_tensor(rng, (d_model, d_ff), method="he_normal").astype(np.float64)
            self.params[p + "b_ff1"] = np.zeros((d_ff,), dtype=np.float64)
            self.params[p + "W_ff2"] = initialize_tensor(rng, (d_ff, d_model), method="xavier_uniform").astype(
                np.float64
            )
            self.params[p + "b_ff2"] = np.zeros((d_model,), dtype=np.float64)
            self.params[p + "ln2_g"] = np.ones((d_model,), dtype=np.float64)
            self.params[p + "ln2_b"] = np.zeros((d_model,), dtype=np.float64)

    @property
    def parameter_count(self) -> int:
        return int(sum(arr.size for arr in self.params.values()))

    def forward(self, token_ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        bsz, seq_len = token_ids.shape
        pos_ids = np.minimum(np.arange(seq_len, dtype=np.int64), self.max_len - 1)
        x = (self.params["E_tok"][token_ids] + self.params["E_pos"][pos_ids][None, :, :]) * mask[:, :, None]

        # Additive attention mask: 0 for valid keys, -1e9 for padded keys
        key_mask = (1.0 - mask)[:, None, None, :] * -1e9
        layer_caches: list[dict[str, Any]] = []
        scale = 1.0 / np.sqrt(float(self.head_dim))

        for layer in range(self.num_layers):
            p = f"L{layer}_"
            x_in = x
            q = (x_in @ self.params[p + "W_q"] + self.params[p + "b_q"]).reshape(
                bsz, seq_len, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
            k = (x_in @ self.params[p + "W_k"] + self.params[p + "b_k"]).reshape(
                bsz, seq_len, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
            v = (x_in @ self.params[p + "W_v"] + self.params[p + "b_v"]).reshape(
                bsz, seq_len, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

            scores = (q @ k.transpose(0, 1, 3, 2)) * scale + key_mask
            attn = softmax(scores, axis=-1)
            context = (attn @ v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.d_model)
            attn_out = context @ self.params[p + "W_o"] + self.params[p + "b_o"]

            res1 = x_in + attn_out
            ln1_out, ln1_cache = _layer_norm_forward(res1, self.params[p + "ln1_g"], self.params[p + "ln1_b"])

            ff_pre = ln1_out @ self.params[p + "W_ff1"] + self.params[p + "b_ff1"]
            ff_act = np.maximum(0.0, ff_pre)
            ff_out = ff_act @ self.params[p + "W_ff2"] + self.params[p + "b_ff2"]

            res2 = ln1_out + ff_out
            ln2_out, ln2_cache = _layer_norm_forward(res2, self.params[p + "ln2_g"], self.params[p + "ln2_b"])
            x = ln2_out * mask[:, :, None]

            layer_caches.append(
                {
                    "x_in": x_in,
                    "q": q,
                    "k": k,
                    "v": v,
                    "attn": attn,
                    "context": context,
                    "ln1_cache": ln1_cache,
                    "ln1_out": ln1_out,
                    "ff_pre": ff_pre,
                    "ff_act": ff_act,
                    "ln2_cache": ln2_cache,
                }
            )

        logits = x @ self.params["W_out"] + self.params["b_out"]
        ctx = {
            "token_ids": token_ids,
            "pos_ids": pos_ids,
            "mask": mask,
            "x_final": x,
            "layer_caches": layer_caches,
        }
        return logits, ctx

    def backward(self, d_logits: np.ndarray, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
        token_ids: np.ndarray = ctx["token_ids"]
        pos_ids: np.ndarray = ctx["pos_ids"]
        mask: np.ndarray = ctx["mask"]
        x_final: np.ndarray = ctx["x_final"]
        bsz, seq_len = token_ids.shape
        scale = 1.0 / np.sqrt(float(self.head_dim))

        grads: dict[str, np.ndarray] = {}
        grads["W_out"] = x_final.reshape(-1, self.d_model).T @ d_logits.reshape(-1, self.num_tags)
        grads["b_out"] = np.sum(d_logits, axis=(0, 1))
        dx = d_logits @ self.params["W_out"].T

        for layer in range(self.num_layers - 1, -1, -1):
            p = f"L{layer}_"
            lc = ctx["layer_caches"][layer]
            dx_masked = dx * mask[:, :, None]

            d_res2, grads[p + "ln2_g"], grads[p + "ln2_b"] = _layer_norm_backward(dx_masked, lc["ln2_cache"])
            d_ln1_out = d_res2.copy()
            d_ff_out = d_res2

            grads[p + "W_ff2"] = lc["ff_act"].reshape(-1, self.d_ff).T @ d_ff_out.reshape(-1, self.d_model)
            grads[p + "b_ff2"] = np.sum(d_ff_out, axis=(0, 1))
            d_ff_act = d_ff_out @ self.params[p + "W_ff2"].T
            d_ff_pre = d_ff_act * (lc["ff_pre"] > 0.0)

            grads[p + "W_ff1"] = lc["ln1_out"].reshape(-1, self.d_model).T @ d_ff_pre.reshape(-1, self.d_ff)
            grads[p + "b_ff1"] = np.sum(d_ff_pre, axis=(0, 1))
            d_ln1_out = d_ln1_out + (d_ff_pre @ self.params[p + "W_ff1"].T)

            d_res1, grads[p + "ln1_g"], grads[p + "ln1_b"] = _layer_norm_backward(d_ln1_out, lc["ln1_cache"])
            d_x_in = d_res1.copy()
            d_attn_out = d_res1

            grads[p + "W_o"] = lc["context"].reshape(-1, self.d_model).T @ d_attn_out.reshape(-1, self.d_model)
            grads[p + "b_o"] = np.sum(d_attn_out, axis=(0, 1))
            d_context = (d_attn_out @ self.params[p + "W_o"].T).reshape(
                bsz, seq_len, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

            attn = lc["attn"]
            d_attn = d_context @ lc["v"].transpose(0, 1, 3, 2)
            d_v = attn.transpose(0, 1, 3, 2) @ d_context

            d_scores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True))
            d_q = (d_scores @ lc["k"]) * scale
            d_k = (d_scores.transpose(0, 1, 3, 2) @ lc["q"]) * scale

            dq_flat = d_q.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.d_model)
            dk_flat = d_k.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.d_model)
            dv_flat = d_v.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.d_model)

            x_in_flat = lc["x_in"].reshape(-1, self.d_model)
            for suffix, d_proj in (("q", dq_flat), ("k", dk_flat), ("v", dv_flat)):
                grads[p + f"W_{suffix}"] = x_in_flat.T @ d_proj.reshape(-1, self.d_model)
                grads[p + f"b_{suffix}"] = np.sum(d_proj, axis=(0, 1))
                d_x_in = d_x_in + d_proj @ self.params[p + f"W_{suffix}"].T

            dx = d_x_in

        dx_embed = dx * mask[:, :, None]
        dE_tok = np.zeros_like(self.params["E_tok"])
        np.add.at(dE_tok, token_ids, dx_embed)
        dE_pos = np.zeros_like(self.params["E_pos"])
        np.add.at(dE_pos, pos_ids, np.sum(dx_embed, axis=0))
        grads["E_tok"] = dE_tok
        grads["E_pos"] = dE_pos
        return grads


@dataclass
class TransformerExtractor:
    """Approach 5: Pure-NumPy randomly initialized Transformer Encoder sequence tagger."""

    d_model: int = 32
    num_heads: int = 4
    d_ff: int = 64
    num_layers: int = 1
    max_len: int = 128
    epochs: int = 25
    batch_size: int = 8
    lr: float = 0.01
    seed: int = 42
    model_name: str = "RandomInitTransformerExtractor"
    model_version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.vocab: Vocabulary | None = None
        self.model: TransformerEncoderModel | None = None
        self.terminology = BilingualTerminologyRegistry()
        self.loss_history: list[float] = []

    def fit(self, corpus: list[tuple[str, list[tuple[int, int, str]]]]) -> dict[str, Any]:
        token_lists = [[t.text for t in tokenize(text)] for text, _ in corpus]
        self.vocab = Vocabulary.build(token_lists)
        self.model = TransformerEncoderModel(
            vocab_size=len(self.vocab),
            max_len=self.max_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            num_tags=len(TAGS),
            seed=self.seed,
        )
        encoded = []
        for text, spans in corpus:
            toks = tokenize(text)
            if not toks:
                continue
            encoded.append(
                (self.vocab.encode([t.text for t in toks]), [TAG_TO_ID[t] for t in spans_to_bio(toks, spans)])
            )

        optimizer = AdamWOptimizer(self.model.params, lr=self.lr)
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
                logits, ctx = self.model.forward(tok_arr, mask_arr)
                loss, d_logits = masked_cross_entropy_and_grad(logits, tag_arr, mask_arr)
                grads = self.model.backward(d_logits, ctx)
                optimizer.step(self.model.params, grads)
                epoch_loss += loss
                batches += 1
            self.loss_history.append(epoch_loss / max(1, batches))

        manifest = build_random_init_manifest(
            architecture_name=self.model_name,
            architecture_version=self.model_version,
            init_method="xavier_uniform",
            seed=self.seed,
            parameters=self.model.params,
        )
        return {
            "initial_loss": round(self.loss_history[0], 4) if self.loss_history else None,
            "final_loss": round(self.loss_history[-1], 4) if self.loss_history else None,
            "parameter_count": self.model.parameter_count,
            "weights_sha256": manifest.initial_weights_sha256,
            "pretrained_weights_used": manifest.pretrained_weights_used,
        }

    def check_gradients(self) -> dict[str, float]:
        """Run central finite-difference gradient check on a tiny synthetic batch."""
        model = TransformerEncoderModel(
            vocab_size=8,
            max_len=16,
            d_model=8,
            num_heads=2,
            d_ff=16,
            num_layers=1,
            num_tags=5,
            seed=self.seed,
        )
        tok_arr, mask_arr, tag_arr = pad_sequence_batch(
            [[1, 2, 3, 4], [2, 5, 1]],
            [[0, 1, 2, 0], [3, 4, 0]],
        )
        assert tag_arr is not None

        def loss_and_grads() -> tuple[float, dict[str, np.ndarray]]:
            logits, ctx = model.forward(tok_arr, mask_arr)
            loss, d_logits = masked_cross_entropy_and_grad(logits, tag_arr, mask_arr)
            grads = model.backward(d_logits, ctx)
            return loss, grads

        return numerical_gradient_check(model.params, loss_and_grads)

    def predict_bio(self, text: str) -> tuple[list[Token], list[str], list[float]]:
        if self.model is None or self.vocab is None:
            raise RuntimeError("TransformerExtractor must be fitted before prediction.")
        toks = tokenize(text)
        if not toks:
            return [], [], []
        x_ids = self.vocab.encode([t.text for t in toks])
        tok_arr, mask_arr, _ = pad_sequence_batch([x_ids])
        logits, _ = self.model.forward(tok_arr, mask_arr)
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
        assert self.model is not None
        manifest = build_random_init_manifest(
            architecture_name=self.model_name,
            architecture_version=self.model_version,
            init_method="xavier_uniform",
            seed=self.seed,
            parameters=self.model.params,
        )
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
                "approach": "transformer_encoder_random_init",
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "parameter_count": self.model.parameter_count,
                "weights_sha256": manifest.initial_weights_sha256,
                "pretrained_weights_used": False,
            },
            terminology=self.terminology,
            page_number=page_number,
        )
