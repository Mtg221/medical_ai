"""Approach 4: Pure-NumPy Randomly Initialized BiLSTM-CRF Sequence Tagger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from medresearch_platform.ai.random_init import build_random_init_manifest, initialize_tensor
from medresearch_platform.ai.terminology import BilingualTerminologyRegistry
from medresearch_platform.extraction.bilstm import BiLSTMEncoder
from medresearch_platform.extraction.contract import ExtractionResult, records_from_bio_tags
from medresearch_platform.extraction.nn_utils import (
    AdamWOptimizer,
    numerical_gradient_check,
    pad_sequence_batch,
    softmax,
)
from medresearch_platform.extraction.text import TAGS, TAG_TO_ID, Token, Vocabulary, spans_to_bio, tokenize


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    res = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(res, axis=axis)


def build_bio_transition_mask(tags: tuple[str, ...] = TAGS) -> tuple[np.ndarray, np.ndarray]:
    """Return (trans_mask [C, C], start_mask [C]) where 1.0 = BIO-valid, 0.0 = illegal."""
    num_tags = len(tags)
    trans_mask = np.ones((num_tags, num_tags), dtype=np.float64)
    start_mask = np.ones((num_tags,), dtype=np.float64)

    for j, to_tag in enumerate(tags):
        if to_tag.startswith("I-"):
            start_mask[j] = 0.0
            to_type = to_tag[2:]
            for i, from_tag in enumerate(tags):
                if from_tag == "O":
                    trans_mask[i, j] = 0.0
                elif from_tag[2:] != to_type:
                    trans_mask[i, j] = 0.0
    return trans_mask, start_mask


class LinearChainCRF:
    """Linear-chain CRF with BIO constraints, exact forward-backward NLL, and Viterbi decoding."""

    def __init__(self, tags: tuple[str, ...] = TAGS, seed: int = 42) -> None:
        self.tags = tags
        self.num_tags = len(tags)
        rng = np.random.default_rng(seed + 101)
        self.params: dict[str, np.ndarray] = {
            "crf_trans": initialize_tensor(rng, (self.num_tags, self.num_tags), method="xavier_uniform").astype(
                np.float64
            ),
            "crf_start": np.zeros((self.num_tags,), dtype=np.float64),
            "crf_end": np.zeros((self.num_tags,), dtype=np.float64),
        }
        self.trans_mask, self.start_mask = build_bio_transition_mask(tags)

    def _effective_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eff_trans = self.params["crf_trans"] * self.trans_mask - 1e4 * (1.0 - self.trans_mask)
        eff_start = self.params["crf_start"] * self.start_mask - 1e4 * (1.0 - self.start_mask)
        eff_end = self.params["crf_end"]
        return eff_trans, eff_start, eff_end

    def nll_and_grads(
        self,
        emissions: np.ndarray,
        tags: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
        """Compute mean sequence CRF NLL loss and exact gradients wrt emissions & CRF params."""
        bsz, _, num_tags = emissions.shape
        eff_trans, eff_start, eff_end = self._effective_params()

        d_emissions = np.zeros_like(emissions)
        d_eff_trans = np.zeros_like(eff_trans)
        d_eff_start = np.zeros_like(eff_start)
        d_eff_end = np.zeros_like(eff_end)
        total_loss = 0.0

        for b in range(bsz):
            seq_len = int(np.sum(mask[b] > 0.0))
            if seq_len == 0:
                continue
            emit = emissions[b, :seq_len]
            y = tags[b, :seq_len]

            # Forward pass
            alpha = np.zeros((seq_len, num_tags), dtype=np.float64)
            alpha[0] = eff_start + emit[0]
            for t in range(1, seq_len):
                # alpha[t, j] = logsumexp_i(alpha[t-1, i] + eff_trans[i, j]) + emit[t, j]
                alpha[t] = _logsumexp(alpha[t - 1, :, None] + eff_trans, axis=0) + emit[t]
            log_z = float(_logsumexp(alpha[seq_len - 1] + eff_end, axis=0))

            # Backward pass
            beta = np.zeros((seq_len, num_tags), dtype=np.float64)
            beta[seq_len - 1] = eff_end
            for t in range(seq_len - 2, -1, -1):
                # beta[t, i] = logsumexp_j(eff_trans[i, j] + emit[t+1, j] + beta[t+1, j])
                beta[t] = _logsumexp(eff_trans + emit[t + 1, None, :] + beta[t + 1, None, :], axis=1)

            # Gold sequence score
            gold_score = float(eff_start[y[0]] + np.sum(emit[np.arange(seq_len), y]) + eff_end[y[seq_len - 1]])
            if seq_len > 1:
                gold_score += float(np.sum(eff_trans[y[:-1], y[1:]]))

            total_loss += (log_z - gold_score)

            # Marginals
            node_marg = np.exp(alpha + beta - log_z)
            d_emit_b = node_marg.copy()
            d_emit_b[np.arange(seq_len), y] -= 1.0
            d_emissions[b, :seq_len] += d_emit_b

            d_eff_start += node_marg[0]
            d_eff_start[y[0]] -= 1.0

            d_eff_end += node_marg[seq_len - 1]
            d_eff_end[y[seq_len - 1]] -= 1.0

            for t in range(seq_len - 1):
                pair_log = (
                    alpha[t, :, None]
                    + eff_trans
                    + emit[t + 1, None, :]
                    + beta[t + 1, None, :]
                    - log_z
                )
                d_eff_trans += np.exp(pair_log)
                d_eff_trans[y[t], y[t + 1]] -= 1.0

        inv_b = 1.0 / float(max(1, bsz))
        loss = total_loss * inv_b
        d_emissions *= inv_b
        crf_grads = {
            "crf_trans": (d_eff_trans * self.trans_mask) * inv_b,
            "crf_start": (d_eff_start * self.start_mask) * inv_b,
            "crf_end": d_eff_end * inv_b,
        }
        return loss, d_emissions, crf_grads

    def viterbi_decode(self, emissions: np.ndarray, mask: np.ndarray) -> list[list[int]]:
        """Decode highest-scoring BIO-valid tag sequence for each batch item."""
        bsz, _, num_tags = emissions.shape
        eff_trans, eff_start, eff_end = self._effective_params()
        paths: list[list[int]] = []

        for b in range(bsz):
            seq_len = int(np.sum(mask[b] > 0.0))
            if seq_len == 0:
                paths.append([])
                continue
            emit = emissions[b, :seq_len]
            score = eff_start + emit[0]
            backpointers: list[np.ndarray] = []

            for t in range(1, seq_len):
                cand = score[:, None] + eff_trans
                best_prev = np.argmax(cand, axis=0)
                score = cand[best_prev, np.arange(num_tags)] + emit[t]
                backpointers.append(best_prev)

            score = score + eff_end
            best_last = int(np.argmax(score))
            seq = [best_last]
            for bp in reversed(backpointers):
                best_last = int(bp[best_last])
                seq.append(best_last)
            seq.reverse()
            paths.append(seq)

        return paths


@dataclass
class BiLSTMCRFExtractor:
    """Approach 4: Pure-NumPy randomly initialized BiLSTM-CRF medical sequence tagger."""

    embed_dim: int = 32
    hidden_dim: int = 32
    epochs: int = 25
    batch_size: int = 8
    lr: float = 0.015
    seed: int = 42
    model_name: str = "RandomInitBiLSTMCRFExtractor"
    model_version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.vocab: Vocabulary | None = None
        self.encoder: BiLSTMEncoder | None = None
        self.crf: LinearChainCRF | None = None
        self.terminology = BilingualTerminologyRegistry()
        self.loss_history: list[float] = []

    def _all_params(self) -> dict[str, np.ndarray]:
        assert self.encoder is not None and self.crf is not None
        return {**self.encoder.params, **self.crf.params}

    @property
    def parameter_count(self) -> int:
        if self.encoder is None or self.crf is None:
            return 0
        return int(sum(arr.size for arr in self._all_params().values()))

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
        self.crf = LinearChainCRF(tags=TAGS, seed=self.seed)

        encoded: list[tuple[list[int], list[int]]] = []
        for text, spans in corpus:
            toks = tokenize(text)
            if not toks:
                continue
            encoded.append(
                (self.vocab.encode([t.text for t in toks]), [TAG_TO_ID[t] for t in spans_to_bio(toks, spans)])
            )

        all_params = self._all_params()
        optimizer = AdamWOptimizer(all_params, lr=self.lr)
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
                emissions, ctx = self.encoder.forward(tok_arr, mask_arr)
                loss, d_emit, crf_grads = self.crf.nll_and_grads(emissions, tag_arr, mask_arr)
                enc_grads = self.encoder.backward(d_emit, ctx)
                optimizer.step(all_params, {**enc_grads, **crf_grads})
                epoch_loss += loss
                batches += 1
            self.loss_history.append(epoch_loss / max(1, batches))

        manifest = build_random_init_manifest(
            architecture_name=self.model_name,
            architecture_version=self.model_version,
            init_method="xavier_uniform",
            seed=self.seed,
            parameters=all_params,
        )
        return {
            "initial_loss": round(self.loss_history[0], 4) if self.loss_history else None,
            "final_loss": round(self.loss_history[-1], 4) if self.loss_history else None,
            "parameter_count": self.parameter_count,
            "weights_sha256": manifest.initial_weights_sha256,
            "pretrained_weights_used": manifest.pretrained_weights_used,
        }

    def check_gradients(self) -> dict[str, float]:
        """Run central finite-difference gradient check on a tiny synthetic batch."""
        tiny_tags = ("O", "B-SYMPTOM", "I-SYMPTOM", "B-DIAGNOSIS", "I-DIAGNOSIS")
        enc = BiLSTMEncoder(vocab_size=8, embed_dim=6, hidden_dim=5, num_tags=len(tiny_tags), seed=self.seed)
        crf = LinearChainCRF(tags=tiny_tags, seed=self.seed)
        all_params = {**enc.params, **crf.params}
        tok_arr, mask_arr, tag_arr = pad_sequence_batch(
            [[1, 2, 3, 4], [2, 5, 1]],
            [[0, 1, 2, 0], [3, 4, 0]],
        )
        assert tag_arr is not None

        def loss_and_grads() -> tuple[float, dict[str, np.ndarray]]:
            emissions, ctx = enc.forward(tok_arr, mask_arr)
            loss, d_emit, crf_grads = crf.nll_and_grads(emissions, tag_arr, mask_arr)
            enc_grads = enc.backward(d_emit, ctx)
            return loss, {**enc_grads, **crf_grads}

        return numerical_gradient_check(all_params, loss_and_grads)

    def predict_bio(self, text: str) -> tuple[list[Token], list[str], list[float]]:
        if self.encoder is None or self.crf is None or self.vocab is None:
            raise RuntimeError("BiLSTMCRFExtractor must be fitted before prediction.")
        toks = tokenize(text)
        if not toks:
            return [], [], []
        x_ids = self.vocab.encode([t.text for t in toks])
        tok_arr, mask_arr, _ = pad_sequence_batch([x_ids])
        emissions, _ = self.encoder.forward(tok_arr, mask_arr)
        best_path = self.crf.viterbi_decode(emissions, mask_arr)[0]
        probs = softmax(emissions[0], axis=-1)
        pred_tags = [TAGS[idx] for idx in best_path]
        confs = [float(probs[t, best_path[t]]) for t in range(len(toks))]
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
        assert self.encoder is not None and self.crf is not None
        manifest = build_random_init_manifest(
            architecture_name=self.model_name,
            architecture_version=self.model_version,
            init_method="xavier_uniform",
            seed=self.seed,
            parameters=self._all_params(),
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
                "approach": "bilstm_crf_random_init",
                "parameter_count": self.parameter_count,
                "weights_sha256": manifest.initial_weights_sha256,
                "pretrained_weights_used": False,
                "bio_transition_constraints": True,
            },
            terminology=self.terminology,
            page_number=page_number,
        )
