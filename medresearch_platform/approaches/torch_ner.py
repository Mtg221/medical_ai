"""From-scratch PyTorch Char-CNN + BiLSTM + Partial-Label Marginal CRF for multilingual clinical NER.

Strictly trained from random initialization (pretrained_weights_used = False, zero external weights).
Supports exact Marginal Likelihood CRF training (Greenberg et al., EMNLP 2018) so heterogeneous
corpora with partial entity schemas (e.g., PxCorpus 5 types, E3C 6 types, MACCROBAT 11 types,
BC5CDR 2 types, SYNTH-MED-V3.1 14 types) can be trained jointly without false-negative 'O' corruption.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from medresearch_platform.approaches.classical_ml import _doc_classifier_fit_predict
from medresearch_platform.approaches.context_negex import apply_context_and_relations
from medresearch_platform.approaches.neural_seq import CHAR_FEAT_DIM, _ortho_features
from medresearch_platform.approaches.ocr_augment import build_ocr_augmented_train_set
from medresearch_platform.approaches.self_supervised import (
    SelfSupervisedMedicalEmbeddings,
    load_optional_self_supervised,
)
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
from medresearch_platform.data.schema import ENTITY_TYPES, AnnotatedDocument
from medresearch_platform.eval_harness.metrics import DocumentPrediction, PredictedEntity

NUM_TAGS = len(BIO_TAGS)
MAX_WORD_CHARS = 24
MAX_CHUNK_TOKENS = 160

ALL_ENTITY_TYPES: frozenset[str] = frozenset(ENTITY_TYPES)

CORPUS_ANNOTATED_TYPES: dict[str, frozenset[str]] = {
    "maccrobat2020": frozenset(
        {
            "DIAGNOSIS",
            "SYMPTOM",
            "MEDICATION",
            "DOSAGE",
            "FREQUENCY",
            "ROUTE",
            "DURATION",
            "PROCEDURE",
            "LAB_VALUE",
            "ANATOMY",
            "DATE",
        }
    ),
    "pxcorpus": frozenset({"MEDICATION", "DOSAGE", "DURATION", "FREQUENCY", "ROUTE"}),
    "e3c_layer1": frozenset({"DIAGNOSIS", "ANATOMY", "LAB_VALUE", "DATE", "DURATION", "FREQUENCY"}),
    "bc5cdr": frozenset({"DIAGNOSIS", "MEDICATION"}),
    "ncbi_disease": frozenset({"DIAGNOSIS"}),
}


def is_torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def resolve_torch_device(device: str | None = "auto") -> Any:
    """Resolve 'auto' to CUDA when available, otherwise CPU (or explicit device string)."""
    import torch

    if device is None or str(device).strip().lower() in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def infer_annotated_types(doc: AnnotatedDocument) -> frozenset[str]:
    """Return the exact entity-type subset annotated by `doc.source`."""
    meta_types = doc.metadata.get("annotated_types")
    if isinstance(meta_types, list) and meta_types:
        return frozenset(str(t) for t in meta_types if str(t) in ALL_ENTITY_TYPES)
    src = doc.source.lower()
    for prefix, types in CORPUS_ANNOTATED_TYPES.items():
        if src.startswith(prefix):
            return types
    return ALL_ENTITY_TYPES


def build_char_vocab(train_docs: list[AnnotatedDocument], max_chars: int = 160) -> dict[str, int]:
    """Build train-only character vocabulary (<PAD>=0, <UNK>=1)."""
    counts: dict[str, int] = {}
    for d in train_docs:
        for ch in d.text:
            counts[ch] = counts.get(ch, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: max_chars - 2]
    c2i = {"<PAD>": 0, "<UNK>": 1}
    for ch, _ in ordered:
        c2i[ch] = len(c2i)
    return c2i


def _encode_word_chars(word: str, char_to_id: dict[str, int]) -> list[int]:
    ids = [char_to_id.get(ch, 1) for ch in word[:MAX_WORD_CHARS]]
    if len(ids) < MAX_WORD_CHARS:
        ids.extend([0] * (MAX_WORD_CHARS - len(ids)))
    return ids


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+")


def _chunk_tokens_for_training(
    tokens: list[Token],
    max_len: int = MAX_CHUNK_TOKENS,
) -> list[tuple[int, int]]:
    """Split a long token sequence into sentence-aligned chunks <= `max_len` tokens."""
    n = len(tokens)
    if n == 0:
        return []
    if n <= max_len:
        return [(0, n)]
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + max_len)
        if end < n:
            # Prefer breaking right after sentence-ending punctuation
            best = end
            for k in range(end - 1, max(start + max_len // 2, start), -1):
                if tokens[k].text in {".", "!", "?", ";"}:
                    best = k + 1
                    break
            end = best
        chunks.append((start, end))
        start = end
    return chunks


def build_allowed_tag_mask(
    y_ids: list[int],
    annotated_types: frozenset[str],
) -> np.ndarray:
    """Build [T, NUM_TAGS] boolean mask for exact Partial-Label Marginal CRF loss."""
    t_len = len(y_ids)
    allowed = np.zeros((t_len, NUM_TAGS), dtype=bool)
    unannotated_tag_ids = [
        TAG_TO_ID[tag]
        for tag in BIO_TAGS
        if tag != "O" and tag.split("-", 1)[1] not in annotated_types
    ]
    o_id = TAG_TO_ID["O"]
    for i, tid in enumerate(y_ids):
        if tid == o_id:
            allowed[i, o_id] = True
            for utid in unannotated_tag_ids:
                allowed[i, utid] = True
        else:
            allowed[i, tid] = True
    return allowed


def _build_torch_ner_module(
    *,
    vocab_size: int,
    char_vocab_size: int,
    word_emb_dim: int,
    char_emb_dim: int,
    char_filters: int,
    cluster_vocab_size: int,
    cluster_emb_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    encoder_type: str = "bilstm",
) -> Any:
    import torch
    import torch.nn as nn

    class CharCNNBiLSTMCRFNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_type = encoder_type
            self.word_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
            self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
            self.char_convs = nn.ModuleList(
                [
                    nn.Conv1d(char_emb_dim, char_filters, kernel_size=k, padding=k // 2)
                    for k in (3, 4, 5)
                ]
            )
            self.cluster_emb = nn.Embedding(cluster_vocab_size, cluster_emb_dim, padding_idx=0)
            in_dim = word_emb_dim + 3 * char_filters + CHAR_FEAT_DIM + cluster_emb_dim
            self.in_drop = nn.Dropout(dropout)
            d_model = 2 * hidden_dim
            if encoder_type == "transformer":
                n_heads = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else 2)
                self.in_proj = nn.Linear(in_dim, d_model)
                self.pos_emb = nn.Embedding(512, d_model)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(enc_layer, num_layers=max(2, num_layers))
                self.final_norm = nn.LayerNorm(d_model)
                self.lstm: nn.Module | None = None
            else:
                self.lstm = nn.LSTM(
                    input_size=in_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
            self.out_drop = nn.Dropout(dropout)
            self.fc = nn.Linear(d_model, NUM_TAGS)

            # Linear-chain CRF parameters (transitions[i, j] = score i -> j)
            self.transitions = nn.Parameter(torch.empty(NUM_TAGS, NUM_TAGS))
            self.start_transitions = nn.Parameter(torch.empty(NUM_TAGS))
            self.end_transitions = nn.Parameter(torch.empty(NUM_TAGS))
            self._init_random_weights()

        def _init_random_weights(self) -> None:
            nn.init.uniform_(self.word_emb.weight, -0.1, 0.1)
            nn.init.uniform_(self.char_emb.weight, -0.1, 0.1)
            nn.init.uniform_(self.cluster_emb.weight, -0.1, 0.1)
            with torch.no_grad():
                self.word_emb.weight[0].zero_()
                self.char_emb.weight[0].zero_()
                self.cluster_emb.weight[0].zero_()
            for conv in self.char_convs:
                nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
            if self.lstm is not None:
                for name, param in self.lstm.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
            nn.init.uniform_(self.transitions, -0.1, 0.1)
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        def encode(
            self,
            word_ids: torch.Tensor,
            char_ids: torch.Tensor,
            ortho_feats: torch.Tensor,
            cluster_ids: torch.Tensor,
        ) -> torch.Tensor:
            bsz, seq_len, c_len = char_ids.shape
            w_e = self.word_emb(word_ids)
            c_flat = self.char_emb(char_ids.view(bsz * seq_len, c_len)).transpose(1, 2)
            c_outs = [torch.relu(conv(c_flat)).max(dim=-1).values for conv in self.char_convs]
            c_e = torch.cat(c_outs, dim=-1).view(bsz, seq_len, -1)
            cl_e = self.cluster_emb(cluster_ids)
            x = self.in_drop(torch.cat([w_e, c_e, ortho_feats, cl_e], dim=-1))
            if self.encoder_type == "transformer":
                pos = torch.arange(seq_len, device=word_ids.device).clamp(max=511).unsqueeze(0)
                h_in = self.in_proj(x) + self.pos_emb(pos)
                pad_mask = word_ids == 0
                # Ensure no row is 100% masked to avoid NaN in attention
                all_pad = pad_mask.all(dim=1, keepdim=True)
                pad_mask = pad_mask & (~all_pad)
                h = self.final_norm(self.transformer(h_in, src_key_padding_mask=pad_mask))
                return h
            assert self.lstm is not None
            h, _ = self.lstm(x)
            return h

        def forward_emissions(
            self,
            word_ids: torch.Tensor,
            char_ids: torch.Tensor,
            ortho_feats: torch.Tensor,
            cluster_ids: torch.Tensor,
        ) -> torch.Tensor:
            h = self.encode(word_ids, char_ids, ortho_feats, cluster_ids)
            return self.fc(self.out_drop(h))

        def _forward_log_partition(
            self,
            emissions: torch.Tensor,
            seq_mask: torch.Tensor,
        ) -> torch.Tensor:
            # emissions: [B, T, C], seq_mask: [B, T] bool
            score = self.start_transitions.unsqueeze(0) + emissions[:, 0, :]
            trans = self.transitions.unsqueeze(0)  # [1, C, C]
            for t in range(1, emissions.shape[1]):
                emit_t = emissions[:, t, :].unsqueeze(1)  # [B, 1, C]
                next_score = torch.logsumexp(score.unsqueeze(2) + trans + emit_t, dim=1)
                m_t = seq_mask[:, t].unsqueeze(1)
                score = torch.where(m_t, next_score, score)
            score = score + self.end_transitions.unsqueeze(0)
            return torch.logsumexp(score, dim=1)

        def partial_crf_loss(
            self,
            emissions: torch.Tensor,
            allowed_mask: torch.Tensor,
            seq_mask: torch.Tensor,
        ) -> torch.Tensor:
            log_z = self._forward_log_partition(emissions, seq_mask)
            masked_emissions = emissions.masked_fill(~allowed_mask, -1e4)
            log_z_constrained = self._forward_log_partition(masked_emissions, seq_mask)
            lengths = seq_mask.float().sum(dim=1).clamp(min=1.0)
            return torch.mean((log_z - log_z_constrained) / lengths)

        def viterbi_decode(
            self,
            emissions: torch.Tensor,
            seq_mask: torch.Tensor,
        ) -> tuple[list[list[int]], list[list[float]]]:
            bsz, seq_len, _ = emissions.shape
            probs = torch.softmax(emissions, dim=-1)
            score = self.start_transitions.unsqueeze(0) + emissions[:, 0, :]
            history: list[torch.Tensor] = []
            trans = self.transitions.unsqueeze(0)
            for t in range(1, seq_len):
                broadcast = score.unsqueeze(2) + trans  # [B, C, C]
                best_prev_score, best_prev_idx = broadcast.max(dim=1)
                next_score = best_prev_score + emissions[:, t, :]
                m_t = seq_mask[:, t].unsqueeze(1)
                score = torch.where(m_t, next_score, score)
                history.append(best_prev_idx)
            score = score + self.end_transitions.unsqueeze(0)
            best_last = score.argmax(dim=1)

            paths: list[list[int]] = []
            confs: list[list[float]] = []
            lengths = seq_mask.long().sum(dim=1).tolist()
            for b in range(bsz):
                L = int(lengths[b])
                cur = int(best_last[b].item())
                path = [cur]
                for t in range(L - 2, -1, -1):
                    cur = int(history[t][b, cur].item())
                    path.append(cur)
                path.reverse()
                p_conf = [float(probs[b, idx, tid].item()) for idx, tid in enumerate(path)]
                paths.append(path)
                confs.append(p_conf)
            return paths, confs

    return CharCNNBiLSTMCRFNet()


class PyTorchCharCNNBiLSTMCRF:
    """Trainable, saveable from-scratch PyTorch Char-CNN + BiLSTM + Partial-Label Marginal CRF."""

    def __init__(
        self,
        *,
        word_emb_dim: int = 48,
        char_emb_dim: int = 16,
        char_filters: int = 16,
        cluster_emb_dim: int = 16,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.25,
        lr: float = 0.003,
        epochs: int = 12,
        batch_size: int = 16,
        use_self_supervised: bool = True,
        self_supervised_path: str | None = None,
        ocr_augment_ratio: float = 0.0,
        encoder_type: str = "bilstm",
        seed: int = 42,
        device: str | None = "auto",
    ) -> None:
        self.word_emb_dim = word_emb_dim
        self.char_emb_dim = char_emb_dim
        self.char_filters = char_filters
        self.cluster_emb_dim = cluster_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_self_supervised = use_self_supervised
        self.self_supervised_path = self_supervised_path
        self.ocr_augment_ratio = ocr_augment_ratio
        self.encoder_type = encoder_type
        self.seed = seed
        self.device_spec = device
        self._device: Any = None

        self.vocab = TrainOnlyVocabulary(max_size=24000, min_freq=1)
        self.char_to_id: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.cluster_vocab_size: int = 66  # 0=PAD/OOV, 1..64=fine clusters, 65=fallback
        self.sem_model: SelfSupervisedMedicalEmbeddings | None = (
            load_optional_self_supervised(self_supervised_path) if use_self_supervised else None
        )
        self.word_to_cluster_id: dict[str, int] = {}
        if self.sem_model is not None:
            for w, idx in self.sem_model.word_to_id.items():
                self.word_to_cluster_id[w] = min(self.cluster_vocab_size - 1, int(self.sem_model.clusters_fine[idx]) + 1)
        self.net: Any = None
        self.initial_weights_sha256: str = ""
        self.trained_weights_sha256: str = ""
        self.training_device: str = "cpu"
        self._fitted = False

    def _cluster_id(self, word: str) -> int:
        wl = word.lower()
        if self.word_to_cluster_id:
            return self.word_to_cluster_id.get(wl, 0)
        if self.sem_model is None:
            return 0
        cl = self.sem_model.get_clusters(wl)
        if cl is None:
            return 0
        return min(self.cluster_vocab_size - 1, cl[0] + 1)

    def _encode_chunk(
        self,
        toks: list[Token],
        y_ids: list[int] | None = None,
        annotated_types: frozenset[str] = ALL_ENTITY_TYPES,
    ) -> dict[str, Any]:
        w_ids = [self.vocab.encode_token(t.text) for t in toks]
        c_ids = [_encode_word_chars(t.text, self.char_to_id) for t in toks]
        ortho = _ortho_features(toks).astype(np.float32)
        cl_ids = [self._cluster_id(t.text) for t in toks]
        item: dict[str, Any] = {
            "w_ids": np.array(w_ids, dtype=np.int64),
            "c_ids": np.array(c_ids, dtype=np.int64),
            "ortho": ortho,
            "cl_ids": np.array(cl_ids, dtype=np.int64),
        }
        if y_ids is not None:
            item["allowed"] = build_allowed_tag_mask(y_ids, annotated_types)
        return item

    def _collate_batch(
        self,
        items: list[dict[str, Any]],
        *,
        word_drop_p: float = 0.0,
        device: Any = None,
    ) -> dict[str, Any]:
        import torch

        bsz = len(items)
        max_t = max(len(it["w_ids"]) for it in items)
        w = np.zeros((bsz, max_t), dtype=np.int64)
        c = np.zeros((bsz, max_t, MAX_WORD_CHARS), dtype=np.int64)
        o = np.zeros((bsz, max_t, CHAR_FEAT_DIM), dtype=np.float32)
        cl = np.zeros((bsz, max_t), dtype=np.int64)
        mask = np.zeros((bsz, max_t), dtype=bool)
        has_allowed = "allowed" in items[0]
        allowed = np.zeros((bsz, max_t, NUM_TAGS), dtype=bool) if has_allowed else None
        if allowed is not None:
            allowed[:, :, TAG_TO_ID["O"]] = True

        for i, it in enumerate(items):
            L = len(it["w_ids"])
            w_arr = it["w_ids"]
            if word_drop_p > 0.0:
                drop = np.random.random(L) < word_drop_p
                w_arr = np.where(drop, 1, w_arr)
            w[i, :L] = w_arr
            c[i, :L, :] = it["c_ids"]
            o[i, :L, :] = it["ortho"]
            cl[i, :L] = it["cl_ids"]
            mask[i, :L] = True
            if allowed is not None:
                allowed[i, :L, :] = it["allowed"]

        target_dev = device if device is not None else self._device
        out: dict[str, Any] = {
            "w": torch.from_numpy(w),
            "c": torch.from_numpy(c),
            "o": torch.from_numpy(o),
            "cl": torch.from_numpy(cl),
            "mask": torch.from_numpy(mask),
        }
        if allowed is not None:
            out["allowed"] = torch.from_numpy(allowed)
        if target_dev is not None:
            out = {k: v.to(target_dev, non_blocking=(target_dev.type == "cuda")) for k, v in out.items()}
        return out

    def fit(
        self,
        train_docs: list[AnnotatedDocument],
        *,
        verbose: bool = False,
        pretrain_texts: list[str] | None = None,
        mlm_epochs: int = 0,
        mlm_lr: float = 0.002,
    ) -> PyTorchCharCNNBiLSTMCRF:
        if not is_torch_available():
            raise RuntimeError("PyTorch is required for PyTorchCharCNNBiLSTMCRF (install .[vision]).")
        import torch

        dev = resolve_torch_device(self.device_spec)
        self._device = dev
        self.training_device = str(dev)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        else:
            torch.set_num_threads(min(6, max(1, torch.get_num_threads())))

        fit_docs = (
            build_ocr_augmented_train_set(train_docs, ratio=self.ocr_augment_ratio, seed=self.seed)
            if self.ocr_augment_ratio > 0.0
            else list(train_docs)
        )
        self.vocab.fit(fit_docs)
        self.char_to_id = build_char_vocab(fit_docs)

        all_chunks: list[dict[str, Any]] = []
        for d in fit_docs:
            toks = tokenize_with_offsets(d.text)
            if not toks:
                continue
            tags = align_entities_to_bio(toks, d.entities)
            y_ids = [TAG_TO_ID[t] for t in tags]
            ann_types = infer_annotated_types(d)
            for s, e in _chunk_tokens_for_training(toks):
                all_chunks.append(self._encode_chunk(toks[s:e], y_ids[s:e], ann_types))

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(all_chunks))
        n_es = max(1, int(0.10 * len(all_chunks))) if len(all_chunks) >= 20 else 0
        es_chunks = [all_chunks[int(i)] for i in perm[:n_es]]
        tr_chunks = [all_chunks[int(i)] for i in perm[n_es:]] if n_es > 0 else all_chunks

        self.net = _build_torch_ner_module(
            vocab_size=len(self.vocab),
            char_vocab_size=len(self.char_to_id),
            word_emb_dim=self.word_emb_dim,
            char_emb_dim=self.char_emb_dim,
            char_filters=self.char_filters,
            cluster_vocab_size=self.cluster_vocab_size,
            cluster_emb_dim=self.cluster_emb_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            encoder_type=self.encoder_type,
        ).to(dev)
        self.initial_weights_sha256 = self._weights_sha256()

        # Optional from-scratch Self-Supervised Masked Language Modeling (MLM) pretraining
        if pretrain_texts and mlm_epochs > 0:
            import torch.nn as nn

            mlm_chunks: list[dict[str, Any]] = []
            for txt in pretrain_texts:
                toks_p = tokenize_with_offsets(txt)
                if len(toks_p) < 6:
                    continue
                for s, e in _chunk_tokens_for_training(toks_p):
                    mlm_chunks.append(self._encode_chunk(toks_p[s:e]))
            if mlm_chunks:
                mlm_head = nn.Linear(2 * self.hidden_dim, len(self.vocab)).to(dev)
                mlm_opt = torch.optim.AdamW(
                    list(self.net.parameters()) + list(mlm_head.parameters()),
                    lr=mlm_lr,
                    weight_decay=1e-4,
                )
                mlm_ce = nn.CrossEntropyLoss(ignore_index=-100)
                mlm_bs = max(self.batch_size, 32)
                for m_ep in range(mlm_epochs):
                    t_m = time.perf_counter()
                    self.net.train()
                    mlm_head.train()
                    m_order = rng.permutation(len(mlm_chunks))
                    m_losses: list[float] = []
                    for start in range(0, len(m_order), mlm_bs):
                        b_items = [mlm_chunks[int(i)] for i in m_order[start : start + mlm_bs]]
                        batch = self._collate_batch(b_items, word_drop_p=0.0, device=dev)
                        w_orig = batch["w"]
                        # Mask 15% of known content tokens (word_id > 1)
                        valid_tok = batch["mask"] & (w_orig > 1)
                        rand_mask = (torch.rand_like(w_orig.float()) < 0.15) & valid_tok
                        if not bool(rand_mask.any().item()):
                            continue
                        targets = torch.where(rand_mask, w_orig, torch.full_like(w_orig, -100))
                        w_in = torch.where(rand_mask, torch.ones_like(w_orig), w_orig)
                        c_in = batch["c"].masked_fill(rand_mask.unsqueeze(-1), 0)
                        o_in = batch["o"].masked_fill(rand_mask.unsqueeze(-1), 0.0)
                        cl_in = batch["cl"].masked_fill(rand_mask, 0)

                        mlm_opt.zero_grad()
                        h_seq = self.net.encode(w_in, c_in, o_in, cl_in)
                        logits = mlm_head(h_seq)
                        loss_mlm = mlm_ce(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                        loss_mlm.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(self.net.parameters()) + list(mlm_head.parameters()), 5.0
                        )
                        mlm_opt.step()
                        m_losses.append(float(loss_mlm.item()))
                    if verbose:
                        print(
                            f"  [MLM seed={self.seed} ep={m_ep + 1}/{mlm_epochs} chunks={len(mlm_chunks)}] "
                            f"loss={float(np.mean(m_losses)) if m_losses else 0.0:.4f} ({time.perf_counter() - t_m:.1f}s)",
                            flush=True,
                        )
                del mlm_head, mlm_opt

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.epochs))

        best_es_loss = float("inf")
        best_state: dict[str, Any] = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        stale = 0

        for ep in range(self.epochs):
            t_ep = time.perf_counter()
            self.net.train()
            order = rng.permutation(len(tr_chunks))
            tr_losses: list[float] = []
            for start in range(0, len(order), self.batch_size):
                b_items = [tr_chunks[int(i)] for i in order[start : start + self.batch_size]]
                batch = self._collate_batch(b_items, word_drop_p=0.10, device=dev)
                optimizer.zero_grad()
                emissions = self.net.forward_emissions(batch["w"], batch["c"], batch["o"], batch["cl"])
                loss = self.net.partial_crf_loss(emissions, batch["allowed"], batch["mask"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                optimizer.step()
                tr_losses.append(float(loss.item()))
            scheduler.step()

            if es_chunks:
                self.net.eval()
                es_losses: list[float] = []
                with torch.no_grad():
                    for start in range(0, len(es_chunks), self.batch_size):
                        batch = self._collate_batch(
                            es_chunks[start : start + self.batch_size],
                            word_drop_p=0.0,
                            device=dev,
                        )
                        em = self.net.forward_emissions(batch["w"], batch["c"], batch["o"], batch["cl"])
                        es_losses.append(float(self.net.partial_crf_loss(em, batch["allowed"], batch["mask"]).item()))
                mean_es = float(np.mean(es_losses))
                if verbose:
                    print(
                        f"  [CharCNNBiLSTMCRF dev={dev}] ep={ep+1}/{self.epochs} tr_loss={np.mean(tr_losses):.4f} "
                        f"es_loss={mean_es:.4f} ({time.perf_counter()-t_ep:.1f}s)",
                        flush=True,
                    )
                if mean_es < best_es_loss - 1e-4:
                    best_es_loss = mean_es
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= 3:
                        break
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}

        self.net.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        self.net.eval()
        self.trained_weights_sha256 = self._weights_sha256()
        self._fitted = True
        return self

    def _weights_sha256(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.net.state_dict().keys()):
            h.update(k.encode("utf-8"))
            h.update(self.net.state_dict()[k].detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def predict_entities(self, text: str) -> list[PredictedEntity]:
        if not self._fitted or self.net is None:
            raise RuntimeError("PyTorchCharCNNBiLSTMCRF must be fitted before predict_entities()")
        import torch

        toks = tokenize_with_offsets(text)
        if not toks:
            return []
        self.net.eval()
        dev = self._device if self._device is not None else next(self.net.parameters()).device
        all_tags: list[str] = []
        all_confs: list[float] = []
        with torch.no_grad():
            for s, e in _chunk_tokens_for_training(toks):
                chunk_item = self._encode_chunk(toks[s:e])
                batch = self._collate_batch([chunk_item], word_drop_p=0.0, device=dev)
                em = self.net.forward_emissions(batch["w"], batch["c"], batch["o"], batch["cl"])
                paths, confs = self.net.viterbi_decode(em, batch["mask"])
                all_tags.extend(ID_TO_TAG[tid] for tid in paths[0])
                all_confs.extend(confs[0])
        return decode_bio_to_entities(text, toks, all_tags, all_confs)

    def save(self, model_dir: Path | str) -> Path:
        if not self._fitted or self.net is None:
            raise RuntimeError("PyTorchCharCNNBiLSTMCRF must be fitted before save()")
        import torch

        out_dir = Path(model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_path = out_dir / "char_cnn_bilstm_crf.pt"
        cpu_state = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        torch.save(cpu_state, weights_path)
        meta = {
            "model_type": "pytorch_char_cnn_bilstm_crf",
            "pretrained_weights_used": False,
            "training_device": self.training_device,
            "initial_weights_sha256": self.initial_weights_sha256,
            "trained_weights_sha256": self.trained_weights_sha256,
            "weights_file_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "word_emb_dim": self.word_emb_dim,
                "char_emb_dim": self.char_emb_dim,
                "char_filters": self.char_filters,
                "cluster_vocab_size": self.cluster_vocab_size,
                "cluster_emb_dim": self.cluster_emb_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "encoder_type": self.encoder_type,
                "use_self_supervised": self.use_self_supervised,
                "self_supervised_path": self.self_supervised_path,
                "seed": self.seed,
            },
            "vocab": {
                "max_size": self.vocab.max_size,
                "min_freq": self.vocab.min_freq,
                "word_to_id": self.vocab.word_to_id,
            },
            "char_to_id": self.char_to_id,
            "word_to_cluster_id": self.word_to_cluster_id,
        }
        (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return out_dir

    @classmethod
    def load(cls, model_dir: Path | str, *, device: str | None = "cpu") -> PyTorchCharCNNBiLSTMCRF:
        if not is_torch_available():
            raise RuntimeError("PyTorch is required to load PyTorchCharCNNBiLSTMCRF.")
        import torch

        in_dir = Path(model_dir)
        meta = json.loads((in_dir / "manifest.json").read_text(encoding="utf-8"))
        weights_path = in_dir / "char_cnn_bilstm_crf.pt"
        actual_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if actual_sha != meta.get("weights_file_sha256"):
            raise ValueError(f"SHA-256 mismatch for {weights_path}")

        dev = resolve_torch_device(device)
        cfg = meta["config"]
        obj = cls(
            word_emb_dim=int(cfg["word_emb_dim"]),
            char_emb_dim=int(cfg["char_emb_dim"]),
            char_filters=int(cfg["char_filters"]),
            cluster_emb_dim=int(cfg["cluster_emb_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
            encoder_type=str(cfg.get("encoder_type", "bilstm")),
            use_self_supervised=bool(cfg.get("use_self_supervised", True)),
            self_supervised_path=cfg.get("self_supervised_path"),
            seed=int(cfg["seed"]),
            device=str(dev),
        )
        obj._device = dev
        obj.training_device = str(meta.get("training_device", "cpu"))
        obj.cluster_vocab_size = int(cfg["cluster_vocab_size"])
        obj.vocab.max_size = int(meta["vocab"]["max_size"])
        obj.vocab.min_freq = int(meta["vocab"]["min_freq"])
        obj.vocab.word_to_id = {str(k): int(v) for k, v in meta["vocab"]["word_to_id"].items()}
        obj.char_to_id = {str(k): int(v) for k, v in meta["char_to_id"].items()}
        if isinstance(meta.get("word_to_cluster_id"), dict):
            obj.word_to_cluster_id = {str(k): int(v) for k, v in meta["word_to_cluster_id"].items()}
        obj.net = _build_torch_ner_module(
            vocab_size=len(obj.vocab),
            char_vocab_size=len(obj.char_to_id),
            word_emb_dim=obj.word_emb_dim,
            char_emb_dim=obj.char_emb_dim,
            char_filters=obj.char_filters,
            cluster_vocab_size=obj.cluster_vocab_size,
            cluster_emb_dim=obj.cluster_emb_dim,
            hidden_dim=obj.hidden_dim,
            num_layers=obj.num_layers,
            dropout=obj.dropout,
            encoder_type=obj.encoder_type,
        ).to(dev)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        obj.net.load_state_dict({k: v.to(dev) for k, v in state.items()})
        obj.net.eval()
        obj.initial_weights_sha256 = str(meta.get("initial_weights_sha256", ""))
        obj.trained_weights_sha256 = str(meta.get("trained_weights_sha256", ""))
        obj._fitted = True
        return obj


def ensemble_torch_ner_predictions(
    models: list[PyTorchCharCNNBiLSTMCRF],
    text: str,
) -> list[PredictedEntity]:
    """Consensus multi-seed span ensemble for PyTorchCharCNNBiLSTMCRF models."""
    if not models:
        return []
    if len(models) == 1:
        return models[0].predict_entities(text)

    n_models = len(models)
    votes: dict[tuple[str, int, int], list[PredictedEntity]] = {}
    for m in models:
        for ent in m.predict_entities(text):
            votes.setdefault((ent.type, ent.start, ent.end), []).append(ent)

    min_votes = max(1, (2 * n_models + 2) // 3)
    hi_conf_threshold = 0.94 if n_models >= 5 else 0.88
    candidates: list[tuple[int, float, PredictedEntity]] = []
    for (_etype, _s, _e), ents in votes.items():
        voter_confs = [float(e.confidence) for e in ents]
        mean_conf = float(np.mean(voter_confs))
        if len(ents) >= min_votes or mean_conf >= hi_conf_threshold:
            full_scores = voter_confs + [0.0] * (n_models - len(ents))
            seed_std = round(float(np.std(full_scores)), 4)
            is_high_unc = bool(seed_std > 0.12)
            boost = 0.04 * (len(ents) - 1)
            merged_ent = PredictedEntity(
                type=ents[0].type,
                start=ents[0].start,
                end=ents[0].end,
                text=ents[0].text,
                assertion=ents[0].assertion,
                confidence=round(min(0.99, mean_conf + boost), 4),
                needs_review=is_high_unc,
                confidence_variance=seed_std,
                high_uncertainty=is_high_unc,
            )
            candidates.append((len(ents), merged_ent.confidence, merged_ent))

    # Greedy non-overlapping selection ordered by vote count then confidence
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2].start))
    selected: list[PredictedEntity] = []
    for _cnt, _conf, ent in candidates:
        if any(max(ent.start, s.start) < min(ent.end, s.end) for s in selected):
            continue
        selected.append(ent)
    return sorted(selected, key=lambda x: x.start)


def predict_torch_char_cnn_bilstm_crf(
    train_docs: list[AnnotatedDocument],
    eval_docs: list[AnnotatedDocument],
    seed: int,
    cfg: dict[str, Any],
) -> list[DocumentPrediction]:
    """Runner-compatible entrypoint for from-scratch PyTorch Char-CNN + BiLSTM + Partial-Label CRF."""
    model = PyTorchCharCNNBiLSTMCRF(
        word_emb_dim=int(cfg.get("word_emb_dim", 48)),
        char_emb_dim=int(cfg.get("char_emb_dim", 16)),
        char_filters=int(cfg.get("char_filters", 16)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.25)),
        lr=float(cfg.get("lr", 0.003)),
        epochs=int(cfg.get("epochs", 12)),
        batch_size=int(cfg.get("batch_size", 16)),
        use_self_supervised=bool(cfg.get("use_self_supervised", True)),
        ocr_augment_ratio=float(cfg.get("ocr_augment_ratio", 0.20)),
        seed=seed,
    ).fit(train_docs, verbose=bool(cfg.get("verbose", False)))
    pred_doc_types = _doc_classifier_fit_predict(train_docs, eval_docs, seed)
    preds: list[DocumentPrediction] = []
    for d, dt_pred in zip(eval_docs, pred_doc_types, strict=True):
        ents = model.predict_entities(d.text)
        enriched, relations, timex = apply_context_and_relations(d, ents)
        preds.append(
            DocumentPrediction(
                doc_id=d.doc_id,
                language=d.language,
                doc_type=d.doc_type,
                predicted_doc_type=dt_pred,
                entities=enriched,
                relations=relations,
                timex=timex,
            )
        )
    return preds
