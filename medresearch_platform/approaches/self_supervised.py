"""From-scratch self-supervised EN/FR clinical word representations (SVD-PPMI + semantic clusters).

Trains strictly from random initialization / unlabeled clinical text (zero external pretrained weights)
using sparse Shifted Positive Pointwise Mutual Information (SPPMI) + truncated SVD + MiniBatchKMeans.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans

from medresearch_platform.approaches.tokenizer import tokenize_with_offsets

_GLOBAL_MODEL_CACHE: dict[str, SelfSupervisedMedicalEmbeddings] = {}


class SelfSupervisedMedicalEmbeddings:
    """From-scratch dense SVD-PPMI embeddings and discrete semantic word clusters."""

    def __init__(
        self,
        word_to_id: dict[str, int],
        embeddings: np.ndarray,
        clusters_fine: np.ndarray,
        clusters_coarse: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.word_to_id = word_to_id
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.clusters_fine = clusters_fine.astype(np.int32, copy=False)
        self.clusters_coarse = clusters_coarse.astype(np.int32, copy=False)
        self.metadata = metadata or {}

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def get_clusters(self, word: str) -> tuple[int, int] | None:
        idx = self.word_to_id.get(word.lower())
        if idx is None:
            return None
        return int(self.clusters_fine[idx]), int(self.clusters_coarse[idx])

    def get_vector(self, word: str) -> np.ndarray | None:
        idx = self.word_to_id.get(word.lower())
        if idx is None:
            return None
        return self.embeddings[idx]

    def project_to_vocab(self, vocab_word_to_id: dict[str, int], target_dim: int) -> tuple[np.ndarray, int]:
        """Return an embedding matrix aligned to `vocab_word_to_id` (zero for OOV) and matched count."""
        mat = np.zeros((len(vocab_word_to_id), target_dim), dtype=np.float64)
        k = min(target_dim, self.dim)
        matched = 0
        for w, vid in vocab_word_to_id.items():
            idx = self.word_to_id.get(w.lower())
            if idx is not None and 0 <= vid < len(vocab_word_to_id):
                vec = self.embeddings[idx, :k]
                norm = float(np.linalg.norm(vec))
                if norm > 1e-8:
                    mat[vid, :k] = (vec / norm) * 0.25
                    matched += 1
        return mat, matched

    @classmethod
    def fit_from_texts(
        cls,
        texts: list[str],
        *,
        max_vocab: int = 12000,
        min_count: int = 3,
        window: int = 4,
        dim: int = 32,
        n_clusters_fine: int = 64,
        n_clusters_coarse: int = 16,
        seed: int = 42,
        source_label: str = "unlabeled_clinical",
    ) -> SelfSupervisedMedicalEmbeddings:
        """Fit SPPMI-SVD embeddings and k-means clusters strictly from scratch on raw texts."""
        tokenized: list[list[str]] = []
        counts: Counter[str] = Counter()
        for txt in texts:
            toks = [t.text.lower() for t in tokenize_with_offsets(txt) if len(t.text) >= 2]
            if toks:
                tokenized.append(toks)
                counts.update(toks)

        vocab_items = [w for w, c in counts.most_common(max_vocab) if c >= min_count]
        if len(vocab_items) < 4:
            vocab_items = [w for w, _ in counts.most_common(max_vocab)]
        word_to_id = {w: i for i, w in enumerate(vocab_items)}
        v_size = len(word_to_id)

        if v_size < 4:
            return cls(
                word_to_id=word_to_id,
                embeddings=np.zeros((v_size, dim), dtype=np.float32),
                clusters_fine=np.zeros(v_size, dtype=np.int32),
                clusters_coarse=np.zeros(v_size, dtype=np.int32),
                metadata={"source": source_label, "pretrained_weights_used": False, "n_docs": len(texts)},
            )

        cooc: Counter[tuple[int, int]] = Counter()
        unigram = np.zeros(v_size, dtype=np.float64)
        for toks in tokenized:
            ids = [word_to_id[w] for w in toks if w in word_to_id]
            n_ids = len(ids)
            for i, wi in enumerate(ids):
                unigram[wi] += 1.0
                hi = min(n_ids, i + window + 1)
                for j in range(i + 1, hi):
                    wj = ids[j]
                    if wi == wj:
                        continue
                    cooc[(wi, wj)] += 1
                    cooc[(wj, wi)] += 1

        total_pairs = float(sum(cooc.values()))
        total_uni = max(1.0, float(np.sum(unigram)))
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        # Shifted PPMI (shift = log(2))
        shift = float(np.log(2.0))
        for (i, j), cnt in cooc.items():
            p_ij = cnt / max(1.0, total_pairs)
            p_i = unigram[i] / total_uni
            p_j = unigram[j] / total_uni
            pmi = float(np.log(max(p_ij / max(p_i * p_j, 1e-12), 1e-12))) - shift
            if pmi > 0.0:
                rows.append(i)
                cols.append(j)
                vals.append(pmi)

        mat = sparse.csr_matrix((vals, (rows, cols)), shape=(v_size, v_size), dtype=np.float64)
        eff_dim = min(dim, max(2, v_size - 2))
        if mat.nnz == 0:
            emb = np.zeros((v_size, eff_dim), dtype=np.float32)
        else:
            u, s, _ = svds(mat, k=eff_dim, random_state=seed)
            order = np.argsort(s)[::-1]
            u = u[:, order]
            s = np.maximum(s[order], 0.0)
            emb = (u * np.sqrt(s)[None, :]).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = np.where(norms > 1e-8, emb / np.maximum(norms, 1e-8), 0.0).astype(np.float32)

        k_fine = min(n_clusters_fine, max(2, v_size // 2))
        k_coarse = min(n_clusters_coarse, max(2, v_size // 4))
        km_fine = MiniBatchKMeans(n_clusters=k_fine, random_state=seed, batch_size=512, n_init=3)
        km_coarse = MiniBatchKMeans(n_clusters=k_coarse, random_state=seed + 1, batch_size=512, n_init=3)
        c_fine = km_fine.fit_predict(emb).astype(np.int32)
        c_coarse = km_coarse.fit_predict(emb).astype(np.int32)

        meta = {
            "source": source_label,
            "pretrained_weights_used": False,
            "method": "from_scratch_sppmi_svd_kmeans",
            "n_docs": len(texts),
            "vocab_size": v_size,
            "dim": eff_dim,
            "n_clusters_fine": k_fine,
            "n_clusters_coarse": k_coarse,
            "seed": seed,
        }
        return cls(word_to_id=word_to_id, embeddings=emb, clusters_fine=c_fine, clusters_coarse=c_coarse, metadata=meta)

    def save(self, out_path: Path | str) -> Path:
        dst = Path(out_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        words = np.array(
            [w for w, _ in sorted(self.word_to_id.items(), key=lambda kv: kv[1])],
            dtype=object,
        )
        np.savez_compressed(
            dst,
            words=np.array(list(words), dtype=str),
            embeddings=self.embeddings,
            clusters_fine=self.clusters_fine,
            clusters_coarse=self.clusters_coarse,
            metadata_json=np.array(json.dumps(self.metadata, sort_keys=True)),
        )
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        manifest_path = dst.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps({**self.metadata, "artifact": dst.name, "sha256": sha}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return dst

    @classmethod
    def load(cls, path: Path | str) -> SelfSupervisedMedicalEmbeddings:
        src = Path(path)
        key = str(src.resolve())
        if key in _GLOBAL_MODEL_CACHE:
            return _GLOBAL_MODEL_CACHE[key]
        with np.load(src, allow_pickle=False) as data:
            words = [str(w) for w in data["words"].tolist()]
            word_to_id = {w: i for i, w in enumerate(words)}
            emb = np.array(data["embeddings"], copy=True)
            c_fine = np.array(data["clusters_fine"], copy=True)
            c_coarse = np.array(data["clusters_coarse"], copy=True)
            meta = json.loads(str(data["metadata_json"].item()))
        obj = cls(
            word_to_id=word_to_id,
            embeddings=emb,
            clusters_fine=c_fine,
            clusters_coarse=c_coarse,
            metadata=meta,
        )
        _GLOBAL_MODEL_CACHE[key] = obj
        return obj


def load_optional_self_supervised(path: Path | str | None = None) -> SelfSupervisedMedicalEmbeddings | None:
    """Load from-scratch self-supervised clinical embeddings if present on disk."""
    target = Path(path) if path is not None else Path("models/e3c_self_supervised_en_fr.npz")
    if not target.exists():
        return None
    try:
        return SelfSupervisedMedicalEmbeddings.load(target)
    except Exception:
        return None
