"""Step 2 Leakage Audit on `locked_test_v2` using dHash + PyTorch CNN penultimate embeddings."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.cnn_torch import (
    PyTorchDermResNet,
    _compute_binary_auroc,
)
from medresearch_platform.vision.eval_vision import load_stratified_skin_dataset


def _compute_dhash_bits(img_float01: np.ndarray) -> np.ndarray:
    """Compute 64-bit dHash (shape (64,)) from a float [0,1] HxWx3 array as loaded by the stratified loader."""
    img_uint8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(img_uint8).convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.int16)
    diff = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    return diff.astype(np.uint8)


def _extract_embeddings_and_probs(
    model: PyTorchDermResNet,
    samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return L2-normalised 256-d pooled pixel embeddings (N, 256) and malignancy probs (N,)."""
    net = model.net
    net.eval()
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            chunk = samples[i : i + batch_size]
            xb = torch.stack([model._arr_to_tensor(a) for a, _, _ in chunk], dim=0)
            h = net.stage4(net.stage3(net.stage2(net.stage1(net.stem(xb)))))
            pooled = net.pool(h).flatten(1)
            norm = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-8)
            embs.append(norm.cpu().numpy())
    p_mal, _, _ = model.predict_batch(samples, batch_size=64)
    return np.concatenate(embs, axis=0), np.asarray(p_mal)


def run_v2_leakage_audit(
    out_json: str | Path = "docs/results/vision/v2_leakage_audit.json",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    loaded = load_stratified_skin_dataset(
        meta_dir="var/cache/derm/meta",
        image_size=96,
        include_masked_mg=True,
        locked_test_list_path="configs/vision/locked_test_v2.txt",
        min_real_samples=100,
    )
    train_lab = [s for s in loaded["train"] if s[2].malignancy_group in ("benign", "malignant")]
    test_lab = [s for s in loaded["locked_test"] if s[2].malignancy_group in ("benign", "malignant")]

    model = PyTorchDermResNet.load("models/derm_cnn_seed1")
    tr_emb, _ = _extract_embeddings_and_probs(model, train_lab)
    te_emb, te_probs = _extract_embeddings_and_probs(model, test_lab)

    # dHash matrices (uint8 0/1 of shape N x 64)
    tr_dh = np.stack([_compute_dhash_bits(s[0]) for s in train_lab], axis=0).astype(np.float32)
    te_dh = np.stack([_compute_dhash_bits(s[0]) for s in test_lab], axis=0).astype(np.float32)
    # Hamming distance = sum(a) + sum(b) - 2 * a @ b.T
    tr_sum = tr_dh.sum(axis=1)[None, :]
    te_sum = te_dh.sum(axis=1)[:, None]
    hamming = te_sum + tr_sum - 2.0 * (te_dh @ tr_dh.T)
    min_dhash = hamming.min(axis=1)

    # Cosine similarity (N_test x N_train)
    cos_mat = te_emb @ tr_emb.T
    max_cos_any = cos_mat.max(axis=1)

    # Same modality + same dx mask
    tr_mod = np.array([s[2].modality for s in train_lab])
    tr_dx = np.array([s[2].diagnosis_harmonised for s in train_lab])
    same_mod_dx_max_cos = np.zeros(len(test_lab), dtype=np.float32)
    for idx, (_, _, rec) in enumerate(test_lab):
        mask = (tr_mod == rec.modality) & (tr_dx == rec.diagnosis_harmonised)
        if np.any(mask):
            same_mod_dx_max_cos[idx] = float(cos_mat[idx, mask].max())
        else:
            same_mod_dx_max_cos[idx] = -1.0

    y_test = np.array([1 if s[2].malignancy_group == "malignant" else 0 for s in test_lab], dtype=np.int64)
    te_mods = np.array([s[2].modality for s in test_lab])

    # Thresholds
    close_dhash10 = min_dhash <= 10.0
    close_emb95 = same_mod_dx_max_cos >= 0.95
    close_combined = close_dhash10 | close_emb95

    def _subset_stats(mask: np.ndarray) -> dict[str, Any]:
        n = int(mask.sum())
        pos = int(y_test[mask].sum()) if n else 0
        neg = n - pos
        auroc = _compute_binary_auroc(y_test[mask], te_probs[mask]) if (pos > 0 and neg > 0) else None
        return {
            "n": n,
            "malignant": pos,
            "benign": neg,
            "auroc": round(float(auroc), 4) if auroc is not None else None,
        }

    by_modality: dict[str, Any] = {}
    for mod in ("dermoscopic", "clinical", "tbp_tile"):
        mm = te_mods == mod
        by_modality[mod] = {
            "all": _subset_stats(mm),
            "close_neighbour": _subset_stats(mm & close_combined),
            "no_close_neighbour": _subset_stats(mm & ~close_combined),
        }

    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "model": "models/derm_cnn_seed1",
        "n_train_labelled": len(train_lab),
        "n_test_v2_labelled": len(test_lab),
        "rule": "close_neighbour = (min_dhash_dist <= 10) OR (max_cos_sim >= 0.95 AND same modality AND same diagnosis_harmonised)",
        "overall": {
            "all": _subset_stats(np.ones(len(test_lab), dtype=bool)),
            "close_neighbour_combined": _subset_stats(close_combined),
            "no_close_neighbour_combined": _subset_stats(~close_combined),
            "close_dhash_le_10": _subset_stats(close_dhash10),
            "no_close_dhash_le_10": _subset_stats(~close_dhash10),
            "close_emb_ge_095_same_mod_dx": _subset_stats(close_emb95),
            "no_close_emb_ge_095_same_mod_dx": _subset_stats(~close_emb95),
        },
        "by_modality": by_modality,
        "min_dhash_quantiles": {
            q: round(float(np.quantile(min_dhash, p)), 2)
            for q, p in [("p05", 0.05), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75)]
        },
        "max_cos_any_quantiles": {
            q: round(float(np.quantile(max_cos_any, p)), 4)
            for q, p in [("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p95", 0.95)]
        },
    }

    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_v2_leakage_audit(), indent=2))
