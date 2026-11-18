"""Step 2 (W1) leakage audit v2: threshold ladder + chance-level null, replacing the saturated first audit.

The first audit (`v2_leakage_audit.json`) used dHash <= 10 and cosine >= 0.95; both were true for most
images by chance (median min dHash 9; median max cosine 0.991), so it could not separate subsets.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from audit_v2_leakage import _compute_dhash_bits, _extract_embeddings_and_probs  # noqa: E402  (same scripts/ dir)

from medresearch_platform.vision.cnn_torch import PyTorchDermResNet, _compute_binary_auroc
from medresearch_platform.vision.eval_vision import load_stratified_skin_dataset

MIN_PER_CLASS = 20
DHASH_LADDER = (0, 1, 2, 4, 6)
EMB_NEAR_TOP_FRACTIONS = (0.05, 0.10, 0.25)


def _min_hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Min Hamming distance of each row of `a` to all rows of `b` (both float32 0/1 matrices)."""
    return (a.sum(1)[:, None] + b.sum(1)[None, :] - 2.0 * (a @ b.T)).min(axis=1)


def _subset(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    n = int(mask.sum())
    pos = int(y[mask].sum()) if n else 0
    neg = n - pos
    ok = pos >= MIN_PER_CLASS and neg >= MIN_PER_CLASS
    return {
        "n": n,
        "malignant": pos,
        "benign": neg,
        "auroc": round(float(_compute_binary_auroc(y[mask], p[mask])), 4) if ok else None,
        "auroc_note": None if ok else f"not reported: needs >= {MIN_PER_CLASS} per class",
    }


def run(out_json: str | Path = "docs/results/vision/v2_leakage_audit_ladder.json") -> dict[str, Any]:
    t0 = time.perf_counter()
    loaded = load_stratified_skin_dataset(
        meta_dir="var/cache/derm/meta",
        image_size=96,
        include_masked_mg=True,
        locked_test_list_path="configs/vision/locked_test_v2.txt",
        min_real_samples=100,
    )
    keep = ("benign", "malignant")
    train = [s for s in loaded["train"] if s[2].malignancy_group in keep]
    test = [s for s in loaded["locked_test"] if s[2].malignancy_group in keep]

    model = PyTorchDermResNet.load("models/derm_cnn_seed1")
    tr_emb, _ = _extract_embeddings_and_probs(model, train)
    te_emb, te_p = _extract_embeddings_and_probs(model, test)

    tr_dh = np.stack([_compute_dhash_bits(s[0]) for s in train]).astype(np.float32)
    te_dh = np.stack([_compute_dhash_bits(s[0]) for s in test]).astype(np.float32)
    min_dh = _min_hamming(te_dh, tr_dh)

    # Null: independent random bits with the train per-bit marginals (and uniform bits), same sizes.
    rng = np.random.default_rng(0)
    marg = tr_dh.mean(axis=0)
    null_marg = (rng.random(te_dh.shape) < marg).astype(np.float32)
    null_unif = (rng.random(te_dh.shape) < 0.5).astype(np.float32)
    null_min_marg = _min_hamming(null_marg, tr_dh)
    null_min_unif = _min_hamming(null_unif, tr_dh)

    # Centred embeddings remove the shared positive offset that saturated raw cosine.
    mu = tr_emb.mean(axis=0, keepdims=True)
    trc = tr_emb - mu
    trc /= np.linalg.norm(trc, axis=1, keepdims=True) + 1e-8
    tec = te_emb - mu
    tec /= np.linalg.norm(tec, axis=1, keepdims=True) + 1e-8
    max_cos_c = (tec @ trc.T).max(axis=1)

    y = np.array([1 if s[2].malignancy_group == "malignant" else 0 for s in test], dtype=np.int64)
    mods = np.array([s[2].modality for s in test])
    all_mask = np.ones(len(test), dtype=bool)

    def quant(a: np.ndarray) -> dict[str, float]:
        return {f"p{int(q * 100):02d}": round(float(np.quantile(a, q)), 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}

    def ladder(base: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {"all": _subset(y, te_p, base)}
        for k in DHASH_LADDER:
            near = base & (min_dh <= k)
            out[f"dhash_le_{k}"] = {"near": _subset(y, te_p, near), "far": _subset(y, te_p, base & ~near)}
        for frac in EMB_NEAR_TOP_FRACTIONS:
            thr = float(np.quantile(max_cos_c[base], 1.0 - frac))
            near = base & (max_cos_c >= thr)
            out[f"emb_top_{int(frac * 100)}pct"] = {
                "centred_cos_threshold": round(thr, 4),
                "near": _subset(y, te_p, near),
                "far": _subset(y, te_p, base & ~near),
            }
        return out

    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "model": "models/derm_cnn_seed1",
        "n_train_labelled": len(train),
        "n_test_v2_labelled": len(test),
        "min_per_class_for_auroc": MIN_PER_CLASS,
        "dhash_min_distance_quantiles_test_vs_train": quant(min_dh),
        "dhash_null_min_distance_quantiles": {
            "train_bit_marginals": quant(null_min_marg),
            "uniform_bits": quant(null_min_unif),
        },
        "frac_test_with_dhash_le": {
            str(k): round(float((min_dh <= k).mean()), 4) for k in DHASH_LADDER
        },
        "frac_null_marginals_with_dhash_le": {
            str(k): round(float((null_min_marg <= k).mean()), 4) for k in DHASH_LADDER
        },
        "centred_max_cos_quantiles": quant(max_cos_c),
        "overall": ladder(all_mask),
        "by_modality": {m: ladder(mods == m) for m in ("dermoscopic", "clinical", "tbp_tile")},
    }
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
