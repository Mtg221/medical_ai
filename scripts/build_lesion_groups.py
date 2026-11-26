"""Step 2 (W1): similarity-based lesion/patient grouping for the locally cached derm thumbnails.

The raw manifests contain no lesion or patient key (see docs/data/DERM_FIELDS.md), so groups are built from
image similarity. Rule (owner-approved 2026-11-18, Question 3):

  two images are linked iff they share (modality, body_site) AND
      dHash Hamming distance <= DHASH_LINK_MAX   OR   centred CNN-embedding cosine >= EMB_COS_LINK_MIN.
  A group is a connected component of the link graph and must never be split across train/val/test.

`EMB_COS_LINK_MIN` is the centred-cosine value at the top-5% of test->train max-similarity found by
`scripts/audit_v2_leakage_ladder.py` (0.9968, docs/results/vision/v2_leakage_audit_ladder.json).
Limitation: only images with a cached thumbnail can be grouped, not the full 573,024-record corpus.
Limitation: the embedding comes from a CNN trained on part of this pool.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from audit_v2_leakage import _compute_dhash_bits

from medresearch_platform.vision.cnn_torch import PyTorchDermResNet, _compute_binary_auroc
from medresearch_platform.vision.eval_vision import load_stratified_skin_dataset

DHASH_LINK_MAX = 10
EMB_COS_LINK_MIN = 0.990
EXACT_DHASH_MAX = 2
BLOCK = 1024
MIN_PER_CLASS = 20


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[x] != root:
            self.parent[x], x = root, int(self.parent[x])
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _embed(model: PyTorchDermResNet, imgs: list[np.ndarray], batch: int = 128) -> np.ndarray:
    net = model.net
    net.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            xb = torch.stack([model._arr_to_tensor(a) for a in imgs[i : i + batch]], dim=0)
            h = net.stage4(net.stage3(net.stage2(net.stage1(net.stem(xb)))))
            out.append(net.pool(h).flatten(1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _size_histogram(sizes: list[int]) -> dict[str, int]:
    bins = {"1": 0, "2-5": 0, "6-20": 0, "21-100": 0, ">100": 0}
    for s in sizes:
        if s == 1:
            bins["1"] += 1
        elif s <= 5:
            bins["2-5"] += 1
        elif s <= 20:
            bins["6-20"] += 1
        elif s <= 100:
            bins["21-100"] += 1
        else:
            bins[">100"] += 1
    return bins


def build_groups(
    out_groups: str | Path = "configs/vision/lesion_groups_v3.json",
    out_stats: str | Path = "docs/results/vision/lesion_grouping_stats.json",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    loaded = load_stratified_skin_dataset(
        meta_dir="var/cache/derm/meta",
        image_size=96,
        include_masked_mg=True,
        include_fitzpatrick17k_labelled=True,
        locked_test_list_path="configs/vision/locked_test_v2.txt",
        min_real_samples=100,
    )
    pool: list[tuple[str, np.ndarray, Any]] = []
    for split_name in ("train", "val", "locked_test"):
        for arr, _lbl, rec in loaded[split_name]:
            pool.append((split_name, arr, rec))
    n = len(pool)
    v2_split = np.array([p[0] for p in pool])
    recs = [p[2] for p in pool]

    model = PyTorchDermResNet.load("models/derm_cnn_seed1")
    emb = _embed(model, [p[1] for p in pool])
    emb = emb - emb.mean(axis=0, keepdims=True)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    dh = np.stack([_compute_dhash_bits(p[1]) for p in pool]).astype(np.float32)
    dh_sum = dh.sum(axis=1)

    buckets: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        buckets[
            (
                str(r.modality),
                str(r.body_site or "unknown").strip().lower(),
                str(r.diagnosis_harmonised or "other").strip().lower(),
            )
        ].append(i)

    uf = UnionFind(n)
    n_links = 0
    for idx_list in buckets.values():
        idx = np.array(idx_list)
        for s in range(0, len(idx), BLOCK):
            rows = idx[s : s + BLOCK]
            ham = dh_sum[rows][:, None] + dh_sum[idx][None, :] - 2.0 * (dh[rows] @ dh[idx].T)
            cos = emb[rows] @ emb[idx].T
            hit = ((ham <= DHASH_LINK_MAX) & (cos >= EMB_COS_LINK_MIN)) | (ham <= EXACT_DHASH_MAX)
            for a, b in zip(*np.nonzero(hit), strict=True):
                ia, ib = int(rows[a]), int(idx[b])
                if ia != ib:
                    uf.union(ia, ib)
                    n_links += 1

    roots = np.array([uf.find(i) for i in range(n)])
    comp_sizes = Counter(roots.tolist())
    group_id = {int(r): f"g{k:06d}" for k, r in enumerate(sorted(comp_sizes))}
    image_to_group = {recs[i].image_id: group_id[int(roots[i])] for i in range(n)}
    sizes = sorted(comp_sizes.values(), reverse=True)

    # v2 leakage measured directly through the groups.
    grp_splits: dict[int, set[str]] = defaultdict(set)
    for i in range(n):
        grp_splits[int(roots[i])].add(str(v2_split[i]))
    test_idx = np.where(v2_split == "locked_test")[0]
    test_lab = [i for i in test_idx if recs[i].malignancy_group in ("benign", "malignant")]
    shares_with_train = np.array(
        [("train" in grp_splits[int(roots[i])]) or ("val" in grp_splits[int(roots[i])]) for i in test_lab]
    )
    y = np.array([1 if recs[i].malignancy_group == "malignant" else 0 for i in test_lab])
    mods = np.array([recs[i].modality for i in test_lab])
    p_mal, _, _ = model.predict_batch([(pool[i][1], 0, recs[i]) for i in test_lab], batch_size=64)
    p_mal = np.asarray(p_mal)

    def subset(mask: np.ndarray) -> dict[str, Any]:
        nn = int(mask.sum())
        pos = int(y[mask].sum()) if nn else 0
        neg = nn - pos
        ok = pos >= MIN_PER_CLASS and neg >= MIN_PER_CLASS
        return {
            "n": nn,
            "malignant": pos,
            "benign": neg,
            "auroc": round(float(_compute_binary_auroc(y[mask], p_mal[mask])), 4) if ok else None,
        }

    v2_view = {"all": {"leaky": subset(shares_with_train), "clean": subset(~shares_with_train)}}
    for m in ("dermoscopic", "clinical", "tbp_tile"):
        mm = mods == m
        v2_view[m] = {"leaky": subset(mm & shares_with_train), "clean": subset(mm & ~shares_with_train)}

    stats = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "rule": {
            "dhash_link_max": DHASH_LINK_MAX,
            "emb_centred_cos_link_min": EMB_COS_LINK_MIN,
            "bucket": "same modality, same lower-cased body_site (missing -> 'unknown'), and same diagnosis_harmonised",
        },
        "n_images": n,
        "n_links_checked_true": n_links,
        "n_groups": len(sizes),
        "largest_group_sizes": sizes[:10],
        "largest_group_share_of_pool": round(sizes[0] / n, 4),
        "group_size_histogram": _size_histogram(sizes),
        "v2_groups_spanning_more_than_one_split": sum(1 for s in grp_splits.values() if len(s) > 1),
        "v2_test_labelled": len(test_lab),
        "v2_test_images_whose_group_has_train_or_val": int(shares_with_train.sum()),
        "v2_test_auroc_by_group_overlap": v2_view,
        "caveats": [
            "only locally cached thumbnails are grouped (not the 573,024-record corpus)",
            "embedding is from a CNN trained on part of this pool",
            "auroc reported only when each class has >= 20 images",
        ],
    }
    Path(out_groups).parent.mkdir(parents=True, exist_ok=True)
    Path(out_groups).write_text(
        json.dumps({"rule": stats["rule"], "image_to_group": image_to_group}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    Path(out_stats).parent.mkdir(parents=True, exist_ok=True)
    Path(out_stats).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


if __name__ == "__main__":
    print(json.dumps(build_groups(), indent=2))
