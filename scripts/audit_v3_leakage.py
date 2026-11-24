"""Lesion-grouped v3 Vision Leakage Audit: IDs, lesion groups, exact SHA-256 bytes, and 64-bit dHash."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medresearch_platform.vision.catalog import _infer_modality_from_filename, _open_json_auto


def _compute_file_sha256_and_dhash(img_path: Path) -> tuple[str, np.ndarray]:
    raw = img_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    with Image.open(img_path) as im:
        gray = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        arr = np.asarray(gray, dtype=np.int16)
    diff = (arr[:, 1:] > arr[:, :-1]).reshape(-1).astype(np.uint8)
    return sha, diff


def _resolve_path(row: dict[str, Any], thumbs_dir: Path, raw_dir: Path | None) -> Path | None:
    if raw_dir is not None and row.get("f"):
        cand = raw_dir / str(row["f"])
        if cand.exists():
            return cand
    if row.get("t"):
        cand = thumbs_dir / str(row["t"])
        if cand.exists():
            return cand
    return None


def run_v3_leakage_audit(
    *,
    meta_dir: Path = Path("var/cache/derm/meta"),
    thumbs_dir: Path = Path("var/cache/derm/thumbs"),
    raw_dir: Path = Path("var/cache/derm/raw"),
    out_json: Path = Path("docs/results/vision/v3_leakage_audit.json"),
) -> dict[str, Any]:
    t0 = time.perf_counter()
    val_ids = {x.strip() for x in Path("configs/vision/val_v3.txt").read_text(encoding="utf-8").splitlines() if x.strip()}
    test_ids = {x.strip() for x in Path("configs/vision/locked_test_v3.txt").read_text(encoding="utf-8").splitlines() if x.strip()}

    lesion_groups_path = Path("configs/vision/lesion_groups_v3.json")
    id_to_group: dict[str, str] = {}
    if lesion_groups_path.exists():
        lg = json.loads(lesion_groups_path.read_text(encoding="utf-8"))
        raw_map = lg.get("id_to_group", lg) if isinstance(lg, dict) else {}
        if isinstance(raw_map, dict):
            id_to_group = {str(k): str(v) for k, v in raw_map.items()}

    heldout_groups = {id_to_group[i] for i in (val_ids | test_ids) if i in id_to_group}
    raw_root = raw_dir if raw_dir.exists() else None

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for p in sorted(meta_dir.glob("manifest-*.json*")):
        mod_hint = _infer_modality_from_filename(p)
        payload = _open_json_auto(p)
        items = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            continue
        for row in items:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("i", ""))
            if not rid:
                continue
            img_path = _resolve_path(row, thumbs_dir, raw_root)
            if img_path is None:
                continue
            grp = str(row.get("g") or id_to_group.get(rid) or rid)
            rec = {
                "id": rid,
                "group": grp,
                "modality": str(row.get("mod") or mod_hint),
                "malignancy_group": str(row.get("mg") or "unknown").lower(),
                "path": img_path,
            }
            if rid in val_ids:
                splits["val"].append(rec)
            elif rid in test_ids:
                splits["test"].append(rec)
            else:
                if grp in heldout_groups:
                    continue
                splits["train"].append(rec)

    tr_ids = {r["id"] for r in splits["train"]}
    va_ids = {r["id"] for r in splits["val"]}
    te_ids = {r["id"] for r in splits["test"]}

    tr_groups = {r["group"] for r in splits["train"]}
    va_groups = {r["group"] for r in splits["val"]}
    te_groups = {r["group"] for r in splits["test"]}

    tr_shas: set[str] = set()
    tr_dhashes: list[np.ndarray] = []
    for r in splits["train"]:
        sha, dh = _compute_file_sha256_and_dhash(r["path"])
        tr_shas.add(sha)
        tr_dhashes.append(dh)

    te_shas: list[str] = []
    te_dhashes: list[np.ndarray] = []
    te_mods: list[str] = []
    for r in splits["test"]:
        sha, dh = _compute_file_sha256_and_dhash(r["path"])
        te_shas.append(sha)
        te_dhashes.append(dh)
        te_mods.append(r["modality"])

    tr_mat = np.stack(tr_dhashes, axis=0).astype(np.float32)
    te_mat = np.stack(te_dhashes, axis=0).astype(np.float32)

    # Vectorized Hamming distance: sum(a) + sum(b) - 2 * (a @ b.T)
    tr_sum = tr_mat.sum(axis=1)[None, :]
    te_sum = te_mat.sum(axis=1)[:, None]
    hamming = np.maximum(0.0, te_sum + tr_sum - 2.0 * (te_mat @ tr_mat.T))
    min_dhash = hamming.min(axis=1)

    exact_sha_overlap = sum(1 for s in te_shas if s in tr_shas)
    te_mods_arr = np.array(te_mods)

    def _dhash_summary(vals: np.ndarray) -> dict[str, Any]:
        if len(vals) == 0:
            return {"n": 0}
        return {
            "n": int(len(vals)),
            "exact_dhash_0": int(np.sum(vals <= 0.5)),
            "near_dup_dhash_le_4": int(np.sum(vals <= 4.0)),
            "near_dup_dhash_le_4_rate": round(float(np.mean(vals <= 4.0)), 4),
            "close_dhash_le_8": int(np.sum(vals <= 8.0)),
            "close_dhash_le_10": int(np.sum(vals <= 10.0)),
            "quantiles": {
                q: round(float(np.quantile(vals, p)), 2)
                for q, p in [("p05", 0.05), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75)]
            },
        }

    by_modality = {
        mod: _dhash_summary(min_dhash[te_mods_arr == mod])
        for mod in ("dermoscopic", "clinical", "tbp_tile")
    }

    report: dict[str, Any] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "splits": {
            "train_v3_images": len(splits["train"]),
            "val_v3_images": len(splits["val"]),
            "locked_test_v3_images": len(splits["test"]),
        },
        "overlap_checks": {
            "id_overlap_train_val": len(tr_ids & va_ids),
            "id_overlap_train_test": len(tr_ids & te_ids),
            "id_overlap_val_test": len(va_ids & te_ids),
            "group_overlap_train_val": len(tr_groups & va_groups),
            "group_overlap_train_test": len(tr_groups & te_groups),
            "group_overlap_val_test": len(va_groups & te_groups),
            "exact_sha256_overlap_train_test": int(exact_sha_overlap),
        },
        "dhash_train_vs_locked_test_v3": {
            "overall": _dhash_summary(min_dhash),
            "by_modality": by_modality,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_v3_leakage_audit(), indent=2))
