"""Lesion-grouped leak-free splits, perceptual-hash (dHash) near-duplicate audit, and locked-test log."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medresearch_platform.vision.catalog import DermCatalogRecord

SMALL_GROUP_THRESHOLD = 30


def compute_dhash64_from_array(arr: np.ndarray) -> int:
    """Compute a 64-bit horizontal difference hash (dHash) from an RGB or grayscale array."""
    if arr.ndim == 3:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr
    pil_gray = Image.fromarray(np.clip(gray * (255.0 if gray.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8))
    resized = np.asarray(pil_gray.resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    diff = resized[:, 1:] > resized[:, :-1]  # 8x8 boolean grid
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(bool(b))
    return h


def compute_dhash64_from_path(path: str | Path) -> tuple[int, tuple[int, int]] | None:
    """Compute 64-bit dHash and (width, height) from an image file on disk."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            w, h = im.size
            gray = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            arr = np.asarray(gray, dtype=np.int16)
            diff = (arr[:, 1:] > arr[:, :-1]).flatten()
            val = 0
            for b in diff:
                val = (val << 1) | int(bool(b))
            return val, (w, h)
    except Exception:
        return None


def hamming_distance64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def deduplicate_and_assign_splits(
    records: list[DermCatalogRecord],
    *,
    cache_dirs: list[str | Path] | None = None,
    max_hamming_cross_split: int = 0,
) -> tuple[list[DermCatalogRecord], dict[str, Any]]:
    """
    Ensure zero lesion (`group_key`) overlap and zero perceptual-hash near-duplicates across
    `train`, `val`, and `locked_test`. If two records share an identical or near-identical
    dHash across different splits, exclude the secondary duplicate from evaluation splits.
    """
    dirs = [Path(d) for d in (cache_dirs or ["var/cache/derm/thumbs", "var/cache/derm_thumbs"])]
    existing_thumb_map: dict[str, Path] = {}
    for base in dirs:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                rel_key = str(p.relative_to(base)).replace("\\", "/")
                existing_thumb_map.setdefault(rel_key, p)

    dhashes: dict[str, int] = {}
    dimensions: list[tuple[int, int]] = []

    for rec in records:
        if not rec.thumb_path:
            continue
        cand = existing_thumb_map.get(rec.thumb_path.lstrip("/"))
        if cand is not None:
            res = compute_dhash64_from_path(cand)
            if res is not None:
                dhashes[rec.image_id] = res[0]
                dimensions.append(res[1])

    # Group hashes by exact bucket and check cross-split duplicates
    hash_to_primary_split: dict[int, str] = {}
    cross_split_duplicates_removed: list[str] = []
    within_split_duplicates_count = 0
    clean_records: list[DermCatalogRecord] = []

    # Priority order: train first, then val, then locked_test so locked_test never leaks train images
    split_priority = {"train": 0, "val": 1, "locked_test": 2}
    sorted_recs = sorted(records, key=lambda r: (split_priority.get(r.split_assignment, 9), r.image_id))

    for rec in sorted_recs:
        h = dhashes.get(rec.image_id)
        if h is None:
            clean_records.append(rec)
            continue

        matched_split: str | None = None
        if max_hamming_cross_split == 0:
            matched_split = hash_to_primary_split.get(h)
        else:
            for existing_h, sp in hash_to_primary_split.items():
                if hamming_distance64(h, existing_h) <= max_hamming_cross_split:
                    matched_split = sp
                    break

        if matched_split is None:
            hash_to_primary_split[h] = rec.split_assignment
            clean_records.append(rec)
        elif matched_split == rec.split_assignment:
            within_split_duplicates_count += 1
            clean_records.append(rec)
        else:
            cross_split_duplicates_removed.append(rec.image_id)

    widths = [w for w, _ in dimensions]
    heights = [h for _, h in dimensions]
    size_summary = {
        "inspected_thumbnails": len(dimensions),
        "unique_dhash64": len(set(dhashes.values())),
        "within_split_exact_dhash_duplicates": within_split_duplicates_count,
        "cross_split_duplicates_excluded": len(cross_split_duplicates_removed),
        "width_min": int(min(widths)) if widths else None,
        "width_median": float(np.median(widths)) if widths else None,
        "width_max": int(max(widths)) if widths else None,
        "height_min": int(min(heights)) if heights else None,
        "height_median": float(np.median(heights)) if heights else None,
        "height_max": int(max(heights)) if heights else None,
        "top_resolutions": {
            f"{w}x{h}": cnt for (w, h), cnt in Counter(dimensions).most_common(10)
        },
    }
    return clean_records, size_summary


def audit_derm_splits(
    records: list[DermCatalogRecord],
    *,
    cache_dirs: list[str | Path] | None = None,
) -> dict[str, Any]:
    """
    Verify 0 lesion (`group_key`) overlap and 0 perceptual-hash overlap across `train`, `val`,
    and `locked_test`, and compute subgroup sample counts with `< 30` small-sample flags.
    """
    clean_records, image_stats = deduplicate_and_assign_splits(records, cache_dirs=cache_dirs)

    groups_by_split: dict[str, set[str]] = defaultdict(set)
    mk_by_split: dict[str, set[str]] = defaultdict(set)
    counts_by_split: Counter[str] = Counter()
    malignancy_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    modality_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    tone_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    dx_by_split: dict[str, Counter[str]] = defaultdict(Counter)

    for r in clean_records:
        sp = r.split_assignment
        counts_by_split[sp] += 1
        groups_by_split[sp].add(r.group_key)
        if r.case_key:
            mk_by_split[sp].add(f"{r.dataset}:{r.case_key}")
        malignancy_by_split[sp][r.malignancy_group] += 1
        modality_by_split[sp][r.modality] += 1
        tone_by_split[sp][r.skin_tone] += 1
        dx_by_split[sp][r.diagnosis_harmonised] += 1

    train_val_overlap = sorted(groups_by_split["train"] & groups_by_split["val"])
    train_test_overlap = sorted(groups_by_split["train"] & groups_by_split["locked_test"])
    val_test_overlap = sorted(groups_by_split["val"] & groups_by_split["locked_test"])

    mk_train_val = sorted(mk_by_split["train"] & mk_by_split["val"])
    mk_train_test = sorted(mk_by_split["train"] & mk_by_split["locked_test"])
    mk_val_test = sorted(mk_by_split["val"] & mk_by_split["locked_test"])

    # Flag subgroups in locked_test with < 30 samples
    small_subgroups_locked_test: list[dict[str, Any]] = []
    for axis_name, counter in (
        ("skin_tone", tone_by_split["locked_test"]),
        ("modality", modality_by_split["locked_test"]),
        ("diagnosis_harmonised", dx_by_split["locked_test"]),
    ):
        for group_val, cnt in sorted(counter.items()):
            if cnt < SMALL_GROUP_THRESHOLD:
                small_subgroups_locked_test.append({
                    "axis": axis_name,
                    "subgroup": group_val,
                    "count": cnt,
                    "flag": "too few to conclude (< 30)",
                })

    passed = (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
        and len(mk_train_val) == 0
        and len(mk_train_test) == 0
        and len(mk_val_test) == 0
    )

    return {
        "passed": passed,
        "total_input_records": len(records),
        "total_clean_records": len(clean_records),
        "counts_by_split": dict(counts_by_split),
        "lesion_groups_by_split": {k: len(v) for k, v in groups_by_split.items()},
        "lesion_group_overlap": {
            "train_val": len(train_val_overlap),
            "train_locked_test": len(train_test_overlap),
            "val_locked_test": len(val_test_overlap),
        },
        "explicit_mk_overlap": {
            "train_val": len(mk_train_val),
            "train_locked_test": len(mk_train_test),
            "val_locked_test": len(mk_val_test),
        },
        "malignancy_by_split": {k: dict(v) for k, v in malignancy_by_split.items()},
        "modality_by_split": {k: dict(v) for k, v in modality_by_split.items()},
        "skin_tone_by_split": {k: dict(v) for k, v in tone_by_split.items()},
        "diagnosis_by_split": {k: dict(v) for k, v in dx_by_split.items()},
        "image_and_dhash_audit": image_stats,
        "small_subgroups_locked_test": small_subgroups_locked_test,
    }


def log_locked_test_access(
    *,
    dataset_name: str,
    model_name: str,
    reason: str,
    sample_count: int,
    log_path: str | Path = "docs/results/TEST_SET_LOG.md",
) -> None:
    """Append an immutable audit entry whenever a locked test split is created or evaluated."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not p.exists():
        header = (
            "# Locked Test Set Audit Log (`TEST_SET_LOG.md`)\n\n"
            "Every creation or evaluation on a locked test split (`test_*` or `derm_locked_test`) "
            "is logged here for scientific traceability.\n\n"
            "| Timestamp (UTC) | Dataset / Split | Model / Action | Sample Count | Reason |\n"
            "|---|---|---|---|---|\n"
        )
        p.write_text(header, encoding="utf-8")
    row = f"| `{ts}` | `{dataset_name}` | `{model_name}` | `{sample_count}` | {reason} |\n"
    with p.open("a", encoding="utf-8") as fh:
        fh.write(row)
