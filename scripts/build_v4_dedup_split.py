"""Build SHA-256 deduplicated `locked_test_v4.txt` (0 ID, 0 lesion-group, 0 exact-byte duplicates vs train_v3)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from medresearch_platform.vision.catalog import _infer_modality_from_filename, _open_json_auto


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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


def build_locked_test_v4(
    *,
    meta_dir: Path = Path("var/cache/derm/meta"),
    thumbs_dir: Path = Path("var/cache/derm/thumbs"),
    raw_dir: Path = Path("var/cache/derm/raw"),
    val_v3_path: Path = Path("configs/vision/val_v3.txt"),
    test_v3_path: Path = Path("configs/vision/locked_test_v3.txt"),
    lesion_groups_path: Path = Path("configs/vision/lesion_groups_v3.json"),
    out_test_v4: Path = Path("configs/vision/locked_test_v4.txt"),
    out_audit_json: Path = Path("docs/results/vision/v4_dedup_audit.json"),
) -> dict[str, Any]:
    t0 = time.perf_counter()
    val_ids = {x.strip() for x in val_v3_path.read_text(encoding="utf-8").splitlines() if x.strip()}
    test_ids_ordered = [x.strip() for x in test_v3_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    test_ids = set(test_ids_ordered)

    id_to_group: dict[str, str] = {}
    if lesion_groups_path.exists():
        lg = json.loads(lesion_groups_path.read_text(encoding="utf-8"))
        raw_map = lg.get("id_to_group", lg) if isinstance(lg, dict) else {}
        if isinstance(raw_map, dict):
            id_to_group = {str(k): str(v) for k, v in raw_map.items()}
    heldout_groups = {id_to_group[i] for i in (val_ids | test_ids) if i in id_to_group}
    raw_root = raw_dir if raw_dir.exists() else None

    train_shas: set[str] = set()
    test_id_to_info: dict[str, dict[str, str]] = {}
    train_count = 0

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
            if rid in test_ids:
                test_id_to_info[rid] = {
                    "sha256": _sha256_file(img_path),
                    "modality": str(row.get("mod") or mod_hint),
                    "group": grp,
                }
            elif rid not in val_ids and grp not in heldout_groups:
                train_shas.add(_sha256_file(img_path))
                train_count += 1

    kept_ids: list[str] = []
    removed_duplicates: list[dict[str, str]] = []
    seen_test_shas: set[str] = set()
    for rid in test_ids_ordered:
        info = test_id_to_info.get(rid)
        if info is None:
            continue
        sha = info["sha256"]
        if sha in train_shas or sha in seen_test_shas:
            removed_duplicates.append(
                {
                    "image_id": rid,
                    "modality": info["modality"],
                    "reason": "exact_sha256_in_train_v3" if sha in train_shas else "exact_sha256_dup_within_test",
                }
            )
            continue
        seen_test_shas.add(sha)
        kept_ids.append(rid)

    out_test_v4.parent.mkdir(parents=True, exist_ok=True)
    out_test_v4.write_text("\n".join(kept_ids) + "\n", encoding="utf-8")

    report: dict[str, Any] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "source_split": str(test_v3_path),
        "output_split": str(out_test_v4),
        "train_v3_images_hashed": train_count,
        "locked_test_v3_count": len(test_ids_ordered),
        "locked_test_v4_count": len(kept_ids),
        "removed_exact_sha256_count": len(removed_duplicates),
        "post_dedup_exact_sha256_overlap_train_test": 0,
        "removed_by_modality": {
            mod: sum(1 for r in removed_duplicates if r["modality"] == mod)
            for mod in ("dermoscopic", "clinical", "tbp_tile")
        },
        "removed_ids": [r["image_id"] for r in removed_duplicates],
    }
    out_audit_json.parent.mkdir(parents=True, exist_ok=True)
    out_audit_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_locked_test_v4(), indent=2))
