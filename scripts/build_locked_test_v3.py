"""Step 3: Build lesion-grouped, tone x mg and modality stratified `locked_test_v3` and `val_v3` splits."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

from medresearch_platform.vision.catalog import DermCatalogRecord, iter_derm_catalog
from medresearch_platform.vision.splits import log_locked_test_access
from medresearch_platform.vision.sync import compute_split_counts

TONES = ("I", "II", "III", "IV", "V", "VI")
MODALITIES = ("dermoscopic", "clinical", "tbp_tile")


def _load_cached_grouped_records(
    meta_dir: str | Path = "var/cache/derm/meta",
    thumbs_dir: str | Path = "var/cache/derm/thumbs",
    groups_path: str | Path = "configs/vision/lesion_groups_v3.json",
) -> list[DermCatalogRecord]:
    base = Path(thumbs_dir)
    existing = {
        str(p.relative_to(base)).replace("\\", "/")
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in (".webp", ".jpg", ".jpeg", ".png")
    }
    lg_raw = json.loads(Path(groups_path).read_text(encoding="utf-8"))
    image_to_group: dict[str, str] = dict(lg_raw.get("image_to_group", lg_raw))

    recs: list[DermCatalogRecord] = []
    for rec in iter_derm_catalog(meta_dir, require_quality_ok=False, require_research_licence=True):
        is_allowed_fitz = rec.dataset == "fitzpatrick17k" and rec.malignancy_group in ("benign", "malignant")
        if not rec.quality_ok and not is_allowed_fitz:
            continue
        if not rec.thumb_path or str(rec.thumb_path).lstrip("/") not in existing:
            continue
        if rec.image_id in image_to_group:
            rec = dc_replace(rec, group_key=image_to_group[rec.image_id])
        recs.append(rec)
    return recs


def build_v3_splits(
    *,
    seed: int = 42,
    out_test_txt: str | Path = "configs/vision/locked_test_v3.txt",
    out_val_txt: str | Path = "configs/vision/val_v3.txt",
    out_card_md: str | Path = "docs/data/DERM_CORPUS_CARD_V3.md",
    out_summary_json: str | Path = "docs/results/vision/v3_split_summary.json",
    log_test_set: bool = True,
) -> dict[str, Any]:
    rng = random.Random(seed)
    recs = _load_cached_grouped_records()

    groups: dict[str, list[DermCatalogRecord]] = defaultdict(list)
    for r in recs:
        groups[r.group_key].append(r)

    # Order groups deterministically, preferring smaller groups first so giant components stay in train
    all_gkeys = sorted(groups.keys())
    rng.shuffle(all_gkeys)
    all_gkeys.sort(key=lambda gk: len(groups[gk]))

    assigned: dict[str, str] = {}  # group_key -> split

    def _fill_quota(
        split_name: str,
        predicate: Any,
        target_count: int,
        *,
        max_group_size: int = 25,
    ) -> int:
        cur = sum(1 for r in recs if assigned.get(r.group_key) == split_name and predicate(r))
        if cur >= target_count:
            return cur
        for gk in all_gkeys:
            if cur >= target_count:
                break
            if gk in assigned:
                continue
            g_recs = groups[gk]
            if len(g_recs) > max_group_size:
                continue
            hits = sum(1 for r in g_recs if predicate(r))
            if hits == 0:
                continue
            assigned[gk] = split_name
            cur += hits
        return cur

    # 1. Fill locked_test_v3 FIRST: tone x mg cells (aim >= 75 ben & >= 75 mal where corpus allows; leave ~30% for train/val when scarce)
    for tone in TONES:
        for mg in ("malignant", "benign"):
            avail = sum(1 for r in recs if r.skin_tone == tone and r.malignancy_group == mg)
            target = min(75, max(1, int(round(avail * 0.65)))) if avail < 110 else 75
            _fill_quota(
                "locked_test_v3",
                lambda r, t=tone, m=mg: r.skin_tone == t and r.malignancy_group == m,
                target,
            )

    # Ensure >= 300 per modality in locked_test_v3 with both classes present (cap tbp_tile malignant at 100 so >= 194 stay for train/val)
    _fill_quota("locked_test_v3", lambda r: r.modality == "tbp_tile" and r.malignancy_group == "malignant", 100)
    _fill_quota("locked_test_v3", lambda r: r.modality == "tbp_tile" and r.malignancy_group == "benign", 250)
    _fill_quota("locked_test_v3", lambda r: r.modality == "clinical" and r.malignancy_group == "malignant", 180)
    _fill_quota("locked_test_v3", lambda r: r.modality == "clinical" and r.malignancy_group == "benign", 180)
    _fill_quota("locked_test_v3", lambda r: r.modality == "dermoscopic" and r.malignancy_group == "malignant", 300)
    _fill_quota("locked_test_v3", lambda r: r.modality == "dermoscopic" and r.malignancy_group == "benign", 400)

    # 2. Fill val_v3 SECOND from remaining groups (keep tone & modality coverage)
    for tone in TONES:
        for mg in ("malignant", "benign"):
            rem = sum(1 for r in recs if r.group_key not in assigned and r.skin_tone == tone and r.malignancy_group == mg)
            target = min(50, max(1, int(rem * 0.35))) if rem > 2 else 0
            if target > 0:
                _fill_quota(
                    "val",
                    lambda r, t=tone, m=mg: r.skin_tone == t and r.malignancy_group == m,
                    target,
                )
    _fill_quota("val", lambda r: r.modality == "tbp_tile" and r.malignancy_group == "malignant", 45)
    _fill_quota("val", lambda r: r.modality == "tbp_tile" and r.malignancy_group == "benign", 250)
    _fill_quota("val", lambda r: r.modality == "clinical" and r.malignancy_group == "malignant", 150)
    _fill_quota("val", lambda r: r.modality == "clinical" and r.malignancy_group == "benign", 150)
    _fill_quota("val", lambda r: r.modality == "dermoscopic" and r.malignancy_group == "malignant", 350)
    _fill_quota("val", lambda r: r.modality == "dermoscopic" and r.malignancy_group == "benign", 500)

    # 3. Remaining groups -> train
    split_records: dict[str, list[DermCatalogRecord]] = {"train": [], "val": [], "locked_test_v3": []}
    for r in recs:
        sp = assigned.get(r.group_key, "train")
        split_records[sp].append(r)

    # Assert 0 group overlap across splits
    g_sets = {sp: {r.group_key for r in rs} for sp, rs in split_records.items()}
    assert not (g_sets["train"] & g_sets["val"])
    assert not (g_sets["train"] & g_sets["locked_test_v3"])
    assert not (g_sets["val"] & g_sets["locked_test_v3"])

    # Freeze locked_test_v3.txt & val_v3.txt
    test_ids = sorted({r.image_id for r in split_records["locked_test_v3"]})
    val_ids = sorted({r.image_id for r in split_records["val"]})
    tp = Path(out_test_txt)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    vp = Path(out_val_txt)
    vp.write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    test_sha = hashlib.sha256(tp.read_bytes()).hexdigest()

    if log_test_set:
        log_locked_test_access(
            dataset_name="derm-corpus:locked_test_v3",
            model_name=f"frozen_list_sha256:{test_sha[:16]}",
            sample_count=len(test_ids),
            reason=f"Frozen configs/vision/locked_test_v3.txt (sha256={test_sha}; supersedes locked_test_v2 due to W1)",
        )

    counts = compute_split_counts(split_records)

    # Tone x mg cells per split
    tone_mg: dict[str, dict[str, dict[str, int]]] = {}
    mod_mg: dict[str, dict[str, dict[str, int]]] = {}
    for sp, rs in split_records.items():
        tone_mg[sp] = {}
        for t in TONES:
            c = Counter(r.malignancy_group for r in rs if r.skin_tone == t)
            tone_mg[sp][t] = {"benign": c["benign"], "malignant": c["malignant"]}
        mod_mg[sp] = {}
        for m in MODALITIES:
            c = Counter(r.malignancy_group for r in rs if r.modality == m)
            mod_mg[sp][m] = {"benign": c["benign"], "malignant": c["malignant"]}

    # Write DERM_CORPUS_CARD_V3.md
    lines = [
        "# Dermatology Corpus Data Card (v3 — Lesion-Similarity Grouped & Tone×Malignancy Stratified)",
        "",
        f"- **Frozen test list:** `configs/vision/locked_test_v3.txt` (`n={len(test_ids)}`, `sha256={test_sha}`)",
        f"- **Val list:** `configs/vision/val_v3.txt` (`n={len(val_ids)}`)",
        "- **Cross-split lesion group overlap:** `0` (`configs/vision/lesion_groups_v3.json`)",
        "- **Superseded notice:** `locked_test_v2` is marked superseded because images of the same lesion could cross splits (W1); v2 numbers are retained for historical comparison only.",
        "",
        "## 1. Split Summary (Canonical `compute_split_counts`)",
        "",
        "| Split | Total | Benign | Malignant | Masked (`mg`) | Clinical | Dermoscopic | TBP Tile |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sp in ("train", "val", "locked_test_v3"):
        c = counts[sp]
        lines.append(
            f"| `{sp}` | {c['total']} | {c['benign']} | {c['malignant']} | {c['masked_mg']} | "
            f"{c['clinical']} | {c['dermoscopic']} | {c['tbp_tile']} |"
        )
    lines.extend([
        "",
        "## 2. `locked_test_v3` Tone × Malignancy & Modality × Malignancy",
        "",
        "| Tone | Benign (`locked_test_v3`) | Malignant (`locked_test_v3`) | Note |",
        "|---|---|---|---|",
    ])
    for t in TONES:
        b = tone_mg["locked_test_v3"][t]["benign"]
        m = tone_mg["locked_test_v3"][t]["malignant"]
        note = "adequate (>=75 each)" if (b >= 75 and m >= 75) else "corpus-limited cell (<75, never padded)"
        lines.append(f"| `{t}` | {b} | {m} | {note} |")

    lines.extend([
        "",
        "| Modality | Benign (`locked_test_v3`) | Malignant (`locked_test_v3`) | Total |",
        "|---|---|---|---|",
    ])
    for m in MODALITIES:
        b = mod_mg["locked_test_v3"][m]["benign"]
        mal = mod_mg["locked_test_v3"][m]["malignant"]
        lines.append(f"| `{m}` | {b} | {mal} | {b + mal} |")

    Path(out_card_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "locked_test_v3_sha256": test_sha,
        "counts": counts,
        "tone_x_mg": tone_mg,
        "modality_x_mg": mod_mg,
    }
    Path(out_summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_summary_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build_v3_splits(), indent=2))
