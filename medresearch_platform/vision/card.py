"""Generator for `docs/data/DERM_CORPUS_CARD.md` and leak-free split audit reports."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medresearch_platform.vision.catalog import (
    DermCatalogRecord,
    _deterministic_split,
    iter_derm_catalog,
    summarize_derm_catalog,
)
from medresearch_platform.vision.splits import (
    audit_derm_splits,
    compute_dhash64_from_path,
    log_locked_test_access,
)
from medresearch_platform.vision.taxonomy import (
    DEFAULT_TAXONOMY_PATH,
    load_dx_taxonomy,
    summarize_taxonomy_mapping,
)


def inspect_local_thumbnail_cache(
    cache_dirs: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Inspect all locally cached thumbnails on disk (dimensions, file sizes, datasets, dHash duplicates)."""
    dirs = [Path(d) for d in (cache_dirs or ["var/cache/derm/thumbs", "var/cache/derm_thumbs"])]
    seen_rel: set[str] = set()
    files: list[Path] = []
    dataset_counts: Counter[str] = Counter()

    for base in dirs:
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in (".webp", ".jpg", ".jpeg", ".png"):
                continue
            rel = str(p.relative_to(base))
            if rel in seen_rel:
                continue
            seen_rel.add(rel)
            files.append(p)
            if "dataset=" in rel:
                ds_part = rel.split("dataset=", 1)[1].split("/", 1)[0]
                dataset_counts[ds_part] += 1
            else:
                dataset_counts["local_cache"] += 1

    widths: list[int] = []
    heights: list[int] = []
    byte_sizes: list[int] = []
    resolutions: Counter[str] = Counter()
    dhashes: list[int] = []
    split_counts: Counter[str] = Counter()

    for p in files:
        byte_sizes.append(p.stat().st_size)
        stem = p.stem
        group_key = f"isic:{stem[:12]}"
        split_counts[_deterministic_split(group_key)] += 1
        res = compute_dhash64_from_path(p)
        if res is not None:
            h_val, (w, h) = res
            dhashes.append(h_val)
            widths.append(w)
            heights.append(h)
            resolutions[f"{w}x{h}"] += 1
        else:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    widths.append(w)
                    heights.append(h)
                    resolutions[f"{w}x{h}"] += 1
            except Exception:
                pass

    unique_hashes = len(set(dhashes))
    exact_dhash_dups = len(dhashes) - unique_hashes

    return {
        "total_cached_thumbnails": len(files),
        "datasets": dict(dataset_counts),
        "total_bytes": int(sum(byte_sizes)),
        "mean_bytes": round(float(np.mean(byte_sizes)), 1) if byte_sizes else 0.0,
        "width_min": int(min(widths)) if widths else None,
        "width_median": float(np.median(widths)) if widths else None,
        "width_max": int(max(widths)) if widths else None,
        "height_min": int(min(heights)) if heights else None,
        "height_median": float(np.median(heights)) if heights else None,
        "height_max": int(max(heights)) if heights else None,
        "top_resolutions": dict(resolutions.most_common(10)),
        "dhash64_evaluated": len(dhashes),
        "dhash64_unique": unique_hashes,
        "dhash64_exact_duplicates": exact_dhash_dups,
        "deterministic_split_counts": dict(split_counts),
    }


def _format_kv_table(mapping: dict[str, Any], col1: str, col2: str = "Count", total: int | None = None) -> str:
    if not mapping:
        return f"| {col1} | {col2} |\n|---|---|\n| *(none)* | `0` |\n"
    lines = [
        f"| {col1} | {col2} | Share | Flag |" if total else f"| {col1} | {col2} |",
        "|---|---|---|---|" if total else "|---|---|",
    ]
    for k, v in mapping.items():
        if total and total > 0 and isinstance(v, int):
            pct = (v / total) * 100.0
            flag = "too few to conclude (`< 30`)" if v < 30 else "adequate (`>= 30`)"
            lines.append(f"| `{k}` | `{v}` | `{pct:.2f}%` | {flag} |")
        else:
            lines.append(f"| `{k}` | `{v}` |")
    return "\n".join(lines) + "\n"


def generate_derm_corpus_card(
    meta_dir: str | Path = "var/cache/derm/meta",
    cache_dirs: list[str | Path] | None = None,
    *,
    out_md_path: str | Path = "docs/data/DERM_CORPUS_CARD.md",
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    log_test_set: bool = True,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Analyze the dermatology metadata catalog and local thumbnail cache, run the leak-free
    split and perceptual-hash audit, and write `docs/data/DERM_CORPUS_CARD.md`.
    """
    meta_path = Path(meta_dir)
    if not meta_path.exists() or not list(meta_path.glob("*.json*")):
        legacy_meta = Path("/tmp/derm_meta")
        if legacy_meta.exists() and list(legacy_meta.glob("*.json*")):
            meta_path = legacy_meta

    records: list[DermCatalogRecord] = list(iter_derm_catalog(meta_path))
    summary = summarize_derm_catalog(meta_path, records=records)
    thumb_cache_stats = inspect_local_thumbnail_cache(cache_dirs)
    tax_cfg = load_dx_taxonomy(taxonomy_path)

    raw_dx_pairs = [(r.diagnosis_raw, r.malignancy_group) for r in records]
    tax_summary = summarize_taxonomy_mapping(raw_dx_pairs, taxonomy_path=taxonomy_path)
    split_audit = audit_derm_splits(records, cache_dirs=cache_dirs)

    if log_test_set and split_audit["counts_by_split"].get("locked_test", 0) > 0:
        log_locked_test_access(
            dataset_name="derm-corpus:locked_test",
            model_name="step1_split_creation_and_audit",
            reason="Initial deterministic lesion-grouped split audit (Step 1)",
            sample_count=int(split_audit["counts_by_split"]["locked_test"]),
            log_path=log_path,
        )

    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total = summary["total_records"]
    metadata_synced = total > 0

    classes_cfg = tax_cfg.get("classes", {})
    tax_rows = []
    for cls_name, cls_info in classes_cfg.items():
        cnt = tax_summary["harmonised_class_counts"].get(cls_name, 0)
        aliases_preview = ", ".join(f"`{a}`" for a in cls_info.get("aliases", [])[:6])
        tax_rows.append(
            f"| `{cls_name}` | `{cls_info.get('malignancy_group')}` | `{cls_info.get('icd10')}` "
            f"| {cls_info.get('label_en')} / {cls_info.get('label_fr')} | `{cnt}` | {aliases_preview} |"
        )
    tax_table = (
        "| Harmonised Class | Malignancy Group | ICD-10 | Label (EN / FR) | Metadata Count | Representative Raw `dx` Aliases |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(tax_rows)
        + "\n"
    )

    status_banner = (
        f"**Metadata Status:** Synced (`{total}` records across `{summary['unique_lesion_groups']}` unique lesion groups)"
        if metadata_synced
        else (
            "**Metadata Status:** Pending remote CloudFront sync (`.env` with `DERM_CLOUDFRONT_USER` / "
            "`DERM_CLOUDFRONT_PASS` not yet present in repository root; `/tmp/derm_meta` was cleared on macOS reboot). "
            f"Local thumbnail cache (`var/cache/derm_thumbs/`) verified on disk with **`{thumb_cache_stats['total_cached_thumbnails']}` real ISIC `.webp` thumbnails**."
        )
    )

    md = f"""# Dermatology Corpus Data Card (`DERM_CORPUS_CARD.md`)

- **Generated (UTC):** `{ts_utc}`
- **Source Endpoint:** `https://d38rablk9v12ns.cloudfront.net/` (HTTP Basic Auth realm `derm-corpus`)
- **Local Cache Paths:** `var/cache/derm/meta/` (metadata), `var/cache/derm/thumbs/` & `var/cache/derm_thumbs/` (thumbnails) — **git-ignored**
- **Taxonomy Config:** [configs/vision/dx_taxonomy.yaml](../../configs/vision/dx_taxonomy.yaml) (`{tax_summary['taxonomy_id']}`, minimum class count `{tax_summary['min_class_count']}`)
- {status_banner}

> **Clinical & Scientific Safety Notice (EN / FR)**
> - **EN:** Research tool — not a diagnosis. A dermatologist must examine the lesion.
> - **FR :** Outil de recherche — ne constitue pas un diagnostic. Un dermatologue doit examiner la lésion.

---

## 1. Corpus Overview & Lesion / Patient Grouping (`mk`)

| Metric | Value |
|---|---|
| Total Metadata Records (`i`) | `{summary['total_records']}` |
| Unique Image IDs (`i`) | `{summary['unique_images']}` |
| Duplicate Image IDs in Metadata | `{summary['duplicate_image_id_count']}` |
| Unique Lesion / Case Groups (`dataset:mk` or `group_key`) | `{summary['unique_lesion_groups']}` |
| Records with Explicit `mk` Lesion/Patient Key | `{summary['explicit_mk_records']}` (`{summary['unique_explicit_mk_lesions']}` unique explicit `mk` keys) |
| Quality Flag `ok=True` | `{summary['quality_ok_count']}` |
| Quality Flag `ok=False` (Rejected) | `{summary['quality_rejected_count']}` |
| Diagnosis vs. `mg` Malignancy Inconsistencies | `{summary['mg_inconsistent_count']}` |

---

## 2. Local Thumbnail Cache & Image Dimension / Perceptual-Hash Audit

Measured directly on locally cached `.webp` thumbnails in `var/cache/derm_thumbs/` (`dataset=isic`):

| Property | Measured Value |
|---|---|
| Cached Thumbnails on Disk | `{thumb_cache_stats['total_cached_thumbnails']}` |
| Cached Datasets | `{thumb_cache_stats['datasets']}` |
| Total Disk Footprint | `{thumb_cache_stats['total_bytes']:,} bytes` (mean `{thumb_cache_stats['mean_bytes']:,.1f} bytes`/image) |
| Width Range (min / median / max) | `{thumb_cache_stats['width_min']}` / `{thumb_cache_stats['width_median']}` / `{thumb_cache_stats['width_max']}` px |
| Height Range (min / median / max) | `{thumb_cache_stats['height_min']}` / `{thumb_cache_stats['height_median']}` / `{thumb_cache_stats['height_max']}` px |
| Top Resolutions (`WxH`) | `{thumb_cache_stats['top_resolutions']}` |
| 64-bit Perceptual Hash (`dHash`) Evaluated | `{thumb_cache_stats['dhash64_evaluated']}` |
| Unique 64-bit `dHash` Signatures | `{thumb_cache_stats['dhash64_unique']}` |
| Exact `dHash` Duplicates in Local Cache | `{thumb_cache_stats['dhash64_exact_duplicates']}` |
| Deterministic Split Distribution (Cached Thumbnails) | `{thumb_cache_stats['deterministic_split_counts']}` |

---

## 3. Breakdown by Modality, Source Dataset, Malignancy, Skin Tone, Body Site & Confirmation

### 3.1 By Modality (`clinical`, `dermoscopic`, `tbp_tile`)
{_format_kv_table(summary['modalities'], 'Modality', total=total)}

### 3.2 By Source Dataset (`d`)
{_format_kv_table(summary['datasets'], 'Source Dataset', total=total)}

### 3.3 By Malignancy Group (`mg`)
{_format_kv_table(summary['malignancy_group'], 'Malignancy Group', total=total)}

### 3.4 By Fitzpatrick Skin Tone (`st`: `I`–`VI`, `UNKNOWN`)
{_format_kv_table(summary['skin_tones'], 'Fitzpatrick Skin Tone', total=total)}

### 3.5 By Body Site (`bs`)
{_format_kv_table(summary['body_sites'], 'Body Site', total=total)}

### 3.6 By Confirmation Method (`cf`)
{_format_kv_table(summary['confirmations'], 'Confirmation Method', total=total)}

---

## 4. Harmonised Diagnosis Taxonomy (`configs/vision/dx_taxonomy.yaml`)

Classes with fewer than `{tax_summary['min_class_count']}` records across the full corpus are automatically collapsed into `other`. Every class specifies its canonical `malignancy_group` (`benign` or `malignant`) and is checked against the record's raw `mg` attribute.

{tax_table}

### Top Raw Diagnosis Strings (`dx`)
{_format_kv_table(summary['raw_diagnoses_top50'], 'Raw Diagnosis (`dx`)', total=total)}

---

## 5. Licence Governance & Privacy Rules (`lc`)

| Licence Category | Rule / Permission | Metadata Count |
|---|---|---|
| **Research Allowed** (`CC0`, `CC-BY*`, `CC-BY-NC*`, `SYNTHETIC-INTERNAL`) | Permitted for local model training & evaluation | `{summary['licence_summary']['research_allowed_count']}` |
| **Redistribution Allowed** (`CC0`, `CC-BY`, `CC-BY-SA`) | Permitted for redistribution with attribution | `{summary['licence_summary']['redistribution_allowed_count']}` |
| **Commercial Use Allowed** (`CC0`, `CC-BY`, `CC-BY-SA`) | Permitted for commercial use | `{summary['licence_summary']['commercial_allowed_count']}` |
| **Safe for Docs & Screenshots** (`CC0`, `CC-BY`, `SYNTHETIC-INTERNAL`) | Only these (with attribution) or synthetic images may appear in docs/screenshots | `{summary['licence_summary']['docs_screenshot_safe_count']}` |
| **Excluded / Unknown Licence** | Excluded from training & evaluation (`require_research_licence=True`) | `{summary['licence_summary']['excluded_or_unknown_count']}` |

### Counts per Raw Licence (`lc`)
{_format_kv_table(summary['licence_summary']['by_licence'], 'Raw Licence (`lc`)', total=total)}

**Privacy & Storage Enforcement:**
- Raw images, thumbnails, and metadata dumps are stored strictly inside `var/cache/derm/` (git-ignored) and are **never committed to git**.
- Clinical photographs may contain identifiable facial features or tattoos: full-size images are restricted to authenticated sessions, and list views render thumbnails only.

---

## 6. Leak-Free Lesion Split & Perceptual-Hash Deduplication Audit

- **Split Rule:** Deterministic SHA-256 hash of `group_key = f"{{dataset}}:{{mk}}"` (`70%` `train`, `15%` `val`, `15%` `locked_test`), ensuring all photographs of the same lesion/patient stay in the exact same split.
- **Cross-Split Near-Duplicate Gate:** 64-bit horizontal difference perceptual hash (`dHash`) computed on thumbnails; any secondary image sharing a `dHash` across splits is excluded before evaluation.
- **Audit Status:** **`{"PASSED (0 overlap)" if split_audit["passed"] else "FAILED"}`**
  - `train ∩ val` lesion overlap: `{split_audit['lesion_group_overlap']['train_val']}`
  - `train ∩ locked_test` lesion overlap: `{split_audit['lesion_group_overlap']['train_locked_test']}`
  - `val ∩ locked_test` lesion overlap: `{split_audit['lesion_group_overlap']['val_locked_test']}`
  - Cross-split `dHash` duplicates excluded: `{split_audit['image_and_dhash_audit']['cross_split_duplicates_excluded']}`
"""

    out_path = Path(out_md_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    tax_summary_compact = dict(tax_summary)
    tax_summary_compact["raw_mapping_table_top30"] = tax_summary["raw_mapping_table"][:30]
    tax_summary_compact.pop("raw_mapping_table", None)

    return {
        "generated_at_utc": ts_utc,
        "card_path": str(out_path),
        "metadata_synced": metadata_synced,
        "catalog_summary": summary,
        "local_thumbnail_cache": thumb_cache_stats,
        "taxonomy_summary": tax_summary_compact,
        "split_audit": split_audit,
    }
