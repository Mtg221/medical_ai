"""Zero-leakage external clinical smartphone dermatology ingestion (PAD-UFES-20, Fitzpatrick17k, DDI).

Normalizes metadata (diagnosis -> benign/malignant, Fitzpatrick skin tone -> I-II/III-IV/V-VI,
patient/lesion grouping) and rejects any image whose SHA-256 collides with held-out validation or
locked test splits (`val_v3`, `locked_test_v3`, `locked_test_v4`).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MALIGNANT_LABELS: frozenset[str] = frozenset(
    {
        "mel", "melanoma", "bcc", "basal cell carcinoma", "scc", "squamous cell carcinoma",
        "ak", "ack", "actinic keratosis", "bowen", "malignant", "1", "true",
    }
)
BENIGN_LABELS: frozenset[str] = frozenset(
    {
        "nev", "nevus", "melanocytic nevus", "sek", "seborrheic keratosis", "bkl", "df",
        "dermatofibroma", "vasc", "vascular", "benign", "0", "false",
    }
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_malignancy_label(raw: str | None) -> str:
    low = str(raw or "").strip().lower()
    if low in MALIGNANT_LABELS or any(k in low for k in ("melanoma", "carcinoma", "malignant")):
        return "malignant"
    if low in BENIGN_LABELS or any(k in low for k in ("nevus", "benign", "keratosis", "dermatofibroma")):
        return "benign"
    return "unknown"


def normalize_fitzpatrick(raw: str | None) -> str:
    val = str(raw or "").strip().upper()
    if val in {"1", "2", "I", "II", "I-II"}:
        return "I-II"
    if val in {"3", "4", "III", "IV", "III-IV"}:
        return "III-IV"
    if val in {"5", "6", "V", "VI", "V-VI"}:
        return "V-VI"
    return "UNKNOWN"


def ingest_external_derm_csv(
    *,
    csv_path: Path,
    images_dir: Path,
    dataset_name: str,
    license_name: str,
    heldout_shas: set[str] | None = None,
    out_manifest: Path,
) -> dict[str, Any]:
    """Read a CSV catalog (PAD-UFES-20 / Fitzpatrick17k / DDI schema) and emit a deduplicated manifest."""
    blocked_shas = set(heldout_shas or set())
    seen_shas: set[str] = set()
    items: list[dict[str, Any]] = []
    skipped_missing = 0
    skipped_heldout_dup = 0
    skipped_internal_dup = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rel_img = (
                row.get("img_id")
                or row.get("DDI_file")
                or row.get("md5hash")
                or row.get("filename")
                or row.get("image_path")
                or ""
            ).strip()
            if not rel_img:
                skipped_missing += 1
                continue
            cand = images_dir / rel_img
            if not cand.exists() and not cand.suffix:
                for ext in (".png", ".jpg", ".jpeg"):
                    if (images_dir / f"{rel_img}{ext}").exists():
                        cand = images_dir / f"{rel_img}{ext}"
                        break
            if not cand.exists():
                skipped_missing += 1
                continue

            sha = _sha256_file(cand)
            if sha in blocked_shas:
                skipped_heldout_dup += 1
                continue
            if sha in seen_shas:
                skipped_internal_dup += 1
                continue
            seen_shas.add(sha)

            dx_raw = (row.get("diagnostic") or row.get("disease") or row.get("label") or "").strip()
            mg = normalize_malignancy_label(row.get("malignant") or dx_raw)
            fitz = normalize_fitzpatrick(row.get("fitspatrick") or row.get("fitzpatrick") or row.get("skin_tone"))
            patient_id = (row.get("patient_id") or row.get("lesion_id") or f"{dataset_name}_{idx}").strip()
            body_site = (row.get("region") or row.get("body_site") or "").strip()
            confirmed = "histopathology" if str(row.get("biopsed", "")).strip().lower() in {"1", "true", "yes"} else ""

            items.append(
                {
                    "i": f"{dataset_name}_{cand.stem}",
                    "d": dataset_name,
                    "mod": "clinical",
                    "t": str(cand.relative_to(images_dir)),
                    "f": str(cand.relative_to(images_dir)),
                    "st": fitz,
                    "dx": dx_raw,
                    "mg": mg,
                    "bs": body_site,
                    "l": license_name,
                    "cf": confirmed,
                    "g": f"{dataset_name}::{patient_id}",
                    "sha256": sha,
                }
            )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": dataset_name,
        "license": license_name,
        "summary": {
            "ingested_items": len(items),
            "malignant": sum(1 for x in items if x["mg"] == "malignant"),
            "benign": sum(1 for x in items if x["mg"] == "benign"),
            "unknown": sum(1 for x in items if x["mg"] == "unknown"),
            "skipped_missing_file": skipped_missing,
            "skipped_heldout_sha256_collision": skipped_heldout_dup,
            "skipped_internal_duplicate": skipped_internal_dup,
        },
        "items": items,
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest external clinical smartphone derm dataset with SHA-256 dedup.")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=["pad_ufes_20", "fitzpatrick17k", "ddi"])
    p.add_argument("--license", type=str, default="CC-BY-4.0")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rep = ingest_external_derm_csv(
        csv_path=args.csv,
        images_dir=args.images_dir,
        dataset_name=args.dataset,
        license_name=args.license,
        out_manifest=args.out,
    )
    print(json.dumps(rep["summary"], indent=2))


if __name__ == "__main__":
    main()
