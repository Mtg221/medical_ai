"""Download real smartphone camera photos of printed paper documents from Hugging Face (`naver-clova-ix/cord-v2`, CC-BY-4.0) into `data/router_real/external_cord_v2_phone/`."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

DATASET_ID = "naver-clova-ix/cord-v2"
DATASET_LICENSE = "CC-BY-4.0"
HF_SERVER = "https://datasets-server.huggingface.co"


def _extract_image_url(row: dict[str, Any]) -> str | None:
    """Find the image URL in a Hugging Face datasets-server row."""
    for val in row.values():
        if isinstance(val, dict) and isinstance(val.get("src"), str) and val["src"].startswith("http"):
            return val["src"]
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and isinstance(item.get("src"), str) and item["src"].startswith("http"):
                    return item["src"]
    return None


def download_cord_v2_phone_docs(
    out_dir: str | Path = "data/router_real/external_cord_v2_phone",
    sources_csv: str | Path = "data/router_real/SOURCES.csv",
    max_images: int = 100,
    preferred_splits: tuple[str, ...] = ("test",),
    bucket_name: str = "external_cord_v2_phone",
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(sources_csv)

    existing_hashes: set[str] = set()
    rows: list[list[str]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = list(csv.reader(fh))
            if reader:
                rows = reader[1:]
                existing_hashes = {r[4] for r in rows if len(r) >= 5}

    headers = {"User-Agent": "medical_ai_research/1.0 (https://github.com/Mtg221/medical_ai)"}
    downloaded = 0
    skipped_dup = 0
    failed = 0
    used_splits: list[str] = []
    new_rows: list[list[str]] = []

    with httpx.Client(headers=headers, timeout=30.0, follow_redirects=True) as client:
        splits_resp = client.get(f"{HF_SERVER}/splits", params={"dataset": DATASET_ID})
        splits_resp.raise_for_status()
        splits_data = splits_resp.json().get("splits") or []
        if not splits_data:
            raise RuntimeError(f"No splits returned for {DATASET_ID}")

        available = [(str(s["config"]), str(s["split"])) for s in splits_data]
        ordered: list[tuple[str, str]] = []
        for pref in preferred_splits:
            for cfg, spl in available:
                if spl == pref and (cfg, spl) not in ordered:
                    ordered.append((cfg, spl))
        if not ordered:
            raise RuntimeError(f"None of requested splits {preferred_splits} found in {available}")

        page_size = 25
        for config, split in ordered:
            if downloaded + skipped_dup >= max_images:
                break
            used_splits.append(f"{config}:{split}")
            offset = 0
            while downloaded + skipped_dup < max_images:
                length = min(page_size, max_images - (downloaded + skipped_dup))
                resp = None
                for attempt in range(4):
                    resp = client.get(
                        f"{HF_SERVER}/rows",
                        params={
                            "dataset": DATASET_ID,
                            "config": config,
                            "split": split,
                            "offset": offset,
                            "length": length,
                        },
                    )
                    if resp.status_code < 500:
                        break
                    time.sleep(1.5 * (attempt + 1))
                assert resp is not None
                resp.raise_for_status()
                batch = resp.json().get("rows") or []
                if not batch:
                    break

                for entry in batch:
                    row_idx = int(entry.get("row_idx", offset))
                    row = entry.get("row") or {}
                    img_url = _extract_image_url(row)
                    if not img_url:
                        failed += 1
                        continue
                    try:
                        img_resp = client.get(img_url)
                        if img_resp.status_code != 200 or len(img_resp.content) < 1000:
                            failed += 1
                            continue
                        sha = hashlib.sha256(img_resp.content).hexdigest()
                        if sha in existing_hashes:
                            skipped_dup += 1
                            continue
                        with Image.open(io.BytesIO(img_resp.content)) as im:
                            rgb = im.convert("RGB")
                            if min(rgb.size) < 64:
                                failed += 1
                                continue
                            fname = f"cord_v2_phone_{split}_{row_idx:03d}_{sha[:10]}.jpg"
                            # Re-encoding via PIL strips EXIF/embedded metadata.
                            rgb.save(out_path / fname, format="JPEG", quality=92)
                        existing_hashes.add(sha)
                        new_rows.append(
                            [
                                bucket_name,
                                "phone_printed_paper",
                                fname,
                                DATASET_LICENSE,
                                sha,
                                f"https://huggingface.co/datasets/{DATASET_ID}",
                                f"{DATASET_ID} [{split}:{row_idx}]",
                            ]
                        )
                        downloaded += 1
                    except Exception:
                        failed += 1
                    time.sleep(0.05)

                offset += len(batch)
                if len(batch) < length:
                    break

    all_rows = rows + new_rows
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "category", "filename", "license", "sha256", "source_url", "title"])
        w.writerows(all_rows)

    return {
        "dataset": DATASET_ID,
        "license": DATASET_LICENSE,
        "used_splits": used_splits,
        "downloaded": downloaded,
        "skipped_duplicate": skipped_dup,
        "failed": failed,
        "total_in_dir": len(list(out_path.glob("*.jpg"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/router_real/external_cord_v2_phone")
    parser.add_argument("--sources-csv", default="data/router_real/SOURCES.csv")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--splits", default="test", help="Comma-separated split names (e.g. 'train' or 'validation')")
    parser.add_argument("--bucket", default="external_cord_v2_phone", help="Bucket name logged in SOURCES.csv")
    parser.add_argument(
        "--download-train-val",
        action="store_true",
        help="Download strictly disjoint train (120) and validation (30) phone-doc splits without touching test.",
    )
    args = parser.parse_args()
    if args.download_train_val:
        res_tr = download_cord_v2_phone_docs(
            "data/router_real/document_scan_phone_train",
            args.sources_csv,
            max_images=120,
            preferred_splits=("train",),
            bucket_name="document_scan_phone_train",
        )
        res_va = download_cord_v2_phone_docs(
            "data/router_real/document_scan_phone_val",
            args.sources_csv,
            max_images=30,
            preferred_splits=("validation",),
            bucket_name="document_scan_phone_val",
        )
        print(json.dumps({"train": res_tr, "validation": res_va}, indent=2))
        return
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    result = download_cord_v2_phone_docs(
        args.out_dir, args.sources_csv, args.max_images, preferred_splits=splits, bucket_name=args.bucket
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
