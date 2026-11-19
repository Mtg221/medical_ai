"""Download de-identified real lab reports & prescriptions from Hugging Face (`ekacare/medical_records_parsing_validation_set`, MIT) into `data/router_real/external_ekacare_mit/`.

Requires accepting the dataset access gate on Hugging Face and providing a read-only `HF_TOKEN`
via environment variable, gitignored `.env` (`HF_TOKEN=hf_...`), or `~/.cache/huggingface/token`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

DATASET_ID = "ekacare/medical_records_parsing_validation_set"
DATASET_LICENSE = "MIT"
HF_SERVER = "https://datasets-server.huggingface.co"


def _load_hf_token() -> str | None:
    """Resolve Hugging Face token from env, gitignored `.env`, or `~/.cache/huggingface/token`."""
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val

    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
                tok = v.strip().strip('"').strip("'")
                if tok:
                    return tok

    cache_tok = Path.home() / ".cache" / "huggingface" / "token"
    if cache_tok.exists():
        tok = cache_tok.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    return None


def _extract_image_url(row: dict[str, Any]) -> tuple[str | None, str]:
    """Find the image URL and optional document type/category field in a dataset row."""
    img_url: str | None = None
    category = "medical_document"
    for key in ("doc_type", "document_type", "type", "category", "label"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            category = val.strip().lower().replace(" ", "_")[:40]
            break

    for val in row.values():
        if isinstance(val, dict) and isinstance(val.get("src"), str) and val["src"].startswith("http"):
            img_url = val["src"]
            break
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and isinstance(item.get("src"), str) and item["src"].startswith("http"):
                    img_url = item["src"]
                    break
            if img_url:
                break
    return img_url, category


def download_ekacare_docs(
    out_dir: str | Path = "data/router_real/external_ekacare_mit",
    sources_csv: str | Path = "data/router_real/SOURCES.csv",
    max_images: int = 288,
) -> dict[str, Any]:
    token = _load_hf_token()
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Add `HF_TOKEN=hf_...` to gitignored `.env` "
            "(or export `HF_TOKEN` / save to `~/.cache/huggingface/token`) after clicking "
            f"'Agree and access repository' at https://huggingface.co/datasets/{DATASET_ID}"
        )

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

    headers = {
        "User-Agent": "medical_ai_research/1.0 (https://github.com/Mtg221/medical_ai)",
        "Authorization": f"Bearer {token}",
    }
    downloaded = 0
    skipped_dup = 0
    failed = 0
    new_rows: list[list[str]] = []

    with httpx.Client(headers=headers, timeout=30.0, follow_redirects=True) as client:
        splits_resp = client.get(f"{HF_SERVER}/splits", params={"dataset": DATASET_ID})
        if splits_resp.status_code in (401, 403):
            raise RuntimeError(
                f"Hugging Face returned HTTP {splits_resp.status_code}. Verify your read-only token is valid "
                f"and that you clicked 'Agree and access repository' on https://huggingface.co/datasets/{DATASET_ID}"
            )
        splits_resp.raise_for_status()
        splits_data = splits_resp.json().get("splits") or []
        if not splits_data:
            raise RuntimeError(f"No splits returned for {DATASET_ID}")
        config = str(splits_data[0]["config"])
        split = str(splits_data[0]["split"])

        offset = 0
        page_size = 25
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
                img_url, category = _extract_image_url(row)
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
                        fname = f"ekacare_{category}_{row_idx:03d}_{sha[:10]}.jpg"
                        rgb.save(out_path / fname, format="JPEG", quality=92)
                    existing_hashes.add(sha)
                    new_rows.append(
                        [
                            "external_ekacare_mit",
                            category,
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
        "config": config,
        "split": split,
        "downloaded": downloaded,
        "skipped_duplicate": skipped_dup,
        "failed": failed,
        "total_in_dir": len(list(out_path.glob("*.jpg"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/router_real/external_ekacare_mit")
    parser.add_argument("--sources-csv", default="data/router_real/SOURCES.csv")
    parser.add_argument("--max-images", type=int, default=288)
    args = parser.parse_args()
    try:
        result = download_ekacare_docs(args.out_dir, args.sources_csv, args.max_images)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
