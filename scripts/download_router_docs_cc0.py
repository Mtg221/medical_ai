"""Download ~60 CC0 / Public-Domain real photos & scans of printed/handwritten documents into data/router_real/document_scan/."""

from __future__ import annotations

import csv
import hashlib
import io
import time
from pathlib import Path

import httpx
from download_router_cc0 import _is_cc0_or_pd, _search_commons
from PIL import Image

DOC_QUERIES: list[tuple[str, str, int]] = [
    ("prescription", "medical prescription paper", 15),
    ("form", "printed form document paper", 15),
    ("letter", "typed letter page document", 15),
    ("receipt", "paper receipt invoice", 15),
    ("notes", "handwritten medical note paper", 15),
]


def download_cc0_document_scans(
    out_dir: str | Path = "data/router_real/document_scan",
    sources_csv: str | Path = "data/router_real/SOURCES.csv",
    target_total: int = 60,
) -> dict[str, int]:
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
    new_rows: list[list[str]] = []

    with httpx.Client(headers=headers, timeout=20.0, follow_redirects=True) as client:
        for tag, query, cap in DOC_QUERIES:
            if downloaded >= target_total:
                break
            pages = _search_commons(client, query, limit=60, offset=0)
            taken = 0
            for page in pages:
                if taken >= cap or downloaded >= target_total:
                    break
                infos = page.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                extmeta = info.get("extmetadata") or {}
                lic = _is_cc0_or_pd(extmeta)
                if not lic:
                    continue
                url = str(info.get("thumburl") or info.get("url") or "")
                if not url:
                    continue
                try:
                    r = client.get(url)
                    if r.status_code != 200 or len(r.content) < 1000:
                        continue
                    sha = hashlib.sha256(r.content).hexdigest()
                    if sha in existing_hashes:
                        continue
                    with Image.open(io.BytesIO(r.content)) as im:
                        rgb = im.convert("RGB")
                        if min(rgb.size) < 64:
                            continue
                        fname = f"doc_cc0_{tag}_{taken:02d}_{sha[:10]}.jpg"
                        rgb.save(out_path / fname, format="JPEG", quality=90)
                    existing_hashes.add(sha)
                    new_rows.append(
                        [
                            "document_scan",
                            tag,
                            fname,
                            lic,
                            sha,
                            url,
                            str(page.get("title") or ""),
                        ]
                    )
                    downloaded += 1
                    taken += 1
                    time.sleep(0.15)
                except Exception:
                    continue

    all_rows = rows + new_rows
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "category", "filename", "license", "sha256", "source_url", "title"])
        w.writerows(all_rows)
    return {"downloaded": downloaded, "total_in_dir": len(list(out_path.glob("*.jpg")))}


if __name__ == "__main__":
    import json

    print(json.dumps(download_cc0_document_scans(), indent=2))
