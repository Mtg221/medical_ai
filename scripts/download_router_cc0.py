"""Download CC0 / Public-Domain everyday ('other') and OOD check images from Wikimedia Commons and record SOURCES.csv."""

from __future__ import annotations

import csv
import hashlib
import io
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
ALLOWED_LICENSES = {"cc0", "public domain", "pd", "cc-zero"}

OTHER_QUERIES: list[tuple[str, str, int]] = [
    ("faces", "portrait face person", 50),
    ("rooms", "room interior architecture", 50),
    ("food", "food dish fruit", 50),
    ("screenshots", "computer screen window", 50),
    ("xrays", "radiograph x-ray medical", 50),
    ("animals", "cat dog bird", 50),
    ("people", "person standing walking", 50),
    ("objects", "chair table book cup", 50),
    ("nature", "tree flower river", 50),
]

OOD_QUERIES: list[tuple[str, str, int]] = [
    ("cars", "automobile car street", 10),
    ("landscapes", "mountain forest landscape", 10),
    ("hands", "human hand palm", 10),
    ("wounds", "skin wound bandage", 10),
    ("tattoos", "arm tattoo skin", 10),
]


def _is_cc0_or_pd(extmetadata: dict[str, Any]) -> str | None:
    short = str((extmetadata.get("LicenseShortName") or {}).get("value") or "").strip()
    low = short.lower()
    if any(tok in low for tok in ("cc0", "public domain", "pd-", "pd ")):
        return short or "CC0/PD"
    return None


def _search_commons(client: httpx.Client, query: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "gsroffset": str(offset),
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": "256",
    }
    resp = client.get(WIKIMEDIA_API, params=params)
    resp.raise_for_status()
    pages = (resp.json().get("query") or {}).get("pages") or {}
    return list(pages.values())


def download_cc0_sets(out_root: str | Path = "data/router_real") -> dict[str, Any]:
    root = Path(out_root)
    other_dir = root / "other"
    ood_dir = root / "ood_check"
    other_dir.mkdir(parents=True, exist_ok=True)
    ood_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    seen_sha: set[str] = set()
    sources_csv = root / "SOURCES.csv"
    if sources_csv.exists():
        with sources_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("sha256") and (root / r["filename"]).exists():
                    rows.append(r)
                    seen_sha.add(r["sha256"])

    headers = {"User-Agent": "medical_ai_research/1.0 (https://github.com/Mtg221/medical_ai)"}

    with httpx.Client(headers=headers, timeout=20.0, follow_redirects=True) as client:
        for split_name, target_dir, queries in [
            ("other", other_dir, OTHER_QUERIES),
            ("ood_check", ood_dir, OOD_QUERIES),
        ]:
            for cat, q, target_n in queries:
                collected = sum(1 for r in rows if r["split"] == split_name and r["category"] == cat)
                if collected >= target_n:
                    print(f"[{split_name}] {cat}: {collected}/{target_n} (cached)", flush=True)
                    continue
                for offset in (0, 100, 200):
                    if collected >= target_n:
                        break
                    try:
                        pages = _search_commons(client, q, limit=100, offset=offset)
                    except Exception:
                        break
                    if not pages:
                        break
                    for page in pages:
                        if collected >= target_n:
                            break
                        infos = page.get("imageinfo") or []
                        if not infos:
                            continue
                        info = infos[0]
                        lic = _is_cc0_or_pd(info.get("extmetadata") or {})
                        if not lic:
                            continue
                        url = info.get("thumburl") or info.get("url")
                        if not url:
                            continue
                        time.sleep(0.25)
                        try:
                            img_resp = client.get(url)
                            if img_resp.status_code != 200 or not img_resp.content:
                                continue
                            with Image.open(io.BytesIO(img_resp.content)) as im:
                                rgb = im.convert("RGB")
                                rgb.thumbnail((256, 256), Image.Resampling.BILINEAR)
                                buf = io.BytesIO()
                                rgb.save(buf, format="JPEG", quality=90)
                                clean_bytes = buf.getvalue()
                        except Exception:
                            continue

                        sha = hashlib.sha256(clean_bytes).hexdigest()
                        if sha in seen_sha:
                            continue
                        seen_sha.add(sha)

                        fname = f"{cat}_{collected:03d}_{sha[:10]}.jpg"
                        (target_dir / fname).write_bytes(clean_bytes)
                        rows.append({
                            "filename": f"{split_name}/{fname}",
                            "category": cat,
                            "split": split_name,
                            "license": lic,
                            "source_url": str(info.get("descriptionurl") or url),
                            "title": str(page.get("title") or ""),
                            "sha256": sha,
                        })
                        collected += 1
                print(f"[{split_name}] {cat}: {collected}/{target_n}", flush=True)

    sources_csv = root / "SOURCES.csv"
    with sources_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "category", "split", "license", "source_url", "title", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)

    other_n = sum(1 for r in rows if r["split"] == "other")
    ood_n = sum(1 for r in rows if r["split"] == "ood_check")
    return {"other_count": other_n, "ood_count": ood_n, "sources_csv": str(sources_csv)}


if __name__ == "__main__":
    import json

    print(json.dumps(download_cc0_sets(), indent=2))
