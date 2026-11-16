"""Authenticated CloudFront metadata & thumbnail synchronizer with resume, rate-limit, SHA-256 checksums, and progress reporting."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from medresearch_platform.vision.catalog import (
    DermCatalogRecord,
    iter_derm_catalog,
)

DEFAULT_CLOUDFRONT_BASE_URL = "https://d38rablk9v12ns.cloudfront.net"
DEFAULT_META_DIR = Path("var/cache/derm/meta")
DEFAULT_THUMBS_DIR = Path("var/cache/derm/thumbs")
LEGACY_THUMBS_DIR = Path("var/cache/derm_thumbs")


@dataclass(frozen=True)
class DermMetaSyncReport:
    downloaded_pages: int
    cached_pages: int
    total_pages: int
    total_records: int
    meta_dir: str
    manifest_sha256: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DermFetchReport:
    downloaded: int
    cached_hits: int
    failed: int
    total_selected: int
    cache_dir: str
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dotenv_credentials(dotenv_path: str | Path = ".env") -> tuple[str | None, str | None]:
    """
    Read DERM_CLOUDFRONT_USER and DERM_CLOUDFRONT_PASS from environment or `.env` file.
    Credentials are never printed, logged, or embedded into URLs.
    """
    user = os.environ.get("DERM_CLOUDFRONT_USER")
    pwd = os.environ.get("DERM_CLOUDFRONT_PASS")
    if user and pwd:
        return user.strip(), pwd.strip()

    candidates = [Path(dotenv_path), Path(__file__).resolve().parents[2] / ".env"]
    env_map: dict[str, str] = {}
    for p in candidates:
        if p.exists() and p.is_file():
            for raw_line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                env_map[k] = v
            break

    return (user or env_map.get("DERM_CLOUDFRONT_USER")), (pwd or env_map.get("DERM_CLOUDFRONT_PASS"))


def _render_progress(prefix: str, current: int, total: int | None) -> None:
    if total and total > 0:
        pct = min(100.0, (current / total) * 100.0)
        bar_len = 24
        filled = int(round(bar_len * current / total))
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stderr.write(f"\r{prefix} [{bar}] {current}/{total} ({pct:5.1f}%)")
    else:
        sys.stderr.write(f"\r{prefix} [{current} items]")
    sys.stderr.flush()


def _discover_manifest_paths_from_root(client: httpx.Client, base_url: str) -> list[str]:
    """Inspect index HTML / JSON / JS bundle on CloudFront to discover exact manifest page paths."""
    discovered: list[str] = []
    root_url = f"{base_url.rstrip('/')}/"
    resp = client.get(root_url)
    if resp.status_code == 401:
        raise RuntimeError(
            "HTTP 401 Unauthorized from derm-corpus CloudFront (check DERM_CLOUDFRONT_USER / DERM_CLOUDFRONT_PASS in .env)."
        )
    if resp.status_code == 200:
        text = resp.text
        # Match any .json or .json.gz paths in HTML
        for m in re.findall(r'["\']([a-zA-Z0-9_./\-]+\.json(?:\.gz)?)["\']', text):
            if not m.endswith("package.json") and not m.endswith("tsconfig.json"):
                discovered.append(m.lstrip("/"))
        # Also check any referenced JS bundles
        js_paths = re.findall(r'src=["\']([^"\']+\.js)["\']', text)
        for js_rel in js_paths[:8]:
            js_url = f"{base_url.rstrip('/')}/{js_rel.lstrip('/')}"
            js_resp = client.get(js_url)
            if js_resp.status_code == 200:
                for m in re.findall(r'["\']([a-zA-Z0-9_./\-]+\.json(?:\.gz)?)["\']', js_resp.text):
                    if "manifest" in m or "meta" in m or "catalog" in m or "index" in m or "derived" in m:
                        discovered.append(m.lstrip("/"))

    # Check common index manifest endpoints
    for idx_candidate in (
        "manifest.json",
        "gallery/manifest.json",
        "index.json",
        "derived/manifest.json",
        "derived/manifests/index.json",
        "meta/index.json",
    ):
        try:
            r = client.get(f"{base_url.rstrip('/')}/{idx_candidate}")
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    # Parse {"modalities": {"clinical": {"pages": ["gallery/manifest-clinical-000.json"]}, ...}}
                    mods = data.get("modalities")
                    if isinstance(mods, dict):
                        for mod_info in mods.values():
                            if isinstance(mod_info, dict) and isinstance(mod_info.get("pages"), list):
                                for page_path in mod_info["pages"]:
                                    if isinstance(page_path, str) and ".json" in page_path:
                                        discovered.append(page_path.lstrip("/"))
                    for key in ("pages", "manifests", "files"):
                        val = data.get(key)
                        if isinstance(val, list):
                            for item in val:
                                if isinstance(item, str) and ".json" in item:
                                    discovered.append(item.lstrip("/"))
                                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                                    discovered.append(item["path"].lstrip("/"))
        except Exception:
            continue

    # Filter out the top-level index file "manifest.json" itself if modality pages were discovered
    page_only = [p for p in discovered if Path(p).name != "manifest.json"]
    return list(dict.fromkeys(page_only if page_only else discovered))


def sync_derm_metadata(
    meta_dir: str | Path = DEFAULT_META_DIR,
    *,
    base_url: str = DEFAULT_CLOUDFRONT_BASE_URL,
    username: str | None = None,
    password: str | None = None,
    rate_limit_rps: float = 8.0,
    timeout_seconds: float = 20.0,
    resume: bool = True,
) -> DermMetaSyncReport:
    """
    Download gzip/plain JSON metadata pages from `derm-corpus` CloudFront into `var/cache/derm/meta/`
    with resume support, rate limiting, SHA-256 verification, and progress reporting.
    """
    t0 = time.perf_counter()
    out_dir = Path(meta_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy any legacy /tmp/derm_meta pages if present
    legacy_tmp = Path("/tmp/derm_meta")
    if legacy_tmp.exists() and legacy_tmp.resolve() != out_dir.resolve():
        for p in sorted(legacy_tmp.glob("*.json*")):
            dest = out_dir / p.name
            if not dest.exists():
                shutil.copy2(p, dest)

    user, pwd = username, password
    if not user or not pwd:
        user, pwd = load_dotenv_credentials()

    existing_pages = sorted(
        p for p in list(out_dir.glob("*.json")) + list(out_dir.glob("*.json.gz"))
        if p.name != "sync_manifest.json"
    )
    if (not user or not pwd) and existing_pages:
        # Offline resume using already downloaded pages
        page_hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in existing_pages
        }
        manifest_bytes = json.dumps(page_hashes, sort_keys=True).encode("utf-8")
        total_recs = sum(1 for _ in iter_derm_catalog(out_dir))
        return DermMetaSyncReport(
            downloaded_pages=0,
            cached_pages=len(existing_pages),
            total_pages=len(existing_pages),
            total_records=total_recs,
            meta_dir=str(out_dir),
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            elapsed_seconds=round(time.perf_counter() - t0, 3),
        )

    if not user or not pwd:
        raise RuntimeError(
            "DERM_CLOUDFRONT_USER and DERM_CLOUDFRONT_PASS are not set in .env or environment, "
            f"and no cached metadata pages were found in {out_dir}."
        )

    auth = httpx.BasicAuth(user, pwd)
    min_interval = 1.0 / max(0.5, rate_limit_rps)
    downloaded_pages = 0
    cached_pages = 0

    with httpx.Client(auth=auth, timeout=timeout_seconds, follow_redirects=True) as client:
        discovered = _discover_manifest_paths_from_root(client, base_url)

        # Download any explicitly discovered manifest paths first
        for rel_path in discovered:
            fname = Path(rel_path).name
            dest = out_dir / fname
            if resume and dest.exists() and dest.stat().st_size > 0:
                cached_pages += 1
                _render_progress("Syncing metadata", cached_pages + downloaded_pages, len(discovered))
                continue
            time.sleep(min_interval)
            resp = client.get(f"{base_url.rstrip('/')}/{rel_path.lstrip('/')}")
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content)
                downloaded_pages += 1
                _render_progress("Syncing metadata", cached_pages + downloaded_pages, len(discovered))

        # Probe standard paginated modality manifests if none were discovered via root index
        if downloaded_pages + cached_pages == 0:
            prefixes = ("", "derived/meta/", "derived/manifests/", "meta/", "manifests/")
            modalities = ("clinical", "dermoscopic", "tbp_tile")
            active_prefix: str | None = None

            # Find which prefix serves manifest-{mod}-000.json
            for pfx in prefixes:
                test_rel = f"{pfx}manifest-clinical-000.json"
                r = client.get(f"{base_url.rstrip('/')}/{test_rel}")
                if r.status_code == 401:
                    raise RuntimeError("HTTP 401 Unauthorized from derm-corpus CloudFront.")
                if r.status_code == 200 and r.content:
                    active_prefix = pfx
                    break

            if active_prefix is not None:
                for mod in modalities:
                    page_idx = 0
                    while page_idx < 500:
                        fname = f"manifest-{mod}-{page_idx:03d}.json"
                        dest = out_dir / fname
                        if resume and dest.exists() and dest.stat().st_size > 0:
                            cached_pages += 1
                            page_idx += 1
                            _render_progress("Syncing metadata", cached_pages + downloaded_pages, None)
                            continue
                        time.sleep(min_interval)
                        rel_url = f"{active_prefix}{fname}"
                        resp = client.get(f"{base_url.rstrip('/')}/{rel_url}")
                        if resp.status_code == 200 and resp.content:
                            dest.write_bytes(resp.content)
                            downloaded_pages += 1
                            page_idx += 1
                            _render_progress("Syncing metadata", cached_pages + downloaded_pages, None)
                        else:
                            break

    if downloaded_pages + cached_pages > 0:
        sys.stderr.write("\n")
        sys.stderr.flush()

    final_pages = sorted(
        p for p in list(out_dir.glob("*.json")) + list(out_dir.glob("*.json.gz"))
        if p.name not in ("sync_manifest.json", "manifest.json")
    )
    page_entries = []
    for p in final_pages:
        raw = p.read_bytes()
        page_entries.append({
            "filename": p.name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    manifest_sha = hashlib.sha256(
        json.dumps(page_entries, sort_keys=True).encode("utf-8")
    ).hexdigest()
    total_records = sum(1 for _ in iter_derm_catalog(out_dir))

    sync_manifest = {
        "synced_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_pages": len(final_pages),
        "total_records": total_records,
        "manifest_sha256": manifest_sha,
        "pages": page_entries,
    }
    (out_dir / "sync_manifest.json").write_text(
        json.dumps(sync_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    return DermMetaSyncReport(
        downloaded_pages=downloaded_pages,
        cached_pages=cached_pages,
        total_pages=len(final_pages),
        total_records=total_records,
        meta_dir=str(out_dir),
        manifest_sha256=manifest_sha,
        elapsed_seconds=round(time.perf_counter() - t0, 3),
    )


def select_balanced_records(
    meta_dir: str | Path,
    *,
    per_class_limit: int = 150,
    require_quality_ok: bool = True,
) -> list[DermCatalogRecord]:
    """Select a balanced benign/malignant subset across splits from metadata pages."""
    counts = {"benign": 0, "malignant": 0}
    selected: list[DermCatalogRecord] = []
    for rec in iter_derm_catalog(
        meta_dir,
        require_quality_ok=require_quality_ok,
        require_research_licence=True,
        allowed_malignancy={"benign", "malignant"},
    ):
        if not rec.thumb_path:
            continue
        mg = rec.malignancy_group
        if counts[mg] >= per_class_limit:
            if all(v >= per_class_limit for v in counts.values()):
                break
            continue
        selected.append(rec)
        counts[mg] += 1
        if all(v >= per_class_limit for v in counts.values()):
            break
    return selected


def download_balanced_thumbnails(
    meta_dir: str | Path = DEFAULT_META_DIR,
    cache_dir: str | Path = DEFAULT_THUMBS_DIR,
    *,
    base_url: str = DEFAULT_CLOUDFRONT_BASE_URL,
    username: str | None = None,
    password: str | None = None,
    per_class_limit: int = 150,
    limit: int | None = None,
    rate_limit_rps: float = 10.0,
    timeout_seconds: float = 15.0,
) -> DermFetchReport:
    """
    Download thumbnails into `var/cache/derm/thumbs/` with resume, legacy cache reuse,
    rate limiting, SHA-256 verification, and progress reporting.
    """
    t0 = time.perf_counter()
    user, pwd = username, password
    if not user or not pwd:
        user, pwd = load_dotenv_credentials()

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if limit is not None:
        per_class = max(1, (limit + 1) // 2)
        records = select_balanced_records(meta_dir, per_class_limit=per_class)[:limit]
    else:
        records = select_balanced_records(meta_dir, per_class_limit=per_class_limit)

    downloaded = 0
    cached_hits = 0
    failed = 0
    min_interval = 1.0 / max(0.5, rate_limit_rps)

    # First check local cache hits (including legacy var/cache/derm_thumbs)
    missing_records: list[DermCatalogRecord] = []
    for rec in records:
        if not rec.thumb_path:
            continue
        local_path = out_dir / rec.thumb_path
        if local_path.exists() and local_path.stat().st_size > 0:
            cached_hits += 1
            continue
        legacy_path = LEGACY_THUMBS_DIR / rec.thumb_path
        if legacy_path.exists() and legacy_path.stat().st_size > 0:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, local_path)
            cached_hits += 1
            continue
        missing_records.append(rec)

    if missing_records:
        if not user or not pwd:
            raise RuntimeError("DERM_CLOUDFRONT_USER and DERM_CLOUDFRONT_PASS must be set in .env for remote thumbnail fetch.")
        auth = httpx.BasicAuth(user, pwd)
        with httpx.Client(auth=auth, timeout=timeout_seconds, follow_redirects=True) as client:
            for idx, rec in enumerate(missing_records, start=1):
                assert rec.thumb_path is not None
                local_path = out_dir / rec.thumb_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                url = f"{base_url.rstrip('/')}/{rec.thumb_path.lstrip('/')}"
                time.sleep(min_interval)
                try:
                    resp = client.get(url)
                    if resp.status_code == 200 and resp.content:
                        local_path.write_bytes(resp.content)
                        downloaded += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                _render_progress("Syncing thumbnails", cached_hits + idx, len(records))
        sys.stderr.write("\n")
        sys.stderr.flush()

    return DermFetchReport(
        downloaded=downloaded,
        cached_hits=cached_hits,
        failed=failed,
        total_selected=len(records),
        cache_dir=str(out_dir),
        elapsed_seconds=round(time.perf_counter() - t0, 3),
    )
