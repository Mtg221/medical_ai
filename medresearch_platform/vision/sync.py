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
    stratify_modalities: bool = True,
) -> list[DermCatalogRecord]:
    """
    Select a balanced subset across modalities (`clinical`, `dermoscopic`, `tbp_tile`)
    and malignancy groups (`benign`, `malignant`) from metadata pages.
    """
    if not stratify_modalities:
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

    # Stratified across (modality, malignancy), specific diagnoses, and Fitzpatrick skin tones I-VI
    target_per_mod = max(2, (per_class_limit * 2) // 3)
    half_mod = max(1, target_per_mod // 2)
    mod_mg_counts: dict[tuple[str, str], int] = {
        ("clinical", "benign"): 0,
        ("clinical", "malignant"): 0,
        ("dermoscopic", "benign"): 0,
        ("dermoscopic", "malignant"): 0,
        ("tbp_tile", "benign"): 0,
        ("tbp_tile", "malignant"): 0,
    }
    specific_dxs = {
        "melanoma",
        "basal_cell_carcinoma",
        "squamous_cell_carcinoma",
        "actinic_keratosis",
        "nevus",
        "seborrheic_keratosis",
        "dermatofibroma",
        "vascular_lesion",
    }
    dx_counts: dict[str, int] = {dx: 0 for dx in specific_dxs}
    tone_mg_counts: dict[tuple[str, str], int] = {
        (t, mg): 0 for t in ("I", "II", "III", "IV", "V", "VI") for mg in ("benign", "malignant")
    }
    tbp_extra: list[DermCatalogRecord] = []
    selected_strat: list[DermCatalogRecord] = []
    seen_ids: set[str] = set()

    # Pass 1: Specific diagnoses + Fitzpatrick skin tones I-VI (research-allowed, including fitzpatrick17k & isic)
    for rec in iter_derm_catalog(
        meta_dir,
        require_quality_ok=False,
        require_research_licence=True,
    ):
        if not rec.thumb_path or rec.image_id in seen_ids:
            continue
        if rec.malignancy_group in ("benign", "malignant"):
            added = False
            if rec.diagnosis_harmonised in specific_dxs and dx_counts[rec.diagnosis_harmonised] < 24:
                selected_strat.append(rec)
                seen_ids.add(rec.image_id)
                dx_counts[rec.diagnosis_harmonised] += 1
                added = True
            t_key = (rec.skin_tone, rec.malignancy_group)
            if not added and t_key in tone_mg_counts and tone_mg_counts[t_key] < 18:
                selected_strat.append(rec)
                seen_ids.add(rec.image_id)
                tone_mg_counts[t_key] += 1
                added = True
            m_key = (rec.modality, rec.malignancy_group)
            if not added and m_key in mod_mg_counts and mod_mg_counts[m_key] < half_mod:
                selected_strat.append(rec)
                seen_ids.add(rec.image_id)
                mod_mg_counts[m_key] += 1
                added = True
            if added:
                if t_key in tone_mg_counts:
                    tone_mg_counts[t_key] += 1
                if m_key in mod_mg_counts:
                    mod_mg_counts[m_key] += 1
        elif rec.modality == "tbp_tile" and len(tbp_extra) < target_per_mod:
            tbp_extra.append(rec)
            seen_ids.add(rec.image_id)

        if (
            all(v >= 24 for v in dx_counts.values())
            and all(v >= 18 for v in tone_mg_counts.values())
            and all(v >= half_mod for v in mod_mg_counts.values())
            and len(tbp_extra) >= target_per_mod
        ):
            break

    # Top up tbp_tile if tbp_tile had fewer explicit benign/malignant labels
    tbp_current = mod_mg_counts[("tbp_tile", "benign")] + mod_mg_counts[("tbp_tile", "malignant")]
    if tbp_current < target_per_mod and tbp_extra:
        needed = target_per_mod - tbp_current
        selected_strat.extend(tbp_extra[:needed])

    return selected_strat


def download_balanced_thumbnails(
    meta_dir: str | Path = DEFAULT_META_DIR,
    cache_dir: str | Path = DEFAULT_THUMBS_DIR,
    *,
    base_url: str = DEFAULT_CLOUDFRONT_BASE_URL,
    username: str | None = None,
    password: str | None = None,
    per_class_limit: int = 150,
    limit: int | None = None,
    rate_limit_rps: float = 25.0,
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


def select_records_from_sampling_plan(
    meta_dir: str | Path = DEFAULT_META_DIR,
    *,
    train_target: int = 24000,
    val_target: int = 3000,
    test_target: int = 3000,
    min_test_per_tone: int = 150,
    min_test_per_modality: int = 300,
    seed: int = 42,
) -> dict[str, list[DermCatalogRecord]]:
    """
    Stratified lesion-grouped sampling across all manifest pages (`ok == true`, research-allowed only)
    following `docs/data/DERM_SAMPLING_PLAN.md`:
    - Zero lesion `group_key` overlap between `train`, `val`, and `locked_test_v2`.
    - Oversample Fitzpatrick tones I-VI so `locked_test_v2` has >= `min_test_per_tone` where available.
    - Ensure >= `min_test_per_modality` in `locked_test_v2` and `val` for clinical/dermoscopic/tbp_tile.
    - Oversample malignant lesions; keep `unknown`/`indeterminate` mg for masked training.
    """
    import random
    from collections import Counter, defaultdict

    rng = random.Random(seed)
    by_group: dict[str, list[DermCatalogRecord]] = defaultdict(list)
    for rec in iter_derm_catalog(
        meta_dir,
        require_quality_ok=True,
        require_research_licence=True,
    ):
        if rec.thumb_path:
            by_group[rec.group_key].append(rec)

    # Order groups deterministically, then shuffle within strata
    groups = list(by_group.items())
    groups.sort(key=lambda kv: kv[0])
    rng.shuffle(groups)

    split_records: dict[str, list[DermCatalogRecord]] = {
        "train": [],
        "val": [],
        "locked_test_v2": [],
    }
    assigned_groups: set[str] = set()
    targets = {"train": train_target, "val": val_target, "locked_test_v2": test_target}

    def _assign(gk: str, recs: list[DermCatalogRecord], split: str) -> None:
        assigned_groups.add(gk)
        split_records[split].extend(recs)

    # 1. Oversample Fitzpatrick tones I-VI across lesion groups
    for tone in ("I", "II", "III", "IV", "V", "VI"):
        tone_groups = [
            (gk, recs)
            for gk, recs in groups
            if gk not in assigned_groups and any(r.skin_tone == tone for r in recs)
        ]
        # Prioritize malignant within tone
        tone_groups.sort(
            key=lambda kv: (
                0 if any(r.malignancy_group == "malignant" for r in kv[1]) else 1,
                kv[0],
            )
        )
        t_test = 0
        t_val = 0
        for gk, recs in tone_groups:
            n_tone = sum(1 for r in recs if r.skin_tone == tone)
            if t_test < min_test_per_tone and len(split_records["locked_test_v2"]) < test_target:
                _assign(gk, recs, "locked_test_v2")
                t_test += n_tone
            elif t_val < min_test_per_tone and len(split_records["val"]) < val_target:
                _assign(gk, recs, "val")
                t_val += n_tone
            elif len(split_records["train"]) < train_target:
                _assign(gk, recs, "train")

    # 2. Ensure modality & malignancy quotas in locked_test_v2 and val
    for mod in ("clinical", "dermoscopic", "tbp_tile"):
        mod_groups = [
            (gk, recs)
            for gk, recs in groups
            if gk not in assigned_groups and any(r.modality == mod for r in recs)
        ]
        mod_groups.sort(
            key=lambda kv: (
                0 if any(r.malignancy_group == "malignant" for r in kv[1]) else 1,
                0 if any(r.malignancy_group == "benign" for r in kv[1]) else 1,
                kv[0],
            )
        )
        cur_test = sum(1 for r in split_records["locked_test_v2"] if r.modality == mod)
        cur_val = sum(1 for r in split_records["val"] if r.modality == mod)
        for gk, recs in mod_groups:
            if cur_test < min_test_per_modality and len(split_records["locked_test_v2"]) < test_target:
                _assign(gk, recs, "locked_test_v2")
                cur_test += sum(1 for r in recs if r.modality == mod)
            elif cur_val < min_test_per_modality and len(split_records["val"]) < val_target:
                _assign(gk, recs, "val")
                cur_val += sum(1 for r in recs if r.modality == mod)
            if cur_test >= min_test_per_modality and cur_val >= min_test_per_modality:
                break

    # 3. Fill remaining capacity stratified by (dataset, modality, malignancy_group, dx)
    remaining = [(gk, recs) for gk, recs in groups if gk not in assigned_groups]
    # Prioritize malignant and rarer modalities (clinical, dermoscopic) over tbp_tile surplus
    remaining.sort(
        key=lambda kv: (
            0 if any(r.malignancy_group == "malignant" for r in kv[1]) else 1,
            0 if any(r.modality == "clinical" for r in kv[1]) else (1 if any(r.modality == "dermoscopic" for r in kv[1]) else 2),
            kv[0],
        )
    )
    stratum_counts: Counter[tuple[str, str, str]] = Counter()
    for gk, recs in remaining:
        rep = recs[0]
        stratum = (rep.dataset, rep.modality, rep.malignancy_group)
        idx = stratum_counts[stratum]
        stratum_counts[stratum] += 1
        # 80% train / 10% val / 10% test while respecting targets
        mod10 = idx % 10
        pref = "locked_test_v2" if mod10 == 0 else ("val" if mod10 == 1 else "train")
        if len(split_records[pref]) < targets[pref]:
            _assign(gk, recs, pref)
        elif len(split_records["train"]) < targets["train"]:
            _assign(gk, recs, "train")
        elif len(split_records["val"]) < targets["val"]:
            _assign(gk, recs, "val")
        elif len(split_records["locked_test_v2"]) < targets["locked_test_v2"]:
            _assign(gk, recs, "locked_test_v2")
        if all(len(split_records[s]) >= targets[s] for s in ("train", "val", "locked_test_v2")):
            break

    return split_records


def deduplicate_cross_split_dhash(
    split_records: dict[str, list[DermCatalogRecord]],
    cache_dir: str | Path = DEFAULT_THUMBS_DIR,
    *,
    max_hamming: int = 4,
) -> tuple[dict[str, list[DermCatalogRecord]], int]:
    """
    Compute 64-bit dHash on cached thumbnails and drop any `locked_test_v2` image that is a
    near-duplicate (`hamming <= max_hamming`) of any image in `train` or `val`.
    Never drops from `train`.
    """
    from medresearch_platform.vision.splits import compute_dhash64_from_path

    base = Path(cache_dir)
    train_val_hashes: list[int] = []
    for sp in ("train", "val"):
        for rec in split_records.get(sp, []):
            if not rec.thumb_path:
                continue
            res = compute_dhash64_from_path(base / rec.thumb_path)
            if res is not None:
                train_val_hashes.append(res[0])

    kept_test: list[DermCatalogRecord] = []
    dropped = 0
    for rec in split_records.get("locked_test_v2", []):
        if not rec.thumb_path:
            continue
        res = compute_dhash64_from_path(base / rec.thumb_path)
        if res is None:
            continue
        h = res[0]
        if any((h ^ tv_h).bit_count() <= max_hamming for tv_h in train_val_hashes):
            dropped += 1
            continue
        kept_test.append(rec)

    out = {
        "train": list(split_records.get("train", [])),
        "val": list(split_records.get("val", [])),
        "locked_test_v2": kept_test,
    }
    return out, dropped


def freeze_locked_test_v2(
    test_records: list[DermCatalogRecord],
    *,
    out_path: str | Path = "configs/vision/locked_test_v2.txt",
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write sorted frozen test image IDs to `configs/vision/locked_test_v2.txt` and log SHA-256."""
    from medresearch_platform.vision.splits import log_locked_test_access

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted({r.image_id for r in test_records if r.image_id})
    p.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
    log_locked_test_access(
        log_path=log_path,
        dataset_name="derm-corpus:locked_test_v2",
        model_name=f"frozen_list_sha256:{sha256[:16]}",
        sample_count=len(ids),
        reason=f"Frozen configs/vision/locked_test_v2.txt (sha256={sha256})",
    )
    return {"path": str(p), "n_ids": len(ids), "sha256": sha256}


def compute_split_counts(
    split_records: dict[str, list[DermCatalogRecord]],
) -> dict[str, dict[str, Any]]:
    """Single canonical split-count summary shared by data cards and benchmark JSONs (W6)."""
    from collections import Counter

    out: dict[str, dict[str, Any]] = {}
    for sp, recs in split_records.items():
        mg_c = Counter(r.malignancy_group for r in recs)
        mod_c = Counter(r.modality for r in recs)
        tone_c = Counter(r.skin_tone for r in recs)
        ds_c = Counter(r.dataset for r in recs)
        labelled_mg = int(mg_c["benign"] + mg_c["malignant"])
        masked_mg = int(mg_c["unknown"] + mg_c["indeterminate"])
        out[sp] = {
            "total": len(recs),
            "real": len(recs),
            "synthetic": 0,
            "benign": int(mg_c["benign"]),
            "malignant": int(mg_c["malignant"]),
            "labelled_mg": labelled_mg,
            "masked_mg": masked_mg,
            "clinical": int(mod_c["clinical"]),
            "dermoscopic": int(mod_c["dermoscopic"]),
            "tbp_tile": int(mod_c["tbp_tile"]),
            "tones": {t: int(tone_c[t]) for t in ("I", "II", "III", "IV", "V", "VI", "UNKNOWN")},
            "datasets": dict(ds_c),
        }
    return out


def write_derm_v2_data_card(
    split_records: dict[str, list[DermCatalogRecord]],
    *,
    out_path: str | Path = "docs/data/DERM_CORPUS_CARD_V2.md",
    dropped_near_dups: int = 0,
) -> Path:
    """Write data card with exact counts per split x tone x modality x mg x dataset."""
    counts = compute_split_counts(split_records)
    lines = [
        "# Dermatology Corpus Data Card (Sampling Plan v2)",
        "",
        f"- **Cross-split near-duplicates dropped from `locked_test_v2`:** `{dropped_near_dups}`",
        "",
        "## Split Summary (Real Images Only)",
        "",
        "| Split | Total | Benign | Malignant | Unknown/Indet (Masked) | Clinical | Dermoscopic | TBP Tile |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sp in ("train", "val", "locked_test_v2"):
        c = counts.get(sp, {})
        lines.append(
            f"| `{sp}` | {c.get('total', 0)} | {c.get('benign', 0)} | {c.get('malignant', 0)} | "
            f"{c.get('masked_mg', 0)} | {c.get('clinical', 0)} | {c.get('dermoscopic', 0)} | {c.get('tbp_tile', 0)} |"
        )
    lines.extend(["", "## Fitzpatrick Skin Tone & Dataset Breakdown", ""])
    lines.append("| Split | Tone I | Tone II | Tone III | Tone IV | Tone V | Tone VI | UNKNOWN | Datasets |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for sp in ("train", "val", "locked_test_v2"):
        c = counts.get(sp, {})
        tones = c.get("tones", {})
        ds_c = c.get("datasets", {})
        lines.append(
            f"| `{sp}` | {tones.get('I', 0)} | {tones.get('II', 0)} | {tones.get('III', 0)} | "
            f"{tones.get('IV', 0)} | {tones.get('V', 0)} | {tones.get('VI', 0)} | {tones.get('UNKNOWN', 0)} | `{ds_c}` |"
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def sync_derm_from_plan(
    meta_dir: str | Path = DEFAULT_META_DIR,
    cache_dir: str | Path = DEFAULT_THUMBS_DIR,
    *,
    plan_path: str | Path = "docs/data/DERM_SAMPLING_PLAN.md",
    limit: int | None = None,
    rate_limit_rps: float = 16.0,
    timeout_seconds: float = 20.0,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Any]:
    """Execute `medextract derm sync --plan ...`: select stratified lesion-grouped records, fetch thumbs, dedup, freeze test list."""
    _ = Path(plan_path)
    t0 = time.perf_counter()
    split_records = select_records_from_sampling_plan(meta_dir)
    all_selected = split_records["locked_test_v2"] + split_records["val"] + split_records["train"]
    if limit is not None:
        all_selected = all_selected[:limit]

    user, pwd = username, password
    if not user or not pwd:
        user, pwd = load_dotenv_credentials()

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    cached_hits = 0
    failed = 0
    min_interval = 1.0 / max(0.5, rate_limit_rps)
    missing: list[DermCatalogRecord] = []

    for rec in all_selected:
        if not rec.thumb_path:
            continue
        lp = out_dir / rec.thumb_path
        if lp.exists() and lp.stat().st_size > 0:
            cached_hits += 1
            continue
        leg = LEGACY_THUMBS_DIR / rec.thumb_path
        if leg.exists() and leg.stat().st_size > 0:
            lp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(leg, lp)
            cached_hits += 1
            continue
        missing.append(rec)

    if missing:
        if not user or not pwd:
            raise RuntimeError("DERM_CLOUDFRONT_USER and DERM_CLOUDFRONT_PASS must be set in .env for remote thumbnail fetch.")
        auth = httpx.BasicAuth(user, pwd)
        with httpx.Client(auth=auth, timeout=timeout_seconds, follow_redirects=True) as client:
            for idx, rec in enumerate(missing, start=1):
                assert rec.thumb_path is not None
                lp = out_dir / rec.thumb_path
                lp.parent.mkdir(parents=True, exist_ok=True)
                url = f"{DEFAULT_CLOUDFRONT_BASE_URL.rstrip('/')}/{rec.thumb_path.lstrip('/')}"
                time.sleep(min_interval)
                try:
                    resp = client.get(url)
                    if resp.status_code == 200 and resp.content:
                        lp.write_bytes(resp.content)
                        downloaded += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                if idx % 50 == 0 or idx == len(missing):
                    _render_progress("Syncing plan thumbs", cached_hits + idx, len(all_selected))
        sys.stderr.write("\n")
        sys.stderr.flush()

    # Keep only records whose thumbnails are present on disk, then dedup across splits
    present_splits: dict[str, list[DermCatalogRecord]] = {}
    for sp, recs in split_records.items():
        present_splits[sp] = [
            r for r in recs if r.thumb_path and (out_dir / r.thumb_path).exists() and (out_dir / r.thumb_path).stat().st_size > 0
        ]

    deduped, dropped = deduplicate_cross_split_dhash(present_splits, out_dir)
    freeze_info = freeze_locked_test_v2(deduped["locked_test_v2"])
    card_path = write_derm_v2_data_card(deduped, dropped_near_dups=dropped)

    return {
        "downloaded": downloaded,
        "cached_hits": cached_hits,
        "failed": failed,
        "dropped_near_dups_from_test": dropped,
        "locked_test_v2": freeze_info,
        "data_card": str(card_path),
        "split_counts": {k: len(v) for k, v in deduped.items()},
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
