"""Dermatology image catalog loader, leak-free lesion splitter, licence gate, and synthetic phantom generator."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from medresearch_platform.vision.taxonomy import (
    evaluate_licence_policy,
    harmonise_diagnosis,
)

MALIGNANCY_CLASSES: tuple[str, ...] = ("benign", "malignant")
SKIN_TONES: tuple[str, ...] = ("I", "II", "III", "IV", "V", "VI", "UNKNOWN")
MODALITIES: tuple[str, ...] = ("clinical", "dermoscopic", "tbp_tile", "unknown")


@dataclass(frozen=True)
class DermCatalogRecord:
    image_id: str
    dataset: str
    modality: str
    thumb_path: str | None
    raw_path: str | None
    skin_tone: str
    diagnosis_raw: str | None
    malignancy_group: str
    body_site: str | None
    license_name: str | None
    quality_ok: bool
    confirmation: str | None
    group_key: str
    split_assignment: str
    diagnosis_harmonised: str = "other"
    mg_consistent: bool = True
    research_allowed: bool = True
    case_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deterministic_split(group_key: str) -> str:
    digest = hashlib.sha256(group_key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "locked_test"


def _infer_modality_from_filename(path: Path) -> str:
    name = path.name.lower()
    for mod in ("clinical", "dermoscopic", "tbp_tile"):
        if mod in name:
            return mod
    return "unknown"


def _open_json_auto(path: Path) -> Any:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        return json.loads(gzip.decompress(raw).decode("utf-8"))
    return json.loads(raw.decode("utf-8"))


def iter_derm_catalog(
    meta_dir: str | Path,
    *,
    require_quality_ok: bool = False,
    require_research_licence: bool = False,
    allowed_malignancy: set[str] | None = None,
    limit: int | None = None,
) -> Iterator[DermCatalogRecord]:
    """Stream DermCatalogRecord items from gzip/plain JSON metadata pages without high RAM usage."""
    base = Path(meta_dir)
    if not base.exists():
        return
    count = 0
    candidate_files = sorted(
        p
        for p in list(base.glob("*.json")) + list(base.glob("*.json.gz"))
        if p.name not in ("sync_manifest.json", "manifest.json")
    )
    for path in candidate_files:
        file_modality = _infer_modality_from_filename(path)
        payload = _open_json_auto(path)
        if isinstance(payload, dict):
            items = payload.get("items", [])
            page_modality = str(payload.get("modality") or file_modality).lower()
        elif isinstance(payload, list):
            items = payload
            page_modality = file_modality
        else:
            continue

        for row in items:
            if not isinstance(row, dict):
                continue
            quality_ok = bool(row.get("ok", False))
            if require_quality_ok and not quality_ok:
                continue

            lc_raw = row.get("lc")
            lc_policy = evaluate_licence_policy(lc_raw)
            if require_research_licence and not lc_policy.research_allowed:
                continue

            raw_dx = row.get("dx")
            raw_mg = str(row.get("mg") or "unknown").lower()
            dx_info = harmonise_diagnosis(raw_dx, raw_mg)
            mg = raw_mg
            if allowed_malignancy is not None and mg not in allowed_malignancy:
                continue

            image_id = str(row.get("i") or "")
            dataset = str(row.get("d") or "unknown")
            raw_mk = row.get("mk")
            is_monk_scale = bool(raw_mk and ("us=" in str(raw_mk) or "india=" in str(raw_mk)))
            explicit_case = str(raw_mk) if (raw_mk and not is_monk_scale) else None
            case_token = explicit_case if explicit_case else image_id.split(":")[-1]
            group_key = f"{dataset}:{case_token}"

            skin_tone = str(row.get("st") or "UNKNOWN").upper()
            if skin_tone not in SKIN_TONES:
                skin_tone = "UNKNOWN"

            row_mod = str(row.get("modality") or page_modality or "unknown").lower()
            if row_mod not in MODALITIES:
                row_mod = "unknown"

            yield DermCatalogRecord(
                image_id=image_id,
                dataset=dataset,
                modality=row_mod,
                thumb_path=row.get("t"),
                raw_path=row.get("f"),
                skin_tone=skin_tone,
                diagnosis_raw=raw_dx,
                malignancy_group=mg,
                body_site=row.get("bs"),
                license_name=lc_raw,
                quality_ok=quality_ok,
                confirmation=row.get("cf"),
                group_key=group_key,
                split_assignment=_deterministic_split(group_key),
                diagnosis_harmonised=dx_info.harmonised_dx,
                mg_consistent=dx_info.mg_consistent,
                research_allowed=lc_policy.research_allowed,
                case_key=explicit_case,
            )
            count += 1
            if limit is not None and count >= limit:
                return


def summarize_derm_catalog(
    meta_dir: str | Path,
    *,
    limit: int | None = None,
    records: list[DermCatalogRecord] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive dataset, modality, licence, malignancy, diagnosis, skin tone, body site, confirmation, and lesion counts."""
    splits: Counter[str] = Counter()
    malignancy: Counter[str] = Counter()
    modalities: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    skin_tones: Counter[str] = Counter()
    licences: Counter[str] = Counter()
    raw_diagnoses: Counter[str] = Counter()
    harmonised_diagnoses: Counter[str] = Counter()
    body_sites: Counter[str] = Counter()
    confirmations: Counter[str] = Counter()

    unique_image_ids: set[str] = set()
    duplicate_image_ids = 0
    unique_group_keys: set[str] = set()
    explicit_mk_keys: set[str] = set()
    explicit_mk_records = 0
    mg_inconsistent_count = 0
    research_allowed_count = 0
    redistribution_allowed_count = 0
    commercial_allowed_count = 0
    docs_safe_count = 0

    total = 0
    quality_ok_count = 0

    stream = records if records is not None else iter_derm_catalog(meta_dir, limit=limit)
    for rec in stream:
        total += 1
        if rec.quality_ok:
            quality_ok_count += 1
        if rec.image_id in unique_image_ids:
            duplicate_image_ids += 1
        else:
            unique_image_ids.add(rec.image_id)

        unique_group_keys.add(rec.group_key)
        if rec.case_key:
            explicit_mk_records += 1
            explicit_mk_keys.add(f"{rec.dataset}:{rec.case_key}")

        splits[rec.split_assignment] += 1
        malignancy[rec.malignancy_group] += 1
        modalities[rec.modality] += 1
        datasets[rec.dataset] += 1
        skin_tones[rec.skin_tone] += 1
        licences[str(rec.license_name or "UNKNOWN")] += 1
        raw_diagnoses[str(rec.diagnosis_raw or "UNKNOWN")] += 1
        harmonised_diagnoses[rec.diagnosis_harmonised] += 1
        body_sites[str(rec.body_site or "UNKNOWN")] += 1
        confirmations[str(rec.confirmation or "UNKNOWN")] += 1

        if not rec.mg_consistent:
            mg_inconsistent_count += 1

        lc_pol = evaluate_licence_policy(rec.license_name)
        if lc_pol.research_allowed:
            research_allowed_count += 1
        if lc_pol.redistribution_allowed:
            redistribution_allowed_count += 1
        if lc_pol.commercial_allowed:
            commercial_allowed_count += 1
        if lc_pol.docs_screenshot_safe:
            docs_safe_count += 1

    return {
        "total_records": total,
        "unique_images": len(unique_image_ids),
        "duplicate_image_id_count": duplicate_image_ids,
        "unique_lesion_groups": len(unique_group_keys),
        "explicit_mk_records": explicit_mk_records,
        "unique_explicit_mk_lesions": len(explicit_mk_keys),
        "quality_ok_count": quality_ok_count,
        "quality_rejected_count": total - quality_ok_count,
        "mg_inconsistent_count": mg_inconsistent_count,
        "licence_summary": {
            "by_licence": dict(licences.most_common()),
            "research_allowed_count": research_allowed_count,
            "redistribution_allowed_count": redistribution_allowed_count,
            "commercial_allowed_count": commercial_allowed_count,
            "docs_screenshot_safe_count": docs_safe_count,
            "excluded_or_unknown_count": total - research_allowed_count,
        },
        "splits": dict(splits),
        "malignancy_group": dict(malignancy),
        "modalities": dict(modalities),
        "datasets": dict(datasets.most_common()),
        "skin_tones": dict(skin_tones),
        "harmonised_diagnoses": dict(harmonised_diagnoses.most_common()),
        "raw_diagnoses_top50": dict(raw_diagnoses.most_common(50)),
        "raw_diagnoses_distinct_count": len(raw_diagnoses),
        "body_sites": dict(body_sites.most_common()),
        "confirmations": dict(confirmations.most_common()),
    }


def generate_synthetic_derm_samples(
    n_samples: int = 48,
    image_size: int = 16,
    seed: int = 42,
) -> list[tuple[np.ndarray, int, DermCatalogRecord]]:
    """
    Generate deterministic synthetic RGB dermatology phantoms (benign=0, malignant=1)
    across varied skin tones, body sites, diagnoses, and modalities for offline CPU verification.
    """
    rng = np.random.default_rng(seed)
    samples: list[tuple[np.ndarray, int, DermCatalogRecord]] = []
    tones = ("I", "II", "III", "IV", "V", "VI")
    mods = ("clinical", "dermoscopic", "tbp_tile")
    benign_dxs = ("nevus", "seborrheic_keratosis", "dermatofibroma", "vascular_lesion", "actinic_keratosis")
    malignant_dxs = ("melanoma", "basal_cell_carcinoma", "squamous_cell_carcinoma")
    sites = ("back", "anterior torso", "upper extremity", "lower extremity", "head/neck", "palms/soles")

    for idx in range(n_samples):
        label = idx % 2  # 0=benign, 1=malignant
        tone = tones[idx % len(tones)]
        modality = mods[idx % len(mods)]
        site = sites[idx % len(sites)]
        group_key = f"synth_case:{idx // 2}"  # pairs stay in the same split
        split = _deterministic_split(group_key)
        raw_dx = benign_dxs[(idx // 2) % len(benign_dxs)] if label == 0 else malignant_dxs[(idx // 2) % len(malignant_dxs)]

        # Skin background intensity varies by tone
        tone_idx = tones.index(tone)
        bg_base = np.array([0.85 - 0.1 * tone_idx, 0.72 - 0.1 * tone_idx, 0.62 - 0.09 * tone_idx])
        img = np.clip(
            bg_base[None, None, :] + rng.normal(0.0, 0.03, size=(image_size, image_size, 3)),
            0.05,
            0.98,
        )

        # Synthetic lesion patch: benign = symmetric light tan circle; malignant = asymmetric dark/variegated
        yy, xx = np.ogrid[:image_size, :image_size]
        cy, cx = image_size // 2, image_size // 2
        if label == 0:
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (image_size * 0.24) ** 2
            img[mask] = np.clip(np.array([0.55, 0.38, 0.28]) + rng.normal(0.0, 0.02, size=3), 0.0, 1.0)
        else:
            mask = (((yy - (cy - 1)) ** 2) / 1.4 + ((xx - (cx + 1)) ** 2) * 1.2) <= (image_size * 0.30) ** 2
            img[mask] = np.clip(np.array([0.18, 0.10, 0.12]) + rng.normal(0.0, 0.04, size=3), 0.0, 1.0)

        rec = DermCatalogRecord(
            image_id=f"synth:{idx:04d}",
            dataset="synthetic_derm_phantom",
            modality=modality,
            thumb_path=None,
            raw_path=None,
            skin_tone=tone,
            diagnosis_raw=raw_dx,
            malignancy_group=MALIGNANCY_CLASSES[label],
            body_site=site,
            license_name="SYNTHETIC-INTERNAL",
            quality_ok=True,
            confirmation="synthetic_phantom",
            group_key=group_key,
            split_assignment=split,
            diagnosis_harmonised=raw_dx,
            mg_consistent=True,
            research_allowed=True,
            case_key=f"case_{idx // 2:04d}",
        )
        samples.append((img.astype(np.float64), label, rec))

    return samples
