"""Dermatology image catalog loader, leak-free splitter, and synthetic offline phantom generator."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator
import numpy as np


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
    allowed_malignancy: set[str] | None = None,
    limit: int | None = None,
) -> Iterator[DermCatalogRecord]:
    """Stream DermCatalogRecord items from gzip/plain JSON metadata pages without high RAM usage."""
    base = Path(meta_dir)
    count = 0
    for path in sorted(base.glob("*.json")):
        modality = _infer_modality_from_filename(path)
        payload = _open_json_auto(path)
        items = payload.get("items", []) if isinstance(payload, dict) else payload
        for row in items:
            quality_ok = bool(row.get("ok", False))
            if require_quality_ok and not quality_ok:
                continue
            mg = str(row.get("mg") or "unknown").lower()
            if allowed_malignancy is not None and mg not in allowed_malignancy:
                continue

            image_id = str(row.get("i") or "")
            dataset = str(row.get("d") or "unknown")
            # Group by dataset + subject/lesion prefix when available to prevent case leakage
            case_token = str(row.get("mk") or image_id.split(":")[-1][:12])
            group_key = f"{dataset}:{case_token}"
            skin_tone = str(row.get("st") or "UNKNOWN").upper()
            if skin_tone not in SKIN_TONES:
                skin_tone = "UNKNOWN"

            yield DermCatalogRecord(
                image_id=image_id,
                dataset=dataset,
                modality=modality,
                thumb_path=row.get("t"),
                raw_path=row.get("f"),
                skin_tone=skin_tone,
                diagnosis_raw=row.get("dx"),
                malignancy_group=mg,
                body_site=row.get("bs"),
                license_name=row.get("lc"),
                quality_ok=quality_ok,
                confirmation=row.get("cf"),
                group_key=group_key,
                split_assignment=_deterministic_split(group_key),
            )
            count += 1
            if limit is not None and count >= limit:
                return


def summarize_derm_catalog(meta_dir: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    """Compute dataset, modality, malignancy_group, skin_tone, and split counts."""
    splits: Counter[str] = Counter()
    malignancy: Counter[str] = Counter()
    modalities: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    skin_tones: Counter[str] = Counter()
    total = 0
    quality_ok_count = 0

    for rec in iter_derm_catalog(meta_dir, limit=limit):
        total += 1
        if rec.quality_ok:
            quality_ok_count += 1
        splits[rec.split_assignment] += 1
        malignancy[rec.malignancy_group] += 1
        modalities[rec.modality] += 1
        datasets[rec.dataset] += 1
        skin_tones[rec.skin_tone] += 1

    return {
        "total_records": total,
        "quality_ok_count": quality_ok_count,
        "splits": dict(splits),
        "malignancy_group": dict(malignancy),
        "modalities": dict(modalities),
        "datasets": dict(datasets),
        "skin_tones": dict(skin_tones),
    }


def generate_synthetic_derm_samples(
    n_samples: int = 48,
    image_size: int = 16,
    seed: int = 42,
) -> list[tuple[np.ndarray, int, DermCatalogRecord]]:
    """
    Generate deterministic synthetic RGB dermatology phantoms (benign=0, malignant=1)
    across varied skin tones and modalities for offline CPU verification.
    """
    rng = np.random.default_rng(seed)
    samples: list[tuple[np.ndarray, int, DermCatalogRecord]] = []
    tones = ("I", "II", "III", "IV", "V", "VI")
    mods = ("clinical", "dermoscopic", "tbp_tile")

    for idx in range(n_samples):
        label = idx % 2  # 0=benign, 1=malignant
        tone = tones[idx % len(tones)]
        modality = mods[idx % len(mods)]
        group_key = f"synth_case:{idx // 2}"  # pairs stay in the same split
        split = _deterministic_split(group_key)

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
            diagnosis_raw="synthetic_benign" if label == 0 else "synthetic_malignant",
            malignancy_group=MALIGNANCY_CLASSES[label],
            body_site="torso",
            license_name="SYNTHETIC-INTERNAL",
            quality_ok=True,
            confirmation="synthetic_phantom",
            group_key=group_key,
            split_assignment=split,
        )
        samples.append((img.astype(np.float64), label, rec))

    return samples
