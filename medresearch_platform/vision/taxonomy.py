"""Harmonised dermatology diagnosis taxonomy, malignancy consistency checker, and licence policy classifier."""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TAXONOMY_PATH = Path("configs/vision/dx_taxonomy.yaml")

HARMONISED_CLASSES: tuple[str, ...] = (
    "melanoma",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "cutaneous_lymphoma_sarcoma",
    "actinic_keratosis",
    "nevus",
    "seborrheic_keratosis",
    "dermatofibroma",
    "vascular_lesion",
    "benign_cyst_adnexal",
    "inflammatory_infectious_dermatosis",
    "benign_unspecified",
    "malignant_unspecified",
    "other",
)


@dataclass(frozen=True)
class HarmonisedDiagnosisResult:
    raw_dx: str | None
    harmonised_dx: str
    expected_malignancy: str
    resolved_malignancy: str
    mg_consistent: bool
    icd10: str
    label_en: str
    label_fr: str


@dataclass(frozen=True)
class LicencePolicyResult:
    raw_license: str | None
    normalized_license: str
    research_allowed: bool
    redistribution_allowed: bool
    commercial_allowed: bool
    docs_screenshot_safe: bool


@lru_cache(maxsize=4)
def load_dx_taxonomy(path: str | Path = DEFAULT_TAXONOMY_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        # Fallback relative to repository root
        repo_root = Path(__file__).resolve().parents[2]
        p = repo_root / path
    if not p.exists():
        raise FileNotFoundError(f"Diagnosis taxonomy file not found: {path}")
    return dict(yaml.safe_load(p.read_text(encoding="utf-8")) or {})


def _extract_primary_dx(val: str | None) -> str:
    if not val:
        return ""
    s = str(val).strip()
    if s.startswith("{") and s.endswith("}"):
        if s == "{}":
            return ""
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict) and parsed:
                top_k = max(parsed.items(), key=lambda kv: float(kv[1]))[0]
                return str(top_k)
        except Exception:
            return ""
    return s


def _normalize_text(val: str | None) -> str:
    primary = _extract_primary_dx(val)
    if not primary:
        return ""
    cleaned = re.sub(r"[_/\-]+", " ", primary.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


@lru_cache(maxsize=16384)
def harmonise_diagnosis(
    raw_dx: str | None,
    raw_mg: str | None = None,
    *,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> HarmonisedDiagnosisResult:
    """Map raw diagnosis string (`dx`) to a harmonised class and check consistency with `mg`."""
    cfg = load_dx_taxonomy(taxonomy_path)
    classes_cfg: dict[str, Any] = cfg.get("classes", {})
    fallback = str(cfg.get("fallback_class", "other"))

    norm_dx = _normalize_text(raw_dx)
    norm_mg = _normalize_text(raw_mg) or "unknown"
    if norm_mg not in ("benign", "malignant", "indeterminate"):
        norm_mg = "unknown"

    matched_class = fallback
    if norm_dx and norm_dx not in ("other", "unknown", "indeterminate", "none"):
        # 1. Exact alias match
        for cls_name, cls_info in classes_cfg.items():
            if cls_name == fallback:
                continue
            aliases = [_normalize_text(a) for a in cls_info.get("aliases", [])]
            if norm_dx in aliases or norm_dx == _normalize_text(cls_name):
                matched_class = cls_name
                break

        # 2. Substring / token boundary match if no exact match
        if matched_class == fallback:
            for cls_name, cls_info in classes_cfg.items():
                if cls_name in (fallback, "benign_unspecified", "malignant_unspecified"):
                    continue
                for alias in cls_info.get("aliases", []):
                    norm_alias = _normalize_text(alias)
                    if len(norm_alias) >= 4 and norm_alias in norm_dx:
                        matched_class = cls_name
                        break
                if matched_class != fallback:
                    break

    cls_meta = classes_cfg.get(matched_class, classes_cfg.get(fallback, {}))
    expected_mg = str(cls_meta.get("malignancy_group", "any"))
    allowed_mg = [str(m).lower() for m in cls_meta.get("allowed_mg", ["benign", "malignant", "unknown"])]

    if norm_mg == "unknown":
        resolved_mg = expected_mg if expected_mg in ("benign", "malignant") else "unknown"
        mg_consistent = True
    else:
        resolved_mg = norm_mg
        mg_consistent = (expected_mg == "any") or (norm_mg in allowed_mg)

    return HarmonisedDiagnosisResult(
        raw_dx=raw_dx,
        harmonised_dx=matched_class,
        expected_malignancy=expected_mg,
        resolved_malignancy=resolved_mg,
        mg_consistent=mg_consistent,
        icd10=str(cls_meta.get("icd10", "L98.9")),
        label_en=str(cls_meta.get("label_en", matched_class)),
        label_fr=str(cls_meta.get("label_fr", matched_class)),
    )


@lru_cache(maxsize=1024)
def evaluate_licence_policy(
    raw_lc: str | None,
    *,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> LicencePolicyResult:
    """Classify an image's licence string (`lc`) for research, redistribution, and documentation safety."""
    cfg = load_dx_taxonomy(taxonomy_path)
    pol = cfg.get("licence_policy", {})
    allowed_res = {str(x).upper() for x in pol.get("allowed_research", [])}
    safe_docs = {str(x).upper() for x in pol.get("attribution_safe_for_docs_and_screenshots", [])}
    nc_or_nd = {str(x).upper() for x in pol.get("non_commercial_or_no_redistribution", [])}
    excluded = {str(x).upper() for x in pol.get("excluded", [])}

    if not raw_lc or not str(raw_lc).strip():
        return LicencePolicyResult(
            raw_license=raw_lc,
            normalized_license="UNKNOWN",
            research_allowed=False,
            redistribution_allowed=False,
            commercial_allowed=False,
            docs_screenshot_safe=False,
        )

    norm = str(raw_lc).strip().upper().replace("_", "-").replace(" ", "-")
    if norm in excluded or "FORBIDDEN" in norm or "ALL-RIGHTS-RESERVED" in norm or "NO-RESEARCH" in norm:
        return LicencePolicyResult(
            raw_license=raw_lc,
            normalized_license=norm,
            research_allowed=False,
            redistribution_allowed=False,
            commercial_allowed=False,
            docs_screenshot_safe=False,
        )

    is_nc = ("NC" in norm) or (norm in nc_or_nd)
    is_nd = "ND" in norm
    is_cc0_or_by = norm in safe_docs or norm.startswith("CC0") or (norm.startswith("CC-BY") and not is_nc and not is_nd)
    research_ok = (norm in allowed_res) or norm.startswith("CC-") or norm.startswith("CC0")

    return LicencePolicyResult(
        raw_license=raw_lc,
        normalized_license=norm,
        research_allowed=research_ok,
        redistribution_allowed=research_ok and not is_nd and not is_nc,
        commercial_allowed=research_ok and not is_nc,
        docs_screenshot_safe=is_cc0_or_by,
    )


def summarize_taxonomy_mapping(
    raw_dx_pairs: Sequence[tuple[str | None, str | None]],
    *,
    min_class_count: int | None = None,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> dict[str, Any]:
    """
    Build a taxonomy summary table over `(raw_dx, raw_mg)` pairs, collapsing classes
    under `min_class_count` into `other`.
    """
    cfg = load_dx_taxonomy(taxonomy_path)
    threshold = min_class_count if min_class_count is not None else int(cfg.get("min_class_count", 30))
    fallback = str(cfg.get("fallback_class", "other"))

    pair_counts = Counter(raw_dx_pairs)
    pair_res = {pair: harmonise_diagnosis(pair[0], pair[1], taxonomy_path=taxonomy_path) for pair in pair_counts}

    initial_counts: Counter[str] = Counter()
    for pair, cnt in pair_counts.items():
        initial_counts[pair_res[pair].harmonised_dx] += cnt

    # Identify classes below threshold (only when total corpus size >= threshold)
    total_n = len(raw_dx_pairs)
    collapsed_classes: set[str] = set()
    if total_n >= threshold:
        for cls_name, cnt in initial_counts.items():
            if cls_name != fallback and cnt < threshold:
                collapsed_classes.add(cls_name)

    final_counts: Counter[str] = Counter()
    raw_to_harmonised: dict[str, dict[str, Any]] = {}
    inconsistent_mg_count = 0

    for (raw_dx, raw_mg), cnt in pair_counts.items():
        res = pair_res[(raw_dx, raw_mg)]
        final_cls = fallback if res.harmonised_dx in collapsed_classes else res.harmonised_dx
        final_counts[final_cls] += cnt
        if not res.mg_consistent:
            inconsistent_mg_count += cnt
        key = str(raw_dx or "<null>")
        entry = raw_to_harmonised.setdefault(
            key,
            {
                "raw_dx": key,
                "initial_class": res.harmonised_dx,
                "final_class": final_cls,
                "expected_malignancy": res.expected_malignancy,
                "count": 0,
                "mg_distribution": Counter(),
                "mg_inconsistent_count": 0,
            },
        )
        entry["count"] += cnt
        entry["mg_distribution"][str(raw_mg or "unknown").lower()] += cnt
        if not res.mg_consistent:
            entry["mg_inconsistent_count"] += cnt

    # Convert Counter to dict for JSON serialization
    mapping_rows = []
    for k in sorted(raw_to_harmonised.keys(), key=lambda x: (-raw_to_harmonised[x]["count"], x)):
        row = dict(raw_to_harmonised[k])
        row["mg_distribution"] = dict(row["mg_distribution"])
        mapping_rows.append(row)

    return {
        "taxonomy_id": cfg.get("taxonomy_id", "DERM-DX-HARMONISED-V1"),
        "min_class_count": threshold,
        "total_records": total_n,
        "collapsed_to_other": sorted(collapsed_classes),
        "harmonised_class_counts": dict(final_counts),
        "mg_inconsistent_count": inconsistent_mg_count,
        "raw_mapping_table": mapping_rows,
    }
