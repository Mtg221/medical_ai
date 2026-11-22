"""Evaluate the 5-seed multi-scale Vision SE-ResNet ensemble (3x224 + 2x256) with 4x TTA,
val_v3-only per-modality temperature calibration, lesion-cluster bootstrap 95% CIs,
histopathology-confirmed subgroup metrics, and cross-seed uncertainty guardrail analysis.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from medresearch_platform.vision.catalog import (
    DermCatalogRecord,
    _infer_modality_from_filename,
    _open_json_auto,
)
from medresearch_platform.vision.cnn_torch import PyTorchDermResNet
from medresearch_platform.vision.eval_vision import (
    compute_auroc,
    compute_ece_and_reliability,
    compute_sensitivity_at_specificity,
    lesion_bootstrap_auroc_ci,
)
from scripts.train_vision_gpu import (
    _fit_temperature_nll,
    _load_samples_for_split,
    _logit,
    _metrics_dict,
    _row_to_record,
    _sigmoid,
)

ENSEMBLE_SPECS: list[tuple[str, int, int]] = [
    ("derm_cnn_bsdrop40_seed1", 1, 224),
    ("derm_cnn_seed2", 2, 224),
    ("derm_cnn_seed3", 3, 224),
    ("derm_cnn_seed4", 4, 256),
    ("derm_cnn_seed5", 5, 256),
]

UNCERTAINTY_HIGH_STD_THRESHOLD = 0.12


def _uncertainty_analysis(
    y: np.ndarray,
    p_cal: np.ndarray,
    std_seeds: np.ndarray,
    threshold: float = UNCERTAINTY_HIGH_STD_THRESHOLD,
) -> dict[str, Any]:
    high_mask = std_seeds > threshold
    low_mask = ~high_mask
    preds = (p_cal >= 0.5).astype(np.int64)
    errors = (preds != y).astype(np.float64)

    hist_edges = [0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.30, 1.0]
    hist_bins: list[dict[str, Any]] = []
    for lo, hi in zip(hist_edges[:-1], hist_edges[1:], strict=True):
        m = (std_seeds >= lo) & (std_seeds < hi if hi < 1.0 else std_seeds <= hi)
        cnt = int(np.sum(m))
        hist_bins.append({
            "range": f"[{lo:.2f}, {hi:.2f})",
            "count": cnt,
            "error_rate": round(float(np.mean(errors[m])), 4) if cnt > 0 else 0.0,
        })

    corr = float(np.corrcoef(std_seeds, errors)[0, 1]) if len(y) > 2 and float(np.std(std_seeds)) > 1e-8 else 0.0
    return {
        "threshold_std": threshold,
        "overall_mean_std": round(float(np.mean(std_seeds)), 4),
        "overall_median_std": round(float(np.median(std_seeds)), 4),
        "overall_p90_std": round(float(np.percentile(std_seeds, 90)), 4),
        "std_error_pearson_r": round(corr, 4),
        "low_uncertainty": {
            "n": int(np.sum(low_mask)),
            "fraction": round(float(np.mean(low_mask)), 4),
            "error_rate": round(float(np.mean(errors[low_mask])), 4) if np.any(low_mask) else 0.0,
            "auroc": round(float(compute_auroc(y[low_mask], p_cal[low_mask])), 4)
            if len(set(y[low_mask].tolist())) >= 2
            else None,
        },
        "high_uncertainty": {
            "n": int(np.sum(high_mask)),
            "fraction": round(float(np.mean(high_mask)), 4),
            "error_rate": round(float(np.mean(errors[high_mask])), 4) if np.any(high_mask) else 0.0,
            "auroc": round(float(compute_auroc(y[high_mask], p_cal[high_mask])), 4)
            if len(set(y[high_mask].tolist())) >= 2
            else None,
        },
        "histogram_bins": hist_bins,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Evaluating 5-seed multi-scale vision ensemble on device={device} (4x TTA)", flush=True)

    meta_dir = Path("var/cache/derm/meta")
    thumbs_dir = Path("var/cache/derm/thumbs")
    raw_dir = Path("var/cache/derm/raw")
    raw_root = raw_dir if raw_dir.exists() else None
    models_root = Path("models")

    val_ids = {x.strip() for x in Path("configs/vision/val_v3.txt").read_text(encoding="utf-8").splitlines() if x.strip()}
    test_ids = {x.strip() for x in Path("configs/vision/locked_test_v3.txt").read_text(encoding="utf-8").splitlines() if x.strip()}

    val_recs: list[DermCatalogRecord] = []
    test_recs: list[DermCatalogRecord] = []
    for p in sorted(meta_dir.glob("manifest-*.json*")):
        mod_hint = _infer_modality_from_filename(p)
        payload = _open_json_auto(p)
        items = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            continue
        for row in items:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("i", ""))
            if rid in val_ids:
                val_recs.append(_row_to_record(row, mod_hint, "val"))
            elif rid in test_ids:
                test_recs.append(_row_to_record(row, mod_hint, "test"))

    # Load images once per resolution (224 and 256) and ensure exact aligned ordering
    val_by_size: dict[int, list[tuple[np.ndarray, int, DermCatalogRecord]]] = {}
    test_by_size: dict[int, list[tuple[np.ndarray, int, DermCatalogRecord]]] = {}
    for sz in (224, 256):
        val_by_size[sz] = [s for s in _load_samples_for_split(val_recs, thumbs_dir, raw_root, sz) if s[1] in (0, 1)]
        test_by_size[sz] = [s for s in _load_samples_for_split(test_recs, thumbs_dir, raw_root, sz) if s[1] in (0, 1)]
        print(f"Loaded {sz}x{sz}: val={len(val_by_size[sz])}, locked_test={len(test_by_size[sz])}", flush=True)

    # Verify exact record alignment across resolutions
    assert [r.image_id for _, _, r in val_by_size[224]] == [r.image_id for _, _, r in val_by_size[256]]
    assert [r.image_id for _, _, r in test_by_size[224]] == [r.image_id for _, _, r in test_by_size[256]]

    val_y = np.array([lbl for _, lbl, _ in val_by_size[224]], dtype=np.int64)
    val_mods = [r.modality for _, _, r in val_by_size[224]]

    test_y = np.array([lbl for _, lbl, _ in test_by_size[224]], dtype=np.int64)
    test_mods = [r.modality for _, _, r in test_by_size[224]]
    test_cfs = [str(r.confirmation or "unknown").lower() for _, _, r in test_by_size[224]]
    test_groups = [r.group_key for _, _, r in test_by_size[224]]

    val_probs_per_model: list[np.ndarray] = []
    test_probs_per_model: list[np.ndarray] = []
    per_seed_summary: dict[str, Any] = {}

    for dir_name, seed, img_size in ENSEMBLE_SPECS:
        m_dir = models_root / dir_name
        model = PyTorchDermResNet.load(m_dir, device=device)
        assert int(model.image_size) == img_size, f"Expected {img_size} for {dir_name}, got {model.image_size}"
        t_m = time.perf_counter()
        vp = model.predict_batch(val_by_size[img_size], batch_size=64, tta=True)[0].astype(np.float64)
        tp = model.predict_batch(test_by_size[img_size], batch_size=64, tta=True)[0].astype(np.float64)
        val_probs_per_model.append(vp)
        test_probs_per_model.append(tp)
        m_stats = _metrics_dict(test_y, tp)
        per_seed_summary[dir_name] = {
            "seed": seed,
            "image_size": img_size,
            "raw_locked_test": m_stats,
            "inference_seconds": round(time.perf_counter() - t_m, 2),
        }
        print(
            f"  [{dir_name} ({img_size}px)] test raw AUROC={m_stats['auroc']:.4f} "
            f"sens@90={m_stats['sens_at_90_spec']:.4f} ECE={m_stats['ece']:.4f} "
            f"({per_seed_summary[dir_name]['inference_seconds']}s)",
            flush=True,
        )

    val_stack = np.stack(val_probs_per_model, axis=0)
    test_stack = np.stack(test_probs_per_model, axis=0)

    val_p = np.mean(val_stack, axis=0)
    test_p = np.mean(test_stack, axis=0)
    test_std = np.std(test_stack, axis=0)

    # Fit per-modality temperature strictly on val_v3
    mod_temps: dict[str, float] = {}
    for mod in ("dermoscopic", "clinical", "tbp_tile"):
        mask_v = np.array([m == mod for m in val_mods], dtype=bool)
        mod_temps[mod] = _fit_temperature_nll(val_p[mask_v], val_y[mask_v]) if int(np.sum(mask_v)) >= 20 else 1.0

    # Update live calibration file for serving
    cal_payload = {
        "fitted_on": "val_v3",
        "architecture_variant": "resnet_v2_se",
        "image_sizes": [224, 256],
        "ensemble_dirs": [name for name, _, _ in ENSEMBLE_SPECS],
        "use_tta": True,
        "seeds": [s for _, s, _ in ENSEMBLE_SPECS],
        "modality_temperatures": mod_temps,
        "uncertainty_high_std_threshold": UNCERTAINTY_HIGH_STD_THRESHOLD,
        "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (models_root / "derm_ensemble_calibration.json").write_text(
        json.dumps(cal_payload, indent=2) + "\n", encoding="utf-8"
    )

    test_p_cal = np.array(
        [
            float(_sigmoid(_logit(np.array([p])) / mod_temps.get(mod, 1.0))[0])
            for p, mod in zip(test_p, test_mods, strict=True)
        ],
        dtype=np.float64,
    )

    overall_raw = _metrics_dict(test_y, test_p)
    overall_cal = _metrics_dict(test_y, test_p_cal)
    overall_cal["auroc_ci95"] = lesion_bootstrap_auroc_ci(test_y, test_p_cal, test_groups, n_boot=500, seed=42)
    overall_cal["sens_at_95_spec"] = round(
        float(compute_sensitivity_at_specificity(test_y, test_p_cal, target_spec=0.95)), 4
    )
    _, rel_bins = compute_ece_and_reliability(test_y, test_p_cal)
    overall_cal["reliability_bins"] = rel_bins

    by_mod_cal: dict[str, Any] = {}
    for mod in ("dermoscopic", "clinical", "tbp_tile"):
        mask_t = np.array([m == mod for m in test_mods], dtype=bool)
        m_dict = _metrics_dict(test_y[mask_t], test_p_cal[mask_t])
        m_dict["mean_seed_std"] = round(float(np.mean(test_std[mask_t])), 4) if np.any(mask_t) else 0.0
        by_mod_cal[mod] = m_dict

    histo_mask = np.array([c == "histopathology" for c in test_cfs], dtype=bool)
    histo_derm_mask = histo_mask & np.array([m == "dermoscopic" for m in test_mods], dtype=bool)

    uncertainty = _uncertainty_analysis(test_y, test_p_cal, test_std, threshold=UNCERTAINTY_HIGH_STD_THRESHOLD)

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pretrained_weights_used": False,
        "architecture": "PyTorchDermSEResNetV2-v2.0.0 (5-Seed Multi-Scale: 3x224 + 2x256 + 4x TTA)",
        "image_sizes": [224, 256],
        "seeds": [s for _, s, _ in ENSEMBLE_SPECS],
        "ensemble_dirs": [name for name, _, _ in ENSEMBLE_SPECS],
        "use_tta": True,
        "n_val": int(len(val_y)),
        "n_locked_test": int(len(test_y)),
        "val_fitted_modality_temperatures": mod_temps,
        "per_seed_locked_test_raw": per_seed_summary,
        "overall_raw": overall_raw,
        "overall_per_modality_calibrated": overall_cal,
        "by_modality_calibrated": by_mod_cal,
        "histopathology_confirmed_all": _metrics_dict(test_y[histo_mask], test_p_cal[histo_mask]),
        "histopathology_confirmed_dermoscopic": _metrics_dict(test_y[histo_derm_mask], test_p_cal[histo_derm_mask]),
        "ensemble_uncertainty": uncertainty,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    out_path = Path("docs/results/vision/vision_5seed_multiscale_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved 5-seed multi-scale benchmark to {out_path} in {report['elapsed_seconds']}s", flush=True)
    print(
        f"OVERALL CAL: AUROC={overall_cal['auroc']:.4f} "
        f"(95% CI [{overall_cal['auroc_ci95']['ci95_low']:.4f}, {overall_cal['auroc_ci95']['ci95_high']:.4f}]) "
        f"Sens@90={overall_cal['sens_at_90_spec']:.4f} ECE={overall_cal['ece']:.4f} Brier={overall_cal['brier']:.4f}",
        flush=True,
    )
    print(
        f"BY MODALITY: derm={by_mod_cal['dermoscopic']['auroc']:.4f} "
        f"clin={by_mod_cal['clinical']['auroc']:.4f} tbp={by_mod_cal['tbp_tile']['auroc']:.4f}",
        flush=True,
    )
    print(
        f"UNCERTAINTY: low_std_auroc={uncertainty['low_uncertainty']['auroc']} "
        f"(err={uncertainty['low_uncertainty']['error_rate']}, n={uncertainty['low_uncertainty']['n']}) vs "
        f"high_std_auroc={uncertainty['high_uncertainty']['auroc']} "
        f"(err={uncertainty['high_uncertainty']['error_rate']}, n={uncertainty['high_uncertainty']['n']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
