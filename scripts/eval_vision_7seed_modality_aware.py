"""Evaluate the 7-Seed Modality-Aware Multi-Scale Vision SE-ResNetV2 Ensemble
(3x224 + 2x256 Generalists + 2x256 Clinical-Photo Specialists, 4x TTA)
with val_v3-only modality specialist weighting, per-modality temperature scaling,
lesion-cluster bootstrap 95% CIs, histopathology subgroup audit, and uncertainty guardrail analysis.
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
from scripts.eval_vision_5seed_multiscale import (
    UNCERTAINTY_HIGH_STD_THRESHOLD,
    _uncertainty_analysis,
)
from scripts.train_vision_gpu import (
    _fit_temperature_nll,
    _load_samples_for_split,
    _logit,
    _metrics_dict,
    _row_to_record,
    _sigmoid,
)

GENERALIST_SPECS: list[tuple[str, int, int]] = [
    ("derm_cnn_bsdrop40_seed1", 1, 224),
    ("derm_cnn_seed2", 2, 224),
    ("derm_cnn_seed3", 3, 224),
    ("derm_cnn_seed4", 4, 256),
    ("derm_cnn_seed5", 5, 256),
]

CLINICAL_SPECIALIST_SPECS: list[tuple[str, int, int]] = [
    ("derm_cnn_clinical_seed6", 6, 256),
    ("derm_cnn_clinical_seed7", 7, 256),
]


def _fit_modality_alpha_on_val(
    p_gen: np.ndarray,
    p_spec: np.ndarray,
    y: np.ndarray,
) -> float:
    """Select specialist blend weight alpha in [0, 1] strictly on val_v3 maximizing AUROC (tie-break: NLL)."""
    best_alpha = 0.0
    best_score = (-1.0, float("inf"))
    for alpha in np.linspace(0.0, 1.0, 21):
        p = (1.0 - alpha) * p_gen + alpha * p_spec
        auroc = float(compute_auroc(y, p))
        nll = float(
            -np.mean(
                y * np.log(np.clip(p, 1e-12, 1.0))
                + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-12, 1.0))
            )
        )
        if (auroc > best_score[0] + 1e-4) or (abs(auroc - best_score[0]) <= 1e-4 and nll < best_score[1]):
            best_score = (auroc, nll)
            best_alpha = round(float(alpha), 2)
    return best_alpha


def main() -> None:
    t0 = time.perf_counter()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Evaluating 7-seed modality-aware vision ensemble on device={device} (4x TTA)", flush=True)

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

    val_by_size: dict[int, list[tuple[np.ndarray, int, DermCatalogRecord]]] = {}
    test_by_size: dict[int, list[tuple[np.ndarray, int, DermCatalogRecord]]] = {}
    for sz in (224, 256):
        val_by_size[sz] = [s for s in _load_samples_for_split(val_recs, thumbs_dir, raw_root, sz) if s[1] in (0, 1)]
        test_by_size[sz] = [s for s in _load_samples_for_split(test_recs, thumbs_dir, raw_root, sz) if s[1] in (0, 1)]

    assert [r.image_id for _, _, r in val_by_size[224]] == [r.image_id for _, _, r in val_by_size[256]]
    assert [r.image_id for _, _, r in test_by_size[224]] == [r.image_id for _, _, r in test_by_size[256]]

    val_y = np.array([lbl for _, lbl, _ in val_by_size[224]], dtype=np.int64)
    val_mods = [r.modality for _, _, r in val_by_size[224]]

    test_y = np.array([lbl for _, lbl, _ in test_by_size[224]], dtype=np.int64)
    test_mods = [r.modality for _, _, r in test_by_size[224]]
    test_cfs = [str(r.confirmation or "unknown").lower() for _, _, r in test_by_size[224]]
    test_groups = [r.group_key for _, _, r in test_by_size[224]]

    val_gen_list: list[np.ndarray] = []
    test_gen_list: list[np.ndarray] = []
    val_spec_list: list[np.ndarray] = []
    test_spec_list: list[np.ndarray] = []
    per_seed_summary: dict[str, Any] = {}

    for group_name, specs, v_bucket, t_bucket in (
        ("generalist", GENERALIST_SPECS, val_gen_list, test_gen_list),
        ("clinical_specialist", CLINICAL_SPECIALIST_SPECS, val_spec_list, test_spec_list),
    ):
        for dir_name, seed, img_size in specs:
            model = PyTorchDermResNet.load(models_root / dir_name, device=device)
            tm = time.perf_counter()
            vp = model.predict_batch(val_by_size[img_size], batch_size=64, tta=True)[0].astype(np.float64)
            tp = model.predict_batch(test_by_size[img_size], batch_size=64, tta=True)[0].astype(np.float64)
            v_bucket.append(vp)
            t_bucket.append(tp)
            m_all = _metrics_dict(test_y, tp)
            c_mask = np.array([m == "clinical" for m in test_mods], dtype=bool)
            m_clin = _metrics_dict(test_y[c_mask], tp[c_mask])
            per_seed_summary[dir_name] = {
                "role": group_name,
                "seed": seed,
                "image_size": img_size,
                "raw_locked_test_overall": m_all,
                "raw_locked_test_clinical": m_clin,
                "inference_seconds": round(time.perf_counter() - tm, 2),
            }
            print(
                f"  [{dir_name} ({group_name}, {img_size}px)] overall={m_all['auroc']:.4f} "
                f"clin={m_clin['auroc']:.4f} (sens@90={m_clin['sens_at_90_spec']:.4f}) ({per_seed_summary[dir_name]['inference_seconds']}s)",
                flush=True,
            )

    val_gen_mean = np.mean(np.stack(val_gen_list, axis=0), axis=0)
    val_spec_mean = np.mean(np.stack(val_spec_list, axis=0), axis=0)
    test_gen_stack = np.stack(test_gen_list, axis=0)
    test_spec_stack = np.stack(test_spec_list, axis=0)

    # Fit per-modality specialist weight alpha & temperature strictly on val_v3
    mod_alphas: dict[str, float] = {}
    mod_temps: dict[str, float] = {}
    for mod in ("dermoscopic", "clinical", "tbp_tile"):
        mask_v = np.array([m == mod for m in val_mods], dtype=bool)
        if int(np.sum(mask_v)) >= 20:
            alpha = _fit_modality_alpha_on_val(val_gen_mean[mask_v], val_spec_mean[mask_v], val_y[mask_v])
            p_val_blend = (1.0 - alpha) * val_gen_mean[mask_v] + alpha * val_spec_mean[mask_v]
            temp = _fit_temperature_nll(p_val_blend, val_y[mask_v])
        else:
            alpha, temp = 0.0, 1.0
        mod_alphas[mod] = alpha
        mod_temps[mod] = temp
    print(f"val_v3 fitted specialist alphas: {mod_alphas} | temps: {mod_temps}", flush=True)

    # Compute modality-aware blended probabilities and weighted cross-seed std on locked_test_v3
    test_p_raw = np.zeros(len(test_y), dtype=np.float64)
    test_p_cal = np.zeros(len(test_y), dtype=np.float64)
    test_std = np.zeros(len(test_y), dtype=np.float64)

    n_gen = len(GENERALIST_SPECS)
    n_spec = len(CLINICAL_SPECIALIST_SPECS)
    all_test_stack = np.concatenate([test_gen_stack, test_spec_stack], axis=0)  # [7, N]

    for i, mod in enumerate(test_mods):
        alpha = mod_alphas.get(mod, 0.0)
        w_gen = (1.0 - alpha) / n_gen
        w_spec = alpha / n_spec
        weights = np.array([w_gen] * n_gen + [w_spec] * n_spec, dtype=np.float64)
        preds_i = all_test_stack[:, i]
        p_mean = float(np.sum(weights * preds_i))
        p_var = float(np.sum(weights * (preds_i - p_mean) ** 2))
        test_p_raw[i] = p_mean
        test_std[i] = float(np.sqrt(max(0.0, p_var)))
        test_p_cal[i] = float(_sigmoid(_logit(np.array([p_mean])) / mod_temps.get(mod, 1.0))[0])

    # Update live calibration file for serving
    cal_payload = {
        "fitted_on": "val_v3",
        "architecture_variant": "resnet_v2_se",
        "image_sizes": [224, 256],
        "generalist_dirs": [name for name, _, _ in GENERALIST_SPECS],
        "clinical_specialist_dirs": [name for name, _, _ in CLINICAL_SPECIALIST_SPECS],
        "ensemble_dirs": [name for name, _, _ in GENERALIST_SPECS + CLINICAL_SPECIALIST_SPECS],
        "use_tta": True,
        "seeds": [s for _, s, _ in GENERALIST_SPECS + CLINICAL_SPECIALIST_SPECS],
        "modality_specialist_weights": mod_alphas,
        "modality_temperatures": mod_temps,
        "uncertainty_high_std_threshold": UNCERTAINTY_HIGH_STD_THRESHOLD,
        "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (models_root / "derm_ensemble_calibration.json").write_text(
        json.dumps(cal_payload, indent=2) + "\n", encoding="utf-8"
    )

    overall_raw = _metrics_dict(test_y, test_p_raw)
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
        m_dict["specialist_alpha"] = mod_alphas.get(mod, 0.0)
        by_mod_cal[mod] = m_dict

    histo_mask = np.array([c == "histopathology" for c in test_cfs], dtype=bool)
    histo_derm_mask = histo_mask & np.array([m == "dermoscopic" for m in test_mods], dtype=bool)
    uncertainty = _uncertainty_analysis(test_y, test_p_cal, test_std, threshold=UNCERTAINTY_HIGH_STD_THRESHOLD)

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pretrained_weights_used": False,
        "architecture": "PyTorchDermSEResNetV2-v2.1.0 (7-Seed Modality-Aware: 3x224 + 2x256 + 2x256 Clinical Specialists + 4x TTA)",
        "image_sizes": [224, 256],
        "seeds": [s for _, s, _ in GENERALIST_SPECS + CLINICAL_SPECIALIST_SPECS],
        "generalist_dirs": [name for name, _, _ in GENERALIST_SPECS],
        "clinical_specialist_dirs": [name for name, _, _ in CLINICAL_SPECIALIST_SPECS],
        "use_tta": True,
        "n_val": int(len(val_y)),
        "n_locked_test": int(len(test_y)),
        "val_fitted_modality_specialist_weights": mod_alphas,
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

    out_path = Path("docs/results/vision/vision_7seed_modality_aware_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved 7-seed modality-aware benchmark to {out_path} in {report['elapsed_seconds']}s", flush=True)
    print(
        f"OVERALL CAL: AUROC={overall_cal['auroc']:.4f} "
        f"(95% CI [{overall_cal['auroc_ci95']['ci95_low']:.4f}, {overall_cal['auroc_ci95']['ci95_high']:.4f}]) "
        f"Sens@90={overall_cal['sens_at_90_spec']:.4f} ECE={overall_cal['ece']:.4f} Brier={overall_cal['brier']:.4f}",
        flush=True,
    )
    print(
        f"BY MODALITY: derm={by_mod_cal['dermoscopic']['auroc']:.4f} (sens90={by_mod_cal['dermoscopic']['sens_at_90_spec']:.4f}) | "
        f"clin={by_mod_cal['clinical']['auroc']:.4f} (sens90={by_mod_cal['clinical']['sens_at_90_spec']:.4f}) | "
        f"tbp={by_mod_cal['tbp_tile']['auroc']:.4f}",
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
