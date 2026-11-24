"""Evaluate the served 9-Seed Cross-Architecture Vision Ensemble on SHA-256-deduplicated `locked_test_v4` (n=1,611)."""

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
from scripts.train_vision_gpu import (
    _load_samples_for_split,
    _logit,
    _metrics_dict,
    _row_to_record,
    _sigmoid,
)

CONVNEXT_DIRS: tuple[str, ...] = ("derm_convnext_seed10", "derm_convnext_seed11")


def run_eval_v4(
    *,
    meta_dir: Path = Path("var/cache/derm/meta"),
    thumbs_dir: Path = Path("var/cache/derm/thumbs"),
    raw_dir: Path = Path("var/cache/derm/raw"),
    models_dir: Path = Path("models"),
    test_v4_path: Path = Path("configs/vision/locked_test_v4.txt"),
    out_json: Path = Path("docs/results/vision/vision_9seed_locked_test_v4_benchmark.json"),
    w_convnext: float = 0.05,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.set_num_threads(6)

    cal = json.loads((models_dir / "derm_ensemble_calibration.json").read_text(encoding="utf-8"))
    gen_dirs: list[str] = list(cal["generalist_dirs"])
    spec_dirs: list[str] = list(cal["clinical_specialist_dirs"])
    mod_alpha: dict[str, float] = {k: float(v) for k, v in cal["modality_specialist_weights"].items()}
    mod_temp: dict[str, float] = {k: float(v) for k, v in cal["modality_temperatures"].items()}
    std_thresh = float(cal.get("uncertainty_high_std_threshold", 0.12))

    test_ids = {x.strip() for x in test_v4_path.read_text(encoding="utf-8").splitlines() if x.strip()}
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
            if rid in test_ids:
                test_recs.append(_row_to_record(row, mod_hint, "test_v4"))

    raw_root = raw_dir if raw_dir.exists() else None
    samples_224 = _load_samples_for_split(test_recs, thumbs_dir, raw_root, 224, lazy_load=True)
    labeled_224 = [(p, lbl, r) for p, lbl, r in samples_224 if lbl in (0, 1)]
    y = np.array([lbl for _, lbl, _ in labeled_224], dtype=np.int64)
    mods = np.array([r.modality for _, _, r in labeled_224])
    print(f"Evaluating 9-seed cross-arch ensemble on locked_test_v4 (n={len(labeled_224)})...", flush=True)

    def _predict_group(dirs: list[str] | tuple[str, ...]) -> list[np.ndarray]:
        preds: list[np.ndarray] = []
        for d in dirs:
            m = PyTorchDermResNet.load(models_dir / d, device="cpu")
            p_mal, _, _ = m.predict_batch(labeled_224, batch_size=64, tta=True)
            preds.append(np.asarray(p_mal, dtype=np.float64))
            print(f"  done {d} ({m.image_size}px, {m.architecture_variant})", flush=True)
        return preds

    gen_preds = _predict_group(gen_dirs)
    spec_preds = _predict_group(spec_dirs)
    cx_preds = _predict_group(CONVNEXT_DIRS)

    p_gen = np.mean(np.stack(gen_preds, axis=0), axis=0)
    p_spec = np.mean(np.stack(spec_preds, axis=0), axis=0)
    p_cx = np.mean(np.stack(cx_preds, axis=0), axis=0)
    ens_std = np.std(np.stack(gen_preds + spec_preds + cx_preds, axis=0), axis=0)

    p_cal = np.zeros_like(p_gen)
    for idx, mod in enumerate(mods):
        alpha = mod_alpha.get(str(mod), 0.0)
        p_se = (1.0 - alpha) * p_gen[idx] + alpha * p_spec[idx]
        p_blend = (1.0 - w_convnext) * p_se + w_convnext * p_cx[idx]
        temp = mod_temp.get(str(mod), 1.0)
        p_cal[idx] = float(_sigmoid(_logit(np.array([p_blend])) / temp)[0])

    low_unc_mask = ens_std <= std_thresh
    by_mod = {
        mod: _metrics_dict(y[mods == mod], p_cal[mods == mod])
        for mod in ("dermoscopic", "clinical", "tbp_tile")
    }

    report: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "split": str(test_v4_path),
        "pretrained_weights_used": False,
        "exact_sha256_duplicates_vs_train": 0,
        "w_convnext": w_convnext,
        "n_locked_test_v4": int(len(y)),
        "overall_calibrated": _metrics_dict(y, p_cal),
        "by_modality_calibrated": by_mod,
        "low_uncertainty_subset": {
            "threshold_std": std_thresh,
            **_metrics_dict(y[low_unc_mask], p_cal[low_unc_mask]),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_eval_v4(), indent=2))
