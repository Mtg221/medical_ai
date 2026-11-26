"""Vision upgrades audit: val-only per-modality temperature calibration, 3-seed ensemble, and biopsy-confirmed (histopathology) evaluation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medresearch_platform.vision.catalog import _infer_modality_from_filename, _open_json_auto
from medresearch_platform.vision.cnn_torch import PyTorchDermResNet
from medresearch_platform.vision.eval_vision import (
    compute_auroc,
    compute_ece_and_reliability,
    compute_sensitivity_at_specificity,
)


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _fit_temperature_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    logits = _logit(probs)
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.35, 3.0, 54):
        p_cal = _sigmoid(logits / float(t))
        nll = float(
            -np.mean(
                labels * np.log(np.clip(p_cal, 1e-12, 1.0))
                + (1.0 - labels) * np.log(np.clip(1.0 - p_cal, 1e-12, 1.0))
            )
        )
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return round(best_t, 4)


def _load_manifest_records(meta_dir: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    rec_by_id: dict[str, dict[str, Any]] = {}
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
            if rid in wanted:
                row_copy = dict(row)
                row_copy.setdefault("mod", mod_hint)
                rec_by_id[rid] = row_copy
    return rec_by_id


def _score_split(
    models: list[PyTorchDermResNet],
    ids: list[str],
    rec_by_id: dict[str, dict[str, Any]],
    thumbs_root: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    probs_list: list[float] = []
    labels_list: list[int] = []
    mods: list[str] = []
    cfs: list[str] = []
    for iid in ids:
        rec = rec_by_id.get(iid)
        if rec is None:
            continue
        mg = str(rec.get("mg", "")).lower()
        if mg not in ("malignant", "benign"):
            continue
        thumb_rel = str(rec.get("t", ""))
        img_path = thumbs_root / thumb_rel
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            mod = str(rec.get("mod", "dermoscopic"))
            bs = str(rec.get("bs") or "other")
            preds = [
                float(m.predict_image(rgb, modality_hint=mod, body_site=bs).malignancy_probability)
                for m in models
            ]
        probs_list.append(float(np.mean(preds)))
        labels_list.append(1 if mg == "malignant" else 0)
        mods.append(mod)
        cfs.append(str(rec.get("cf") or "unknown").lower())
    return np.array(probs_list, dtype=np.float64), np.array(labels_list, dtype=np.int64), mods, cfs


def _metrics_dict(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return {
            "n": int(len(y)),
            "malignant": int(np.sum(y == 1)),
            "benign": int(np.sum(y == 0)),
            "status": "insufficient_classes",
        }
    auroc = compute_auroc(y, p)
    sens90 = compute_sensitivity_at_specificity(y, p, target_spec=0.90)
    ece, _bins = compute_ece_and_reliability(y, p)
    brier = float(np.mean((p - y) ** 2))
    return {
        "n": int(len(y)),
        "malignant": int(np.sum(y == 1)),
        "benign": int(np.sum(y == 0)),
        "auroc": round(float(auroc), 4),
        "sens_at_90_spec": round(float(sens90), 4),
        "ece": round(float(ece), 4),
        "brier": round(brier, 4),
    }


def main() -> None:
    t0 = time.perf_counter()
    meta_dir = Path("var/cache/derm/meta")
    thumbs_root = Path("var/cache/derm/thumbs")
    val_ids = [x.strip() for x in Path("configs/vision/val_v3.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    test_ids = [x.strip() for x in Path("configs/vision/locked_test_v3.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    rec_by_id = _load_manifest_records(meta_dir, set(val_ids) | set(test_ids))

    m1 = PyTorchDermResNet.load(Path("models/derm_cnn_bsdrop40_seed1"))
    m2 = PyTorchDermResNet.load(Path("models/derm_cnn_seed2"))
    m3 = PyTorchDermResNet.load(Path("models/derm_cnn_seed3"))

    out_report: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "post-hoc (test seen) — val_v3-only per-modality temperature calibration + biopsy-confirmed (histopathology) audit on locked_test_v3",
        "models": {},
    }

    for label, model_list in (
        ("bsdrop40_1seed", [m1]),
        ("ensemble_3seed_bsdrop40_s2_s3", [m1, m2, m3]),
    ):
        val_p, val_y, val_mods, _ = _score_split(model_list, val_ids, rec_by_id, thumbs_root)
        test_p, test_y, test_mods, test_cfs = _score_split(model_list, test_ids, rec_by_id, thumbs_root)

        mod_temps: dict[str, float] = {}
        for mod in ("dermoscopic", "clinical", "tbp_tile"):
            mask_v = np.array([m == mod for m in val_mods], dtype=bool)
            if int(np.sum(mask_v)) >= 20:
                mod_temps[mod] = _fit_temperature_nll(val_p[mask_v], val_y[mask_v])
            else:
                mod_temps[mod] = 1.0

        test_p_cal = np.array(
            [
                float(_sigmoid(_logit(np.array([p])) / mod_temps.get(mod, 1.0))[0])
                for p, mod in zip(test_p, test_mods, strict=True)
            ],
            dtype=np.float64,
        )

        by_mod_raw: dict[str, Any] = {}
        by_mod_cal: dict[str, Any] = {}
        for mod in ("dermoscopic", "clinical", "tbp_tile"):
            mask_t = np.array([m == mod for m in test_mods], dtype=bool)
            by_mod_raw[mod] = _metrics_dict(test_y[mask_t], test_p[mask_t])
            by_mod_cal[mod] = _metrics_dict(test_y[mask_t], test_p_cal[mask_t])

        histo_mask = np.array(["histopath" in c for c in test_cfs], dtype=bool)
        histo_derm_mask = histo_mask & np.array([m == "dermoscopic" for m in test_mods], dtype=bool)
        out_report["models"][label] = {
            "val_fitted_modality_temperatures": mod_temps,
            "overall_raw": _metrics_dict(test_y, test_p),
            "overall_per_modality_calibrated": _metrics_dict(test_y, test_p_cal),
            "by_modality_raw": by_mod_raw,
            "by_modality_calibrated": by_mod_cal,
            "histopathology_confirmed_all": _metrics_dict(test_y[histo_mask], test_p_cal[histo_mask]),
            "histopathology_confirmed_dermoscopic": _metrics_dict(test_y[histo_derm_mask], test_p_cal[histo_derm_mask]),
        }
        print(
            f"[{label}] overall raw AUROC={out_report['models'][label]['overall_raw']['auroc']:.4f} "
            f"ECE={out_report['models'][label]['overall_raw']['ece']:.4f} -> "
            f"cal ECE={out_report['models'][label]['overall_per_modality_calibrated']['ece']:.4f} | "
            f"histo_derm AUROC={out_report['models'][label]['histopathology_confirmed_dermoscopic'].get('auroc')} "
            f"(n={out_report['models'][label]['histopathology_confirmed_dermoscopic'].get('n')})",
            flush=True,
        )

    out_report["elapsed_s"] = round(time.perf_counter() - t0, 2)
    dst = Path("docs/results/vision/vision_upgrades_audit.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out_report, indent=2), encoding="utf-8")
    print(f"Wrote {dst} in {out_report['elapsed_s']}s")


if __name__ == "__main__":
    main()
