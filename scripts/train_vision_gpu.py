"""Train & calibrate a multi-seed from-scratch PyTorchDermResNet ensemble on CUDA GPU (Kaggle / Cloud) or CPU.

Respects the locked patient/lesion-grouped v3 splits (`configs/vision/val_v3.txt`, `configs/vision/locked_test_v3.txt`)
and saves CPU-loadable checkpoints with SHA-256 verification and `val_v3`-fitted per-modality temperatures.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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


def _metrics_dict(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return {"n": int(len(y)), "malignant": int(np.sum(y == 1)), "benign": int(np.sum(y == 0)), "status": "insufficient_classes"}
    ece, _ = compute_ece_and_reliability(y, p)
    return {
        "n": int(len(y)),
        "malignant": int(np.sum(y == 1)),
        "benign": int(np.sum(y == 0)),
        "auroc": round(float(compute_auroc(y, p)), 4),
        "sens_at_90_spec": round(float(compute_sensitivity_at_specificity(y, p, target_spec=0.90)), 4),
        "ece": round(float(ece), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
    }


def _row_to_record(row: dict[str, Any], mod_hint: str, split: str) -> DermCatalogRecord:
    rid = str(row.get("i", ""))
    return DermCatalogRecord(
        image_id=rid,
        dataset=str(row.get("d") or "derm_corpus"),
        modality=str(row.get("mod") or mod_hint),
        thumb_path=str(row.get("t") or ""),
        raw_path=str(row.get("f") or ""),
        skin_tone=str(row.get("st") or "UNKNOWN"),
        diagnosis_raw=str(row.get("dx") or "") or None,
        malignancy_group=str(row.get("mg") or "unknown").lower(),
        body_site=str(row.get("bs") or "") or None,
        license_name=str(row.get("l") or "RESEARCH"),
        quality_ok=True,
        confirmation=str(row.get("cf") or "") or None,
        group_key=str(row.get("g") or rid),
        split_assignment=split,
    )


def _load_samples_for_split(
    records: list[DermCatalogRecord],
    thumbs_root: Path,
    raw_root: Path | None,
    image_size: int,
    *,
    max_items: int | None = None,
) -> list[tuple[np.ndarray, int, DermCatalogRecord]]:
    samples: list[tuple[np.ndarray, int, DermCatalogRecord]] = []
    for rec in records:
        if max_items is not None and len(samples) >= max_items:
            break
        img_path: Path | None = None
        if raw_root is not None and rec.raw_path:
            cand = raw_root / rec.raw_path
            if cand.exists():
                img_path = cand
        if img_path is None and rec.thumb_path:
            cand = thumbs_root / rec.thumb_path
            if cand.exists():
                img_path = cand
        if img_path is None:
            continue
        try:
            with Image.open(img_path) as im:
                rgb = im.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
                arr = np.asarray(rgb, dtype=np.uint8)
        except Exception:
            continue
        if rec.malignancy_group == "malignant":
            lbl = 1
        elif rec.malignancy_group == "benign":
            lbl = 0
        else:
            lbl = -1
        samples.append((arr, lbl, rec))
    return samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train from-scratch multi-seed PyTorchDermResNet on CUDA GPU or CPU.")
    p.add_argument("--meta-dir", type=Path, default=Path("var/cache/derm/meta"))
    p.add_argument("--thumbs-dir", type=Path, default=Path("var/cache/derm/thumbs"))
    p.add_argument("--raw-dir", type=Path, default=Path("var/cache/derm/raw"))
    p.add_argument("--image-size", type=int, default=224, help="Input image resolution (e.g., 96, 128, or 224 on GPU).")
    p.add_argument("--architecture", type=str, default="resnet_v2_se", choices=["resnet_v1", "resnet_v2_se"])
    p.add_argument("--tta", action="store_true", help="Enable 4x Test-Time Augmentation during validation calibration & test eval.")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--body-site-dropout", type=float, default=0.40)
    p.add_argument("--max-train-samples", type=int, default=40000)
    p.add_argument("--max-samples-per-epoch", type=int, default=24000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out-models-dir", type=Path, default=Path("models"))
    p.add_argument("--out-report", type=Path, default=Path("docs/results/vision/vision_gpu_benchmark.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    val_ids = {x.strip() for x in Path("configs/vision/val_v3.txt").read_text(encoding="utf-8").splitlines() if x.strip()}
    test_ids = {x.strip() for x in Path("configs/vision/locked_test_v3.txt").read_text(encoding="utf-8").splitlines() if x.strip()}
    lesion_groups_path = Path("configs/vision/lesion_groups_v3.json")
    heldout_groups: set[str] = set()
    if lesion_groups_path.exists():
        lg = json.loads(lesion_groups_path.read_text(encoding="utf-8"))
        id_to_group = lg.get("id_to_group", lg) if isinstance(lg, dict) else {}
        if isinstance(id_to_group, dict):
            for hid in val_ids | test_ids:
                g = id_to_group.get(hid)
                if g:
                    heldout_groups.add(str(g))

    train_recs: list[DermCatalogRecord] = []
    val_recs: list[DermCatalogRecord] = []
    test_recs: list[DermCatalogRecord] = []

    for p in sorted(args.meta_dir.glob("manifest-*.json*")):
        mod_hint = _infer_modality_from_filename(p)
        payload = _open_json_auto(p)
        items = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            continue
        for row in items:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("i", ""))
            if not rid:
                continue
            if rid in val_ids:
                val_recs.append(_row_to_record(row, mod_hint, "val"))
            elif rid in test_ids:
                test_recs.append(_row_to_record(row, mod_hint, "test"))
            else:
                grp = str(row.get("g") or rid)
                if grp in heldout_groups:
                    continue
                train_recs.append(_row_to_record(row, mod_hint, "train"))

    raw_root = args.raw_dir if args.raw_dir.exists() else None
    print(f"Loading cached images at {args.image_size}x{args.image_size} (raw_root={raw_root})...", flush=True)
    train_samples = _load_samples_for_split(train_recs, args.thumbs_dir, raw_root, args.image_size, max_items=args.max_train_samples)
    val_samples = _load_samples_for_split(val_recs, args.thumbs_dir, raw_root, args.image_size)
    test_samples = _load_samples_for_split(test_recs, args.thumbs_dir, raw_root, args.image_size)
    print(f"Loaded train={len(train_samples)} val={len(val_samples)} locked_test={len(test_samples)}", flush=True)
    if not train_samples or not val_samples:
        raise RuntimeError("No cached dermatology images found — sync metadata & thumbnails first.")

    trained_models: list[PyTorchDermResNet] = []
    saved_dirs: list[str] = []
    for idx, sd in enumerate(seeds):
        model = PyTorchDermResNet(
            image_size=args.image_size,
            include_metadata=True,
            metadata_mode="full",
            body_site_dropout=args.body_site_dropout,
            seed=sd,
            device=args.device,
            architecture_variant=args.architecture,
        )
        model.fit(
            train_samples,
            val_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            max_samples_per_epoch=args.max_samples_per_epoch,
            use_amp=True,
        )
        dir_name = f"derm_cnn_bsdrop40_seed{sd}" if idx == 0 else f"derm_cnn_seed{sd}"
        out_dir = args.out_models_dir / dir_name
        model.save(out_dir)
        saved_dirs.append(str(out_dir))
        trained_models.append(model)
        print(f"Saved {out_dir} (best_val_auroc={model.train_history.get('best_val_auroc')})", flush=True)

    # Fit per-modality calibration strictly on val_v3
    def _predict_ens(samples: list[tuple[np.ndarray, int, DermCatalogRecord]]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        labeled = [(a, lbl, r) for a, lbl, r in samples if lbl in (0, 1)]
        if not labeled:
            return np.array([]), np.array([]), [], []
        probs_per_model = [m.predict_batch(labeled, batch_size=args.batch_size, tta=args.tta)[0] for m in trained_models]
        mean_p = np.mean(np.stack(probs_per_model, axis=0), axis=0)
        ys = np.array([lbl for _, lbl, _ in labeled], dtype=np.int64)
        mods = [r.modality for _, _, r in labeled]
        cfs = [str(r.confirmation or "unknown").lower() for _, _, r in labeled]
        return mean_p.astype(np.float64), ys, mods, cfs

    val_p, val_y, val_mods, _ = _predict_ens(val_samples)
    test_p, test_y, test_mods, test_cfs = _predict_ens(test_samples)

    mod_temps: dict[str, float] = {}
    for mod in ("dermoscopic", "clinical", "tbp_tile"):
        mask_v = np.array([m == mod for m in val_mods], dtype=bool)
        mod_temps[mod] = _fit_temperature_nll(val_p[mask_v], val_y[mask_v]) if int(np.sum(mask_v)) >= 20 else 1.0

    cal_payload = {
        "fitted_on": "val_v3",
        "architecture_variant": args.architecture,
        "image_size": args.image_size,
        "use_tta": bool(args.tta),
        "seeds": seeds,
        "modality_temperatures": mod_temps,
        "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (args.out_models_dir / "derm_ensemble_calibration.json").write_text(json.dumps(cal_payload, indent=2) + "\n", encoding="utf-8")

    test_p_cal = np.array(
        [float(_sigmoid(_logit(np.array([p])) / mod_temps.get(mod, 1.0))[0]) for p, mod in zip(test_p, test_mods, strict=True)],
        dtype=np.float64,
    )
    by_mod_cal = {
        mod: _metrics_dict(
            test_y[np.array([m == mod for m in test_mods], dtype=bool)],
            test_p_cal[np.array([m == mod for m in test_mods], dtype=bool)],
        )
        for mod in ("dermoscopic", "clinical", "tbp_tile")
    }
    histo_mask = np.array([c == "histopathology" for c in test_cfs], dtype=bool)
    histo_derm_mask = histo_mask & np.array([m == "dermoscopic" for m in test_mods], dtype=bool)

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pretrained_weights_used": False,
        "image_size": args.image_size,
        "seeds": seeds,
        "saved_dirs": saved_dirs,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_locked_test": len(test_samples),
        "val_fitted_modality_temperatures": mod_temps,
        "overall_raw": _metrics_dict(test_y, test_p),
        "overall_per_modality_calibrated": _metrics_dict(test_y, test_p_cal),
        "by_modality_calibrated": by_mod_cal,
        "histopathology_confirmed_all": _metrics_dict(test_y[histo_mask], test_p_cal[histo_mask]),
        "histopathology_confirmed_dermoscopic": _metrics_dict(test_y[histo_derm_mask], test_p_cal[histo_derm_mask]),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
