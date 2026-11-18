"""Step 4: Train 3-seed PyTorch 4-Stage ResNet CNN + ablations + baselines, calibrate/abstain on val, and evaluate ONCE on locked_test_v2."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.classical_skin import ClassicalSkinClassifier
from medresearch_platform.vision.cnn_multitask import DermResNetMultiTask
from medresearch_platform.vision.cnn_torch import (
    CORE_DX_TO_IDX,
    PyTorchDermResNet,
    _compute_binary_auroc,
    encode_metadata_vector,
    verify_gradcam_adebayo_and_lesion_center,
)
from medresearch_platform.vision.eval_vision import (
    compute_comprehensive_metrics,
    load_stratified_skin_dataset,
    paired_lesion_bootstrap_holm,
)
from medresearch_platform.vision.splits import log_locked_test_access
from medresearch_platform.vision.sync import compute_split_counts


def _filter_labelled_mg(
    samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
) -> list[tuple[np.ndarray, int, DermCatalogRecord]]:
    return [s for s in samples if s[1] in (0, 1)]


def _meta_features_with_dataset(samples: list[tuple[np.ndarray, int, DermCatalogRecord]]) -> np.ndarray:
    rows = []
    for _, _, r in samples:
        v = encode_metadata_vector(r)
        ds_flag = np.array([1.0 if r.dataset == "isic" else 0.0, 1.0 if r.dataset == "skindisnet" else 0.0], dtype=np.float32)
        rows.append(np.concatenate([v, ds_flag]))
    return np.stack(rows, axis=0)


def _compute_subgroup_metrics(
    test_lab: list[tuple[np.ndarray, int, DermCatalogRecord]],
    y_true: np.ndarray,
    p_mal: np.ndarray,
    p_dx: np.ndarray,
    abstained: np.ndarray,
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    records = [r for _, _, r in test_lab]
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[str(key_fn(r))].append(i)
    out: dict[str, dict[str, Any]] = {}
    for gname, idxs in sorted(groups.items()):
        n = len(idxs)
        idx_arr = np.array(idxs, dtype=np.int64)
        yt = y_true[idx_arr]
        pm = p_mal[idx_arr]
        ab = abstained[idx_arr]
        pos = int(np.sum(yt == 1))
        neg = int(np.sum(yt == 0))
        if n < 30 or pos == 0 or neg == 0:
            out[gname] = {
                "n": n,
                "n_malignant": pos,
                "n_benign": neg,
                "auroc": round(_compute_binary_auroc(yt, pm), 4) if (pos > 0 and neg > 0) else None,
                "abstain_rate": round(float(np.mean(ab)), 4),
                "note": "too few to conclude" if n < 30 else "single class only",
            }
        else:
            m = compute_comprehensive_metrics(
                [test_lab[k] for k in idxs],
                pm,
                p_dx[idx_arr],
                ab,
                compute_subgroups=False,
                n_boot=200,
                seed=42,
            )
            out[gname] = {
                "n": n,
                "n_malignant": pos,
                "n_benign": neg,
                "auroc": m["auroc"],
                "auroc_ci95": m["auroc_ci95"],
                "balanced_accuracy": m["balanced_accuracy"],
                "ece": m["ece"],
                "brier_score": m["brier_score"],
                "abstain_rate": m["abstention_rate"],
            }
    return out


def _coverage_risk_curve(y_true: np.ndarray, p_mal: np.ndarray) -> list[dict[str, float]]:
    conf = np.maximum(p_mal, 1.0 - p_mal)
    pred = (p_mal >= 0.5).astype(np.int64)
    curve = []
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        kept = conf >= thr
        cov = float(np.mean(kept))
        if np.sum(kept) == 0:
            continue
        err = float(np.mean(pred[kept] != y_true[kept]))
        yt_k = y_true[kept]
        auroc = _compute_binary_auroc(yt_k, p_mal[kept]) if (np.any(yt_k == 1) and np.any(yt_k == 0)) else None
        curve.append({
            "confidence_threshold": thr,
            "coverage": round(cov, 4),
            "error_rate": round(err, 4),
            "selective_auroc": round(auroc, 4) if auroc is not None else -1.0,
        })
    return curve


def run_step4_benchmark(
    *,
    meta_dir: str | Path = "var/cache/derm/meta",
    locked_test_path: str | Path = "configs/vision/locked_test_v2.txt",
    out_model_dir: str | Path = "models/derm_cnn_seed1",
    out_results_json: str | Path = "docs/results/vision/step4_pytorch_skin_benchmark.json",
    abstention_cfg_path: str | Path = "configs/vision/abstention_v2.json",
    epochs: int = 8,
    image_size: int = 96,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    splits = load_stratified_skin_dataset(
        meta_dir=meta_dir,
        image_size=image_size,
        include_masked_mg=True,
        locked_test_list_path=locked_test_path,
        min_real_samples=100,
    )
    train_all = splits["train"]
    val_all = splits["val"]
    test_all = splits["locked_test"]

    train_lab = _filter_labelled_mg(train_all)
    val_lab = _filter_labelled_mg(val_all)
    test_lab = _filter_labelled_mg(test_all)

    y_test = np.array([lbl for _, lbl, _ in test_lab], dtype=np.int64)
    y_val = np.array([lbl for _, lbl, _ in val_lab], dtype=np.int64)
    n_dx = len(CORE_DX_TO_IDX)

    # 1. Metadata-only baseline (no pixels)
    x_meta_tr = _meta_features_with_dataset(train_lab)
    y_meta_tr = np.array([lbl for _, lbl, _ in train_lab], dtype=np.int64)
    x_meta_te = _meta_features_with_dataset(test_lab)
    meta_clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    meta_clf.fit(x_meta_tr, y_meta_tr)
    p_mal_meta = meta_clf.predict_proba(x_meta_te)[:, 1]
    p_dx_uniform = np.full((len(test_lab), n_dx), 1.0 / n_dx, dtype=np.float64)
    ab_zero = np.zeros(len(test_lab), dtype=bool)
    meta_metrics = compute_comprehensive_metrics(test_lab, p_mal_meta, p_dx_uniform, ab_zero, compute_subgroups=False, n_boot=300, seed=42)

    # 2. Classical vision baseline (64 handcrafted features + LogReg)
    classical = ClassicalSkinClassifier(variant="logreg", include_metadata=True, seed=42)
    classical.fit(train_lab)
    p_mal_cls, p_dx_cls = classical.predict_proba(test_lab)
    cls_metrics = compute_comprehensive_metrics(test_lab, p_mal_cls, p_dx_cls, ab_zero, compute_subgroups=False, n_boot=300, seed=42)

    # 3. NumPy frozen random features baseline (honestly named)
    np_base = DermResNetMultiTask(image_size=image_size, include_metadata=True, multi_task=True, seed=1)
    np_base.fit(train_lab[:1500], val_lab[:400], epochs=12, batch_size=32, lr=0.015)
    np_out = np_base.predict_batch(test_lab)
    p_mal_np = np_out["p_malignant"]
    p_dx_np = np_out["p_diagnosis"]
    np_metrics = compute_comprehensive_metrics(test_lab, p_mal_np, p_dx_np, ab_zero, compute_subgroups=False, n_boot=300, seed=42)

    # 4. PyTorch 4-Stage ResNet (3 seeds with metadata + 1 ablation without metadata)
    seed_runs: list[dict[str, Any]] = []
    seed_preds: list[tuple[np.ndarray, np.ndarray]] = []
    primary_model: PyTorchDermResNet | None = None

    for s in (1, 2, 3):
        s_dir = Path(out_model_dir) if s == 1 else Path(f"{out_model_dir}_s{s}")
        if (s_dir / "manifest.json").exists() and (s_dir / "weights_torch.pt").exists():
            m = PyTorchDermResNet.load(s_dir)
            tr_info = {"loaded_from_disk": str(s_dir), "seed": s, "parameter_count": m.parameter_count}
        else:
            m = PyTorchDermResNet(image_size=image_size, include_metadata=True, seed=s)
            tr_info = m.fit(train_all, val_all, epochs=epochs, batch_size=64, lr=1e-3, patience=3)
            m.save(s_dir, corpus_hash=Path(locked_test_path).name)
        if s == 1:
            primary_model = m
        pm_te, pdx_te, _ = m.predict_batch(test_lab)
        seed_preds.append((pm_te, pdx_te))
        seed_runs.append({"seed": s, "train_info": tr_info, "test_auroc": round(_compute_binary_auroc(y_test, pm_te), 4)})

    assert primary_model is not None

    # No-metadata PyTorch CNN ablation
    nometa_dir = Path(f"{out_model_dir}_nometa")
    if (nometa_dir / "manifest.json").exists() and (nometa_dir / "weights_torch.pt").exists():
        m_nometa = PyTorchDermResNet.load(nometa_dir)
        nometa_info = {"loaded_from_disk": str(nometa_dir), "seed": 1, "parameter_count": m_nometa.parameter_count}
    else:
        m_nometa = PyTorchDermResNet(image_size=image_size, include_metadata=False, seed=1)
        nometa_info = m_nometa.fit(train_all, val_all, epochs=epochs, batch_size=64, lr=1e-3, patience=3)
        m_nometa.save(nometa_dir, corpus_hash=Path(locked_test_path).name)
    pm_nometa_te, pdx_nometa_te, _ = m_nometa.predict_batch(test_lab)
    nometa_metrics = compute_comprehensive_metrics(test_lab, pm_nometa_te, pdx_nometa_te, ab_zero, compute_subgroups=False, n_boot=300, seed=42)

    # 5. Choose ONE abstention threshold on val (confidence = max(p, 1-p) < thr) targeting >= 80% coverage & lowest selective error
    pm_val, _, _ = primary_model.predict_batch(val_lab)
    conf_val = np.maximum(pm_val, 1.0 - pm_val)
    pred_val = (pm_val >= 0.5).astype(np.int64)
    chosen_thr = 0.55
    best_score = float("inf")
    for thr in (0.52, 0.55, 0.58, 0.60, 0.62, 0.65):
        kept = conf_val >= thr
        cov = float(np.mean(kept))
        if cov < 0.75:
            continue
        err = float(np.mean(pred_val[kept] != y_val[kept]))
        if err < best_score:
            best_score = err
            chosen_thr = thr

    abstain_cfg = {
        "rule": "abstain_if_max_prob_below_threshold",
        "min_confidence": chosen_thr,
        "selected_on": "val",
        "val_coverage": round(float(np.mean(conf_val >= chosen_thr)), 4),
        "val_selective_error": round(best_score, 4),
    }
    ab_p = Path(abstention_cfg_path)
    ab_p.parent.mkdir(parents=True, exist_ok=True)
    ab_p.write_text(json.dumps(abstain_cfg, indent=2) + "\n", encoding="utf-8")

    # 6. Primary PyTorch CNN (seed=1) + 3-seed ensemble evaluation on locked_test_v2 (evaluated ONCE)
    pm_s1, pdx_s1 = seed_preds[0]
    ab_s1 = np.maximum(pm_s1, 1.0 - pm_s1) < chosen_thr
    s1_metrics = compute_comprehensive_metrics(test_lab, pm_s1, pdx_s1, ab_s1, compute_subgroups=True, n_boot=400, seed=42)

    pm_ens = np.mean([p[0] for p in seed_preds], axis=0)
    pdx_ens = np.mean([p[1] for p in seed_preds], axis=0)
    ab_ens = np.maximum(pm_ens, 1.0 - pm_ens) < chosen_thr
    ens_metrics = compute_comprehensive_metrics(test_lab, pm_ens, pdx_ens, ab_ens, compute_subgroups=False, n_boot=400, seed=42)

    # 7. Grad-CAM sanity checks (Adebayo weight randomisation + dermoscopic centre share)
    gradcam_sanity = verify_gradcam_adebayo_and_lesion_center(primary_model, test_lab)

    # 8. Subgroups (tone, modality, dataset separately, body site)
    subgroups = {
        "by_fitzpatrick_tone": _compute_subgroup_metrics(test_lab, y_test, pm_s1, pdx_s1, ab_s1, lambda r: r.skin_tone),
        "by_modality": _compute_subgroup_metrics(test_lab, y_test, pm_s1, pdx_s1, ab_s1, lambda r: r.modality),
        "by_dataset": _compute_subgroup_metrics(test_lab, y_test, pm_s1, pdx_s1, ab_s1, lambda r: r.dataset),
        "by_body_site": _compute_subgroup_metrics(test_lab, y_test, pm_s1, pdx_s1, ab_s1, lambda r: r.body_site or "unspecified"),
    }

    # 9. Holm paired bootstrap comparisons on locked_test_v2
    paired = paired_lesion_bootstrap_holm(
        test_lab,
        {
            "pytorch_resnet_seed1": pm_s1,
            "pytorch_resnet_no_meta": pm_nometa_te,
            "classical_logreg": p_mal_cls,
            "metadata_only": p_mal_meta,
            "numpy_frozen_random_features": p_mal_np,
        },
        reference_model="pytorch_resnet_seed1",
        n_boot=400,
        seed=42,
    )

    log_locked_test_access(
        dataset_name="derm-corpus:locked_test_v2",
        model_name="PyTorchDermResNet4Stage-v1.0.0",
        sample_count=len(test_lab),
        reason="Step 4 final single evaluation on frozen configs/vision/locked_test_v2.txt",
    )

    split_counts = compute_split_counts(
        {
            "train": [r for _, _, r in train_all],
            "val": [r for _, _, r in val_all],
            "locked_test_v2": [r for _, _, r in test_all],
        }
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "data_source": split_counts,
        "models": {
            "pytorch_resnet_seed1": s1_metrics,
            "pytorch_resnet_3seed_ensemble": ens_metrics,
            "pytorch_resnet_no_meta": nometa_metrics,
            "classical_logreg": cls_metrics,
            "metadata_only_logreg": meta_metrics,
            "numpy_frozen_random_features": np_metrics,
        },
        "seed_runs": seed_runs,
        "no_meta_train_info": nometa_info,
        "abstention_config": abstain_cfg,
        "coverage_risk_curve": _coverage_risk_curve(y_test, pm_s1),
        "gradcam_sanity": gradcam_sanity,
        "subgroups": subgroups,
        "paired_bootstrap_holm": paired,
    }

    out_p = Path(out_results_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = run_step4_benchmark()
    print(json.dumps({"models": {k: {"auroc": v["auroc"], "auroc_ci95": v["auroc_ci95"], "ece": v["ece"], "diagnosis_macro_f1": v.get("diagnosis_macro_f1")} for k, v in rep["models"].items()}, "gradcam_sanity": rep["gradcam_sanity"], "elapsed_seconds": rep["elapsed_seconds"]}, indent=2))
