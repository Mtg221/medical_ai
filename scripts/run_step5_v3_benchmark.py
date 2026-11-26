"""Step 5: Train 3-seed PyTorchDermResNet + baselines on v3 splits, check Grad-CAM & tone shortcut, evaluate once on locked_test_v3."""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medresearch_platform.vision.catalog import DermCatalogRecord, iter_derm_catalog
from medresearch_platform.vision.classical_skin import ClassicalSkinClassifier
from medresearch_platform.vision.cnn_multitask import DermResNetMultiTask
from medresearch_platform.vision.cnn_torch import (
    PyTorchDermResNet,
    _compute_binary_auroc,
    encode_metadata_vector,
    verify_gradcam_adebayo_and_lesion_center,
)
from medresearch_platform.vision.dx_mapping import CORE_DX_CLASSES
from medresearch_platform.vision.eval_vision import (
    compute_comprehensive_metrics,
    load_stratified_skin_dataset,
    paired_lesion_bootstrap_holm,
)
from medresearch_platform.vision.splits import log_locked_test_access
from medresearch_platform.vision.sync import compute_split_counts

TONES = ("I", "II", "III", "IV", "V", "VI", "UNKNOWN")
DATASETS = ("isic", "fitzpatrick17k", "skindisnet")
ADEBAYO_PASS_THRESHOLD = 0.30


def _filter_labelled_mg(
    samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
) -> list[tuple[np.ndarray, int, DermCatalogRecord]]:
    return [s for s in samples if s[2].malignancy_group in ("benign", "malignant")]


def _fit_val_abstention(y_val: np.ndarray, p_val: np.ndarray) -> dict[str, Any]:
    conf = np.maximum(p_val, 1.0 - p_val)
    preds = (p_val >= 0.5).astype(np.int64)
    best_thr = 0.55
    best_score = -1.0
    for thr in np.linspace(0.52, 0.72, 21):
        keep = conf >= thr
        cov = float(np.mean(keep))
        if cov < 0.75:
            continue
        acc = float(np.mean(preds[keep] == y_val[keep])) if np.any(keep) else 0.0
        score = acc - 0.15 * (1.0 - cov)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    keep = conf >= best_thr
    err = float(1.0 - np.mean(preds[keep] == y_val[keep])) if np.any(keep) else 0.0
    return {
        "min_confidence": round(best_thr, 3),
        "val_coverage": round(float(np.mean(keep)), 4),
        "val_selective_error": round(err, 4),
        "tuned_on": "val_v3_only",
    }


def _coverage_risk_curve(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None
) -> list[dict[str, float]]:
    conf = np.maximum(p, 1.0 - p)
    preds = (p >= 0.5).astype(np.int64)
    w = np.ones(len(y), dtype=np.float64) if weights is None else weights
    total_w = float(np.sum(w))
    rows = []
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        keep = conf >= thr
        if not np.any(keep):
            continue
        cov = float(np.sum(w[keep]) / total_w)
        err = float(np.sum(w[keep] * (preds[keep] != y[keep])) / np.sum(w[keep]))
        rows.append({"threshold": thr, "coverage": round(cov, 4), "risk_error": round(err, 4)})
    return rows


def _compute_corpus_modality_prevalence(meta_dir: str | Path = "var/cache/derm/meta") -> dict[str, dict[str, float]]:
    """Compute realistic corpus malignancy prevalence per modality (and overall) from labelled corpus records."""
    counts: dict[str, Counter[str]] = {m: Counter() for m in ("dermoscopic", "clinical", "tbp_tile")}
    for rec in iter_derm_catalog(meta_dir, require_quality_ok=False, require_research_licence=True):
        is_allowed_fitz = rec.dataset == "fitzpatrick17k" and rec.malignancy_group in ("benign", "malignant")
        if not rec.quality_ok and not is_allowed_fitz:
            continue
        if rec.modality in counts and rec.malignancy_group in ("benign", "malignant"):
            counts[rec.modality][rec.malignancy_group] += 1
    out: dict[str, dict[str, float]] = {}
    tot_b = tot_m = 0
    for mod, c in counts.items():
        b, m = c["benign"], c["malignant"]
        tot_b += b
        tot_m += m
        tot = max(1, b + m)
        out[mod] = {"benign": b, "malignant": m, "total": b + m, "prevalence": round(m / tot, 6)}
    tot_all = max(1, tot_b + tot_m)
    out["overall"] = {"benign": tot_b, "malignant": tot_m, "total": tot_b + tot_m, "prevalence": round(tot_m / tot_all, 6)}
    return out


def _weighted_clinical_metrics(
    y: np.ndarray,
    p: np.ndarray,
    abstained: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    """Compute prevalence-reweighted PPV, NPV, ECE, and abstention rate."""
    w = weights / max(1e-12, float(np.sum(weights)))
    preds = (p >= 0.5).astype(np.int64)
    non_abs = ~abstained
    abst_rate = float(np.sum(w[abstained]))

    pos_pred = non_abs & (preds == 1)
    neg_pred = non_abs & (preds == 0)
    ppv = float(np.sum(w[pos_pred] * (y[pos_pred] == 1)) / np.sum(w[pos_pred])) if np.any(pos_pred) else None
    npv = float(np.sum(w[neg_pred] * (y[neg_pred] == 0)) / np.sum(w[neg_pred])) if np.any(neg_pred) else None

    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (p >= lo) & (p <= hi if i == 9 else p < hi)
        if np.any(mask):
            wb = float(np.sum(w[mask]))
            if wb > 0:
                mp = float(np.sum(w[mask] * p[mask]) / wb)
                acc = float(np.sum(w[mask] * y[mask]) / wb)
                ece += wb * abs(acc - mp)

    return {
        "ppv": round(ppv, 4) if ppv is not None else None,
        "npv": round(npv, 4) if npv is not None else None,
        "ece": round(float(ece), 4),
        "abstention_rate": round(abst_rate, 4),
    }


def _reweighted_view(
    samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
    p_mal: np.ndarray,
    abstained: np.ndarray,
    corpus_prev: dict[str, dict[str, float]],
) -> dict[str, Any]:
    y = np.asarray([s[1] for s in samples], dtype=np.int64)
    mods = np.asarray([s[2].modality for s in samples])

    # Per-modality weights matching each modality's corpus malignancy prevalence
    w_all = np.zeros(len(samples), dtype=np.float64)
    by_mod: dict[str, Any] = {}
    corpus_tot = max(1.0, float(corpus_prev["overall"]["total"]))
    for mod in ("dermoscopic", "clinical", "tbp_tile"):
        mm = mods == mod
        if not np.any(mm):
            continue
        p_corp_mal = float(corpus_prev[mod]["prevalence"])
        p_corp_ben = 1.0 - p_corp_mal
        n_mal_t = max(1, int(np.sum(mm & (y == 1))))
        n_ben_t = max(1, int(np.sum(mm & (y == 0))))
        mod_share_corpus = float(corpus_prev[mod]["total"]) / corpus_tot
        w_mod = np.where(y[mm] == 1, p_corp_mal / n_mal_t, p_corp_ben / n_ben_t)
        by_mod[mod] = {
            "assumed_corpus_prevalence": p_corp_mal,
            **_weighted_clinical_metrics(y[mm], p_mal[mm], abstained[mm], w_mod),
            "coverage_risk_curve": _coverage_risk_curve(y[mm], p_mal[mm], w_mod),
        }
        w_all[mm] = w_mod * mod_share_corpus

    return {
        "corpus_prevalence_source": corpus_prev,
        "by_modality": by_mod,
        "overall": {
            "assumed_corpus_prevalence": corpus_prev["overall"]["prevalence"],
            **_weighted_clinical_metrics(y, p_mal, abstained, w_all),
            "coverage_risk_curve": _coverage_risk_curve(y, p_mal, w_all),
        },
    }


def _encode_tone_collection(rec: DermCatalogRecord) -> np.ndarray:
    vec = np.zeros(len(TONES) + len(DATASETS), dtype=np.float64)
    t_idx = TONES.index(rec.skin_tone) if rec.skin_tone in TONES else TONES.index("UNKNOWN")
    vec[t_idx] = 1.0
    if rec.dataset in DATASETS:
        vec[len(TONES) + DATASETS.index(rec.dataset)] = 1.0
    return vec


def _run_tone_collection_shortcut_check(
    train_lab: list[tuple[np.ndarray, int, DermCatalogRecord]],
    test_lab: list[tuple[np.ndarray, int, DermCatalogRecord]],
) -> dict[str, Any]:
    X_tr = np.stack([_encode_tone_collection(r) for _, _, r in train_lab], axis=0)
    y_tr = np.asarray([y for _, y, _ in train_lab], dtype=np.int64)
    X_te = np.stack([_encode_tone_collection(r) for _, _, r in test_lab], axis=0)
    y_te = np.asarray([y for _, y, _ in test_lab], dtype=np.int64)

    clf = LogisticRegression(C=0.8, class_weight="balanced", max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]
    overall_auroc = round(_compute_binary_auroc(y_te, p_te), 4)

    by_tone: dict[str, Any] = {}
    flagged_tones: list[str] = []
    for t in TONES:
        mask = np.asarray([r.skin_tone == t for _, _, r in test_lab])
        n = int(np.sum(mask))
        pos = int(np.sum(y_te[mask] == 1)) if n else 0
        neg = n - pos
        if pos >= 10 and neg >= 10:
            au = round(_compute_binary_auroc(y_te[mask], p_te[mask]), 4)
            flag = au >= 0.60
            if flag:
                flagged_tones.append(t)
        else:
            au = None
            flag = False
        by_tone[t] = {"n": n, "malignant": pos, "benign": neg, "shortcut_auroc": au, "flagged_ge_060": flag}

    return {
        "overall_auroc": overall_auroc,
        "overall_flagged_ge_060": overall_auroc >= 0.60,
        "flagged_tones": flagged_tones,
        "by_tone": by_tone,
    }


def run_step5_v3_benchmark(
    *,
    epochs: int = 15,
    patience: int = 3,
    eval_only: bool = False,
    out_results_json: str | Path = "docs/results/vision/step5_v3_skin_benchmark.json",
    out_abstain_json: str | Path = "configs/vision/abstention_v3.json",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    splits = load_stratified_skin_dataset(
        meta_dir="var/cache/derm/meta",
        image_size=96,
        include_masked_mg=True,
        include_fitzpatrick17k_labelled=True,
        locked_test_list_path="configs/vision/locked_test_v3.txt",
        val_list_path="configs/vision/val_v3.txt",
        lesion_groups_path="configs/vision/lesion_groups_v3.json",
        min_real_samples=100,
    )
    train_all, val_all, test_all = splits["train"], splits["val"], splits["locked_test"]
    train_lab = _filter_labelled_mg(train_all)
    val_lab = _filter_labelled_mg(val_all)
    test_lab = _filter_labelled_mg(test_all)

    print(
        f"[Step 5] Loaded v3 splits: train={len(train_all)} (lab={len(train_lab)}), "
        f"val={len(val_all)} (lab={len(val_lab)}), locked_test_v3={len(test_all)} (lab={len(test_lab)}). "
        f"Estimated CPU time: ~40-60 min per seed (max {epochs} epochs, patience={patience}).",
        flush=True,
    )

    split_counts = compute_split_counts(
        {
            "train": [r for _, _, r in train_all],
            "val": [r for _, _, r in val_all],
            "locked_test_v3": [r for _, _, r in test_all],
        }
    )

    # 1. Train 3 seeds + no_meta ablation from random init (or load saved v3 models if eval_only)
    models: dict[int, PyTorchDermResNet] = {}
    seed_runs: dict[str, Any] = {}
    for seed in (1, 2, 3):
        mdir = Path(f"models/derm_cnn_seed{seed}")
        if eval_only and (mdir / "manifest.json").exists():
            m = PyTorchDermResNet.load(mdir)
            info = m.train_history or {"seed": seed, "loaded_from_disk": True}
        else:
            m = PyTorchDermResNet(image_size=96, include_metadata=True, seed=seed)
            info = m.fit(train_all, val_lab, epochs=epochs, batch_size=48, lr=1e-3, patience=patience)
            m.save(mdir, corpus_hash="derm-corpus-v3-grouped")
        models[seed] = m
        seed_runs[f"seed{seed}"] = info

    nometa_dir = Path("models/derm_cnn_no_meta")
    if eval_only and (nometa_dir / "manifest.json").exists():
        nometa = PyTorchDermResNet.load(nometa_dir)
        nometa_info = nometa.train_history or {"seed": 1, "include_metadata": False, "loaded_from_disk": True}
    else:
        nometa = PyTorchDermResNet(image_size=96, include_metadata=False, seed=1)
        nometa_info = nometa.fit(train_all, val_lab, epochs=epochs, batch_size=48, lr=1e-3, patience=patience)
        nometa.save(nometa_dir, corpus_hash="derm-corpus-v3-grouped")

    # 2. Tune abstention on val_lab only
    vp_s1, _, _ = models[1].predict_batch(val_lab, batch_size=64)
    y_val = np.asarray([y for _, y, _ in val_lab], dtype=np.int64)
    abstain_cfg = _fit_val_abstention(y_val, vp_s1)
    Path(out_abstain_json).write_text(json.dumps(abstain_cfg, indent=2) + "\n", encoding="utf-8")
    min_conf = float(abstain_cfg["min_confidence"])

    # 3. Grad-CAM sanity on >= 200 dermoscopic val images (stated threshold >= 0.30 before running)
    dermo_val = [s for s in val_lab if s[2].modality == "dermoscopic"][:220]
    gc_raw = verify_gradcam_adebayo_and_lesion_center(models[1], dermo_val)
    gradcam_sanity = {
        **gc_raw,
        "adebayo_pass_threshold": ADEBAYO_PASS_THRESHOLD,
        "adebayo_passed": float(gc_raw["adebayo_mean_l1_shift"]) >= ADEBAYO_PASS_THRESHOLD,
    }

    # 4. Classical, Metadata-Only, and NumPy baselines (trained on train_lab)
    cls_model = ClassicalSkinClassifier(variant="logreg", include_metadata=True, seed=42)
    cls_model.fit(train_lab)

    X_tr_meta = np.stack([encode_metadata_vector(r) for _, _, r in train_lab], axis=0)
    y_tr_meta = np.asarray([y for _, y, _ in train_lab], dtype=np.int64)
    X_te_meta = np.stack([encode_metadata_vector(r) for _, _, r in test_lab], axis=0)
    meta_scaler = StandardScaler().fit(X_tr_meta)
    meta_clf = LogisticRegression(C=0.8, class_weight="balanced", max_iter=500, random_state=42)
    meta_clf.fit(meta_scaler.transform(X_tr_meta), y_tr_meta)
    pm_meta = meta_clf.predict_proba(meta_scaler.transform(X_te_meta))[:, 1]
    pd_meta = np.full((len(test_lab), len(CORE_DX_CLASSES)), 1.0 / len(CORE_DX_CLASSES), dtype=np.float64)

    np_cnn = DermResNetMultiTask(image_size=64, seed=42)
    np_cnn.fit(train_lab[:5000], val_lab, epochs=8, batch_size=64, lr=0.04)

    # 5. SINGLE locked_test_v3 evaluation (logged to TEST_SET_LOG.md)
    pm_seeds = [models[s].predict_batch(test_lab, batch_size=64)[0] for s in (1, 2, 3)]
    pd_seeds = [models[s].predict_batch(test_lab, batch_size=64)[1] for s in (1, 2, 3)]
    pm_s1, pd_s1 = pm_seeds[0], pd_seeds[0]
    pm_ens = (pm_seeds[0] + pm_seeds[1] + pm_seeds[2]) / 3.0
    pd_ens = (pd_seeds[0] + pd_seeds[1] + pd_seeds[2]) / 3.0
    pm_nometa, pd_nometa, _ = nometa.predict_batch(test_lab, batch_size=64)
    pm_cls, pd_cls = cls_model.predict_proba(test_lab)
    np_out = np_cnn.predict_batch(test_lab)
    pm_np = np_out["p_malignant"]
    pd_np = np_out["p_diagnosis"]

    def _abstain(p: np.ndarray) -> np.ndarray:
        return np.maximum(p, 1.0 - p) < min_conf

    s1_metrics = compute_comprehensive_metrics(test_lab, pm_s1, pd_s1, abstained=_abstain(pm_s1), n_boot=300, seed=42)
    ens_metrics = compute_comprehensive_metrics(test_lab, pm_ens, pd_ens, abstained=_abstain(pm_ens), n_boot=300, seed=42)
    nometa_metrics = compute_comprehensive_metrics(test_lab, pm_nometa, pd_nometa, abstained=_abstain(pm_nometa), n_boot=300, seed=42)
    cls_metrics = compute_comprehensive_metrics(test_lab, pm_cls, pd_cls, abstained=_abstain(pm_cls), n_boot=300, seed=42)
    meta_metrics = compute_comprehensive_metrics(test_lab, pm_meta, pd_meta, abstained=_abstain(pm_meta), n_boot=300, seed=42)
    np_metrics = compute_comprehensive_metrics(test_lab, pm_np, pd_np, abstained=_abstain(pm_np), n_boot=300, seed=42)

    corpus_prev = _compute_corpus_modality_prevalence()
    reweighted_ens = _reweighted_view(test_lab, pm_ens, _abstain(pm_ens), corpus_prev)
    reweighted_s1 = _reweighted_view(test_lab, pm_s1, _abstain(pm_s1), corpus_prev)
    shortcut_check = _run_tone_collection_shortcut_check(train_lab, test_lab)

    paired = paired_lesion_bootstrap_holm(
        test_lab,
        {
            "pytorch_resnet_seed1": pm_s1,
            "pytorch_resnet_3seed_ensemble": pm_ens,
            "pytorch_resnet_no_meta": pm_nometa,
            "classical_logreg": pm_cls,
            "metadata_only_logreg": pm_meta,
            "numpy_frozen_random_features": pm_np,
        },
        reference_model="pytorch_resnet_3seed_ensemble",
        n_boot=400,
        seed=42,
    )

    log_locked_test_access(
        dataset_name="derm-corpus:locked_test_v3",
        model_name="PyTorchDermResNet4Stage-v1.0.0",
        sample_count=len(test_lab),
        reason="Step 5 single evaluation on frozen configs/vision/locked_test_v3.txt",
    )

    y_test = np.asarray([y for _, y, _ in test_lab], dtype=np.int64)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "data_source": split_counts,
        "view_a_test_as_sampled": {
            "by_modality_ensemble": ens_metrics.get("subgroups", {}).get("by_modality", {}),
            "by_modality_seed1": s1_metrics.get("subgroups", {}).get("by_modality", {}),
            "overall_note": "Overall AUROC mixes modalities with very different base rates; lead with per-modality numbers.",
            "models": {
                "pytorch_resnet_3seed_ensemble": ens_metrics,
                "pytorch_resnet_seed1": s1_metrics,
                "pytorch_resnet_no_meta": nometa_metrics,
                "classical_logreg": cls_metrics,
                "metadata_only_logreg": meta_metrics,
                "numpy_frozen_random_features": np_metrics,
            },
        },
        "view_b_corpus_prevalence_reweighted": {
            "pytorch_resnet_3seed_ensemble": reweighted_ens,
            "pytorch_resnet_seed1": reweighted_s1,
        },
        "tone_collection_shortcut_check": shortcut_check,
        "seed_runs": seed_runs,
        "no_meta_train_info": nometa_info,
        "abstention_config": abstain_cfg,
        "coverage_risk_curve_as_sampled": _coverage_risk_curve(y_test, pm_ens),
        "gradcam_sanity": gradcam_sanity,
        "paired_bootstrap_holm": paired,
    }
    Path(out_results_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_results_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import sys

    print(json.dumps(run_step5_v3_benchmark(eval_only="--eval-only" in sys.argv), indent=2))
