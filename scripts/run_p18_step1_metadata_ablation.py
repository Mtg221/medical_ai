"""Correction Prompt 18 — Step 1: Val-only metadata ablation (pixels_only vs modality_only vs full+body_site_dropout)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.cnn_torch import PyTorchDermResNet, _compute_binary_auroc
from medresearch_platform.vision.eval_vision import load_stratified_skin_dataset


def _filter_labelled_mg(
    samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
) -> list[tuple[np.ndarray, int, DermCatalogRecord]]:
    return [s for s in samples if s[2].malignancy_group in ("benign", "malignant")]


def _auroc_slice(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    n = int(np.sum(mask))
    pos = int(np.sum(y[mask] == 1)) if n else 0
    neg = int(np.sum(y[mask] == 0)) if n else 0
    au = round(_compute_binary_auroc(y[mask], p[mask]), 4) if (pos >= 1 and neg >= 1) else None
    return {"n": n, "malignant": pos, "benign": neg, "auroc": au}


def _evaluate_val_breakdown(
    model: PyTorchDermResNet,
    val_lab: list[tuple[np.ndarray, int, DermCatalogRecord]],
) -> dict[str, Any]:
    p_mal, _, _ = model.predict_batch(val_lab, batch_size=64)
    y_val = np.asarray([lbl for _, lbl, _ in val_lab], dtype=np.int64)
    overall = round(_compute_binary_auroc(y_val, p_mal), 4)

    by_modality: dict[str, Any] = {}
    for mod in ("dermoscopic", "clinical", "tbp_tile"):
        mask = np.asarray([r.modality == mod for _, _, r in val_lab])
        by_modality[mod] = _auroc_slice(y_val, p_mal, mask)

    by_dataset: dict[str, Any] = {}
    for ds in ("fitzpatrick17k", "isic", "skindisnet"):
        mask = np.asarray([r.dataset == ds for _, _, r in val_lab])
        by_dataset[ds] = _auroc_slice(y_val, p_mal, mask)

    by_dataset_modality: dict[str, Any] = {}
    for ds in ("fitzpatrick17k", "isic"):
        for mod in ("dermoscopic", "clinical", "tbp_tile"):
            mask = np.asarray([r.dataset == ds and r.modality == mod for _, _, r in val_lab])
            if int(np.sum(mask)) > 0:
                by_dataset_modality[f"{ds}__{mod}"] = _auroc_slice(y_val, p_mal, mask)

    return {
        "n_val_labelled": len(val_lab),
        "overall_val_auroc": overall,
        "by_modality": by_modality,
        "by_dataset": by_dataset,
        "by_dataset_modality": by_dataset_modality,
    }


def main() -> None:
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
    train_all, val_all = splits["train"], splits["val"]
    val_lab = _filter_labelled_mg(val_all)
    print(f"[P18 Step 1] Loaded train={len(train_all)}, val_v3_lab={len(val_lab)} (locked_test_v3 NOT touched).", flush=True)

    results: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split_evaluated": "val_v3",
        "variants": {},
    }

    # 1. pixels_only (reuse existing seed=1 v3 checkpoint if present, else train)
    pix_dir = Path("models/derm_cnn_no_meta")
    if (pix_dir / "manifest.json").exists():
        print("[P18 Step 1] Loading existing pixels_only model from models/derm_cnn_no_meta ...", flush=True)
        m_pix = PyTorchDermResNet.load(pix_dir)
    else:
        m_pix = PyTorchDermResNet(image_size=96, metadata_mode="pixels_only", seed=1)
        m_pix.fit(train_all, val_all, epochs=12, batch_size=48, lr=1e-3, patience=3)
        m_pix.save(pix_dir)
    results["variants"]["pixels_only"] = {
        "model_dir": str(pix_dir),
        "train_info": m_pix.train_history,
        "val_metrics": _evaluate_val_breakdown(m_pix, val_lab),
    }

    # 2. modality_only (pixels + modality, body_site zeroed)
    mod_dir = Path("models/derm_cnn_mod_only_seed1")
    print("[P18 Step 1] Training modality_only (seed=1) ...", flush=True)
    m_mod = PyTorchDermResNet(image_size=96, metadata_mode="modality_only", seed=1)
    t_mod = m_mod.fit(train_all, val_all, epochs=12, batch_size=48, lr=1e-3, patience=3)
    m_mod.save(mod_dir)
    results["variants"]["modality_only"] = {
        "model_dir": str(mod_dir),
        "train_info": t_mod,
        "val_metrics": _evaluate_val_breakdown(m_mod, val_lab),
    }

    # 3. full + 40% body_site dropout
    bs_dir = Path("models/derm_cnn_bsdrop40_seed1")
    print("[P18 Step 1] Training full + body_site_dropout=0.40 (seed=1) ...", flush=True)
    m_bs = PyTorchDermResNet(image_size=96, metadata_mode="full", body_site_dropout=0.40, seed=1)
    t_bs = m_bs.fit(train_all, val_all, epochs=12, batch_size=48, lr=1e-3, patience=3)
    m_bs.save(bs_dir)
    results["variants"]["full_body_site_dropout_40"] = {
        "model_dir": str(bs_dir),
        "train_info": t_bs,
        "val_metrics": _evaluate_val_breakdown(m_bs, val_lab),
    }

    # Decide winner on val_v3: gain over pixels_only must survive within EACH dataset (fitzpatrick17k AND isic)
    pix_fitz = results["variants"]["pixels_only"]["val_metrics"]["by_dataset"]["fitzpatrick17k"]["auroc"]
    pix_isic = results["variants"]["pixels_only"]["val_metrics"]["by_dataset"]["isic"]["auroc"]
    pix_overall = results["variants"]["pixels_only"]["val_metrics"]["overall_val_auroc"]

    selected = "pixels_only"
    best_overall = pix_overall
    decision_rows: dict[str, Any] = {}
    for v_name, v_data in results["variants"].items():
        vm = v_data["val_metrics"]
        d_fitz = round(float(vm["by_dataset"]["fitzpatrick17k"]["auroc"]) - float(pix_fitz), 4)
        d_isic = round(float(vm["by_dataset"]["isic"]["auroc"]) - float(pix_isic), 4)
        d_all = round(float(vm["overall_val_auroc"]) - float(pix_overall), 4)
        survives = (d_fitz > 0.0) and (d_isic > 0.0) and (d_all > 0.0)
        decision_rows[v_name] = {
            "delta_vs_pixels_fitzpatrick17k": d_fitz,
            "delta_vs_pixels_isic": d_isic,
            "delta_vs_pixels_overall": d_all,
            "survives_within_each_dataset": survives,
        }
        if survives and float(vm["overall_val_auroc"]) > best_overall:
            best_overall = float(vm["overall_val_auroc"])
            selected = v_name

    results["decision"] = {
        "rule": "Keep the variant whose val AUROC gain over pixels_only survives (> 0) within EACH dataset (fitzpatrick17k AND isic); otherwise serve pixels_only.",
        "selected_variant": selected,
        "selected_model_dir": results["variants"][selected]["model_dir"],
        "comparisons_vs_pixels_only": decision_rows,
    }
    results["elapsed_seconds"] = round(time.perf_counter() - t0, 2)

    out_path = Path("docs/results/vision/p18_step1_val_metadata_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[P18 Step 1] DONE in {results['elapsed_seconds']}s -> selected={selected} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
