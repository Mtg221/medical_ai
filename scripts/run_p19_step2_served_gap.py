"""Correction Prompt 19 — Step 2: Explain the served gap across 4 settings on locked_test_v3 + val_v3 latency/ensemble check."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.cnn_torch import PyTorchDermResNet
from medresearch_platform.vision.eval_vision import (
    compute_comprehensive_metrics,
    load_stratified_skin_dataset,
)
from medresearch_platform.vision.router_torch import PyTorchPictureTriageRouter


def _open_record_image(rec: DermCatalogRecord) -> Image.Image:
    rel = str(rec.thumb_path or rec.raw_path or "")
    candidates = [Path(rel), Path("var/cache/derm/thumbs") / rel, Path("var/cache/derm") / rel]
    img_path = next((p for p in candidates if p.exists()), candidates[1])
    return Image.open(img_path).convert("RGB")


def _predict_direct(
    models: list[PyTorchDermResNet],
    images: list[Image.Image],
    modalities: list[str],
    body_sites: list[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    p_mal_list: list[float] = []
    abstained_list: list[bool] = []
    for pil_im, mod, bs in zip(images, modalities, body_sites, strict=True):
        preds = [m.predict_image(pil_im, modality_hint=mod, body_site=bs) for m in models]
        p_mal = float(np.mean([float(p.malignancy_probability) for p in preds]))
        # Abstain if mean probability is inside the low-confidence band or any model abstains
        abstained = any(bool(p.abstained) for p in preds) if len(models) == 1 else (0.35 <= p_mal <= 0.65)
        p_mal_list.append(p_mal)
        abstained_list.append(abstained)
    elapsed = time.perf_counter() - t0
    return np.asarray(p_mal_list, dtype=np.float64), np.asarray(abstained_list, dtype=bool), elapsed


def _eval_setting(
    name: str,
    models: list[PyTorchDermResNet],
    items: list[tuple[np.ndarray, int, DermCatalogRecord]],
    modalities: list[str],
    body_sites: list[str],
) -> dict[str, Any]:
    images = [_open_record_image(x[2]) for x in items]
    p_mal, abstained, elapsed = _predict_direct(models, images, modalities, body_sites)
    dx_dummy = np.zeros((len(items), 7), dtype=np.float64)
    overall = compute_comprehensive_metrics(
        items,
        p_mal,
        dx_dummy,
        abstained=abstained,
    )
    by_modality: dict[str, Any] = {}
    for mod in ("dermoscopic", "clinical", "tbp_tile"):
        mask = np.asarray([x[2].modality == mod for x in items])
        if int(np.sum(mask)) > 0:
            sub_items = [s for s, m in zip(items, mask) if m]
            by_modality[mod] = compute_comprehensive_metrics(
                sub_items,
                p_mal[mask],
                dx_dummy[mask],
                abstained=abstained[mask],
            )
    return {
        "setting": name,
        "elapsed_seconds": round(elapsed, 3),
        "ms_per_image": round((elapsed / max(1, len(items))) * 1000.0, 2),
        "overall": overall,
        "by_modality": by_modality,
    }


def main() -> None:
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
    val_lab = [s for s in splits["val"] if s[2].malignancy_group in ("benign", "malignant")]
    test_lab = [s for s in splits["locked_test"] if s[2].malignancy_group in ("benign", "malignant")]

    bsdrop40 = PyTorchDermResNet.load(Path("models/derm_cnn_bsdrop40_seed1"))
    seed2 = PyTorchDermResNet.load(Path("models/derm_cnn_seed2"))
    seed3 = PyTorchDermResNet.load(Path("models/derm_cnn_seed3"))
    router = PyTorchPictureTriageRouter.load(Path("models/vision_router_v1"))

    # 1. Val-only decision: 1-seed bsdrop40 vs 3-seed ensemble + CPU latency
    val_true_mod = [str(s[2].modality or "clinical") for s in val_lab]
    val_true_bs = [str(s[2].body_site or "other") for s in val_lab]
    val_1seed = _eval_setting("val_bsdrop40_1seed", [bsdrop40], val_lab, val_true_mod, val_true_bs)
    val_3seed = _eval_setting("val_ensemble_3seed", [bsdrop40, seed2, seed3], val_lab, val_true_mod, val_true_bs)

    # 2. Router predictions on locked_test_v3 (for Setting 2)
    test_true_mod = [str(s[2].modality or "clinical") for s in test_lab]
    test_true_bs = [str(s[2].body_site or "other") for s in test_lab]
    test_router_mod: list[str] = []
    router_counts: Counter[str] = Counter()
    for _arr, _lbl, rec in test_lab:
        rel = str(rec.thumb_path or rec.raw_path or "")
        candidates = [Path(rel), Path("var/cache/derm/thumbs") / rel, Path("var/cache/derm") / rel]
        img_path = next((p for p in candidates if p.exists()), candidates[1])
        pil_im = Image.open(img_path).convert("RGB")
        rpred = router.triage_image(pil_im)
        rcls = str(rpred.predicted_class)
        router_counts[rcls] += 1
        if rcls == "skin_dermoscopic":
            test_router_mod.append("dermoscopic")
        elif rcls == "skin_clinical":
            test_router_mod.append("clinical")
        else:
            test_router_mod.append("clinical")

    s1 = _eval_setting("1_direct_bsdrop40_true_meta", [bsdrop40], test_lab, test_true_mod, test_true_bs)
    s2 = _eval_setting("2_direct_bsdrop40_router_mod_user_bs", [bsdrop40], test_lab, test_router_mod, test_true_bs)
    s4 = _eval_setting("4_direct_3seed_ensemble_true_meta", [bsdrop40, seed2, seed3], test_lab, test_true_mod, test_true_bs)

    s3_path = Path("docs/results/vision/p18_step1_served_path_eval.json")
    s3 = json.loads(s3_path.read_text(encoding="utf-8"))
    s5_bench = json.loads(Path("docs/results/vision/step5_v3_skin_benchmark.json").read_text(encoding="utf-8"))

    out = {
        "val_selection": {
            "bsdrop40_1seed": {
                "overall_auroc": val_1seed["overall"]["auroc"],
                "dermoscopic_auroc": val_1seed["by_modality"].get("dermoscopic", {}).get("auroc"),
                "clinical_auroc": val_1seed["by_modality"].get("clinical", {}).get("auroc"),
                "ms_per_image": val_1seed["ms_per_image"],
            },
            "ensemble_3seed": {
                "overall_auroc": val_3seed["overall"]["auroc"],
                "dermoscopic_auroc": val_3seed["by_modality"].get("dermoscopic", {}).get("auroc"),
                "clinical_auroc": val_3seed["by_modality"].get("clinical", {}).get("auroc"),
                "ms_per_image": val_3seed["ms_per_image"],
            },
        },
        "router_class_counts_on_locked_test_v3": dict(router_counts),
        "settings": {
            "1_direct_bsdrop40_true_meta": s1,
            "2_direct_bsdrop40_router_mod_user_bs": s2,
            "3_full_served_api_upload": {
                "overall": s3["served_path_overall"],
                "by_modality": s3["served_path_by_modality"],
                "routing_counts": s3.get("routing_counts", {}),
                "abstention_counts": s3.get("abstention_counts", {}),
                "elapsed_seconds": s3.get("elapsed_seconds"),
            },
            "4_direct_3seed_ensemble_true_meta": s4,
            "offline_step5_3seed_reference": {
                "overall": s5_bench.get("ensemble_3seed", s5_bench.get("ensemble", {})).get("overall"),
                "by_modality": s5_bench.get("ensemble_3seed", s5_bench.get("ensemble", {})).get("by_modality"),
            },
        },
    }
    out_path = Path("docs/results/vision/p19_step2_served_gap.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("val 1-seed:", out["val_selection"]["bsdrop40_1seed"])
    print("val 3-seed:", out["val_selection"]["ensemble_3seed"])
    for k in ("1_direct_bsdrop40_true_meta", "2_direct_bsdrop40_router_mod_user_bs", "3_full_served_api_upload", "4_direct_3seed_ensemble_true_meta"):
        v = out["settings"][k]
        print(
            k,
            "overall=", round(v["overall"]["auroc"], 4),
            "derm=", round(v["by_modality"].get("dermoscopic", {}).get("auroc", 0.0), 4),
            "clin=", round(v["by_modality"].get("clinical", {}).get("auroc", 0.0), 4),
            "tbp=", round(v["by_modality"].get("tbp_tile", {}).get("auroc", 0.0), 4),
        )


if __name__ == "__main__":
    main()
