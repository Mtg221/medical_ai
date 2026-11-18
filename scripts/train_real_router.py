"""Train and evaluate the Step 3 real 5-class PyTorch picture triage router."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from medresearch_platform.vision.catalog import iter_derm_catalog
from medresearch_platform.vision.router import ROUTER_CLASSES
from medresearch_platform.vision.router_torch import (
    PyTorchPictureTriageRouter,
    evaluate_real_router,
    render_synth_med_document_pages,
)


def _gather_images(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted(
        p
        for p in dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )


def run_step3_real_router(
    *,
    meta_dir: str | Path = "var/cache/derm/meta",
    thumbs_dir: str | Path = "var/cache/derm/thumbs",
    locked_test_path: str | Path = "configs/vision/locked_test_v3.txt",
    router_real_dir: str | Path = "data/router_real",
    out_model_dir: str | Path = "models/vision_router_v1",
    out_eval_json: str | Path = "docs/results/vision/step3_real_router_eval.json",
    epochs: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    t_base = Path(thumbs_dir)
    r_base = Path(router_real_dir)

    frozen_ids: set[str] = set()
    lt = Path(locked_test_path)
    if lt.exists():
        frozen_ids = {ln.strip() for ln in lt.read_text(encoding="utf-8").splitlines() if ln.strip()}

    # 1. Gather real skin thumbnails per modality (disjoint from locked_test_v2)
    by_mod: dict[str, list[Path]] = defaultdict(list)
    for rec in iter_derm_catalog(meta_dir, require_quality_ok=True, require_research_licence=True):
        if rec.image_id in frozen_ids or not rec.thumb_path:
            continue
        if rec.modality not in ("clinical", "dermoscopic", "tbp_tile"):
            continue
        lp = t_base / rec.thumb_path
        if lp.exists() and lp.stat().st_size > 0:
            by_mod[f"skin_{rec.modality}"].append(lp)

    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []
    eval_items: list[tuple[Path, int]] = []

    for mod_cls in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile"):
        cls_idx = ROUTER_CLASSES.index(mod_cls)
        paths = list(by_mod[mod_cls])
        rng.shuffle(paths)
        eval_p = paths[:80]
        val_p = paths[80:160]
        tr_p = paths[160:760]
        train_items.extend((p, cls_idx) for p in tr_p)
        val_items.extend((p, cls_idx) for p in val_p)
        eval_items.extend((p, cls_idx) for p in eval_p)

    # 2. document_scan: phone-photo-augmented rendered SYNTH-MED notes + real CC0/owner photos
    doc_cls_idx = ROUTER_CLASSES.index("document_scan")
    doc_tr = render_synth_med_document_pages("data/processed/v3_1/train.jsonl", r_base / "document_scan_rendered_v2" / "train", max_docs=240, seed=seed, phone_photo_aug=True)
    doc_va = render_synth_med_document_pages("data/processed/v3_1/val.jsonl", r_base / "document_scan_rendered_v2" / "val", max_docs=80, seed=seed + 1, phone_photo_aug=True)
    doc_te = render_synth_med_document_pages("data/processed/v3_1/test_iid.jsonl", r_base / "document_scan_rendered_v2" / "eval", max_docs=80, seed=seed + 2, phone_photo_aug=True)
    train_items.extend((p, doc_cls_idx) for p in doc_tr)
    val_items.extend((p, doc_cls_idx) for p in doc_va)
    eval_items.extend((p, doc_cls_idx) for p in doc_te)

    owner_doc_paths = _gather_images(r_base / "document_scan")
    rng.shuffle(owner_doc_paths)
    owner_eval_paths = owner_doc_paths
    if len(owner_doc_paths) >= 100:
        n_doc_eval = max(60, len(owner_doc_paths) // 4)
        n_doc_val = min(35, len(owner_doc_paths) // 8)
        owner_eval_paths = owner_doc_paths[:n_doc_eval]
        val_items.extend((p, doc_cls_idx) for p in owner_doc_paths[n_doc_eval : n_doc_eval + n_doc_val])
        train_items.extend((p, doc_cls_idx) for p in owner_doc_paths[n_doc_eval + n_doc_val :])
        eval_items.extend((p, doc_cls_idx) for p in owner_eval_paths)
    elif len(owner_doc_paths) >= 50:
        half = len(owner_doc_paths) // 2
        train_items.extend((p, doc_cls_idx) for p in owner_doc_paths[:half])
        owner_eval_paths = owner_doc_paths[half:]
        eval_items.extend((p, doc_cls_idx) for p in owner_eval_paths)

    # 3. other: real CC0/PD everyday photos
    other_cls_idx = ROUTER_CLASSES.index("other")
    other_paths = _gather_images(r_base / "other")
    rng.shuffle(other_paths)
    n_oth = len(other_paths)
    n_oth_eval = max(50, min(70, n_oth // 5))
    n_oth_val = max(40, min(60, n_oth // 6))
    eval_items.extend((p, other_cls_idx) for p in other_paths[:n_oth_eval])
    val_items.extend((p, other_cls_idx) for p in other_paths[n_oth_eval : n_oth_eval + n_oth_val])
    train_items.extend((p, other_cls_idx) for p in other_paths[n_oth_eval + n_oth_val :])

    # 4. Train PyTorch router from random init
    router = PyTorchPictureTriageRouter(image_size=96, seed=seed)
    train_info = router.fit(train_items, val_items, epochs=epochs, batch_size=32, lr=1e-3)
    manifest = router.save(out_model_dir)

    # 5. Evaluate on held-out real set + OOD check
    ood_paths = _gather_images(r_base / "ood_check")
    eval_report = evaluate_real_router(
        router,
        eval_items,
        ood_paths=ood_paths,
        owner_doc_paths=owner_eval_paths,
    )
    eval_report["training"] = train_info
    eval_report["manifest"] = manifest
    eval_report["split_sizes"] = {
        "train": len(train_items),
        "val": len(val_items),
        "held_out_eval": len(eval_items),
        "ood_check": len(ood_paths),
        "owner_document_photos_total": len(owner_doc_paths),
    }

    out_json = Path(out_eval_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(eval_report, indent=2) + "\n", encoding="utf-8")
    return eval_report


if __name__ == "__main__":
    print(json.dumps(run_step3_real_router(), indent=2))
