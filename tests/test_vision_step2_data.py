"""Step 2 real-data invariants: lesion-key split integrity, cross-split dHash dedup, masked mg labels, and no silent synthetic fallback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.eval_vision import (
    assert_zero_lesion_overlap,
    load_stratified_skin_dataset,
)
from medresearch_platform.vision.sync import (
    deduplicate_cross_split_dhash,
    freeze_locked_test_v2,
    select_records_from_sampling_plan,
    write_derm_v2_data_card,
)


def _make_record(image_id: str, group_key: str, split: str, *, modality: str = "dermoscopic", mg: str = "benign", tone: str = "II") -> DermCatalogRecord:
    return DermCatalogRecord(
        image_id=image_id,
        dataset="isic",
        modality=modality,
        thumb_path=f"{modality}/{image_id}.webp",
        raw_path=None,
        skin_tone=tone,
        diagnosis_raw="nevus",
        malignancy_group=mg,
        body_site="back",
        license_name="CC-BY-4.0",
        quality_ok=True,
        confirmation="histopathology",
        group_key=group_key,
        split_assignment=split,
        diagnosis_harmonised="nevus",
        mg_consistent=True,
        research_allowed=True,
    )


def test_lesion_overlap_fails_loudly_and_passes_when_disjoint() -> None:
    arr = np.zeros((16, 16, 3), dtype=np.float64)
    disjoint = {
        "train": [(arr, 0, _make_record("im1", "mk:lesion_1", "train"))],
        "val": [(arr, 0, _make_record("im2", "mk:lesion_2", "val"))],
        "locked_test": [(arr, 1, _make_record("im3", "mk:lesion_3", "locked_test"))],
    }
    assert_zero_lesion_overlap(disjoint)

    leaking = {
        "train": [(arr, 0, _make_record("im1", "mk:shared_lesion", "train"))],
        "locked_test": [(arr, 0, _make_record("im2", "mk:shared_lesion", "locked_test"))],
    }
    with pytest.raises(ValueError, match="Lesion group_key leakage"):
        assert_zero_lesion_overlap(leaking)


def test_cross_split_dhash_dedup_drops_from_test_not_train(tmp_path: Path) -> None:
    cache = tmp_path / "thumbs"
    (cache / "dermoscopic").mkdir(parents=True)

    # Create identical gradient image in train and locked_test_v2, plus a distinct image in locked_test_v2
    grad = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    rgb_grad = np.stack([grad, grad, grad], axis=-1)
    Image.fromarray(rgb_grad).save(cache / "dermoscopic" / "tr_1.webp", format="WEBP")
    Image.fromarray(rgb_grad).save(cache / "dermoscopic" / "te_dup.webp", format="WEBP")

    # Distinct checkerboard-like image
    distinct = np.zeros((64, 64, 3), dtype=np.uint8)
    distinct[::2, 1::2] = 240
    distinct[1::2, ::2] = 180
    Image.fromarray(distinct).save(cache / "dermoscopic" / "te_unique.webp", format="WEBP")

    splits = {
        "train": [_make_record("tr_1", "mk:l1", "train")],
        "val": [],
        "locked_test_v2": [
            _make_record("te_dup", "mk:l2", "locked_test"),
            _make_record("te_unique", "mk:l3", "locked_test"),
        ],
    }
    deduped, dropped = deduplicate_cross_split_dhash(splits, cache)
    assert dropped == 1
    assert len(deduped["train"]) == 1
    assert [r.image_id for r in deduped["locked_test_v2"]] == ["te_unique"]


def test_no_label_fabrication_and_no_silent_synthetic_fallback(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    thumbs = tmp_path / "thumbs"
    meta.mkdir()
    (thumbs / "tbp_tile").mkdir(parents=True)

    # 1. Empty thumbs -> raises RuntimeError (no synthetic fallback)
    with pytest.raises(RuntimeError, match="synthetic fallback is disabled|No real cached thumbnails"):
        load_stratified_skin_dataset(meta_dir=meta, cache_dirs=[thumbs], min_real_samples=1)

    # 2. tbp_tile with mg="unknown" is NEVER fabricated as "benign"
    rows = []
    for i in range(15):
        img_name = f"tbp_{i}.webp"
        Image.new("RGB", (32, 32), color=(120 + i, 90, 80)).save(thumbs / "tbp_tile" / img_name, format="WEBP")
        rows.append({
            "i": f"isic:TBP_{i}",
            "d": "isic",
            "t": f"tbp_tile/{img_name}",
            "mg": "unknown" if i < 5 else ("malignant" if i % 2 == 0 else "benign"),
            "mk": f"lesion_{i}",
            "ok": True,
            "lc": "CC-BY-4.0",
        })
    (meta / "manifest-tbp_tile-000.json").write_text(json.dumps({"items": rows}), encoding="utf-8")

    # Without include_masked_mg: only the 10 benign/malignant records are loaded
    splits_unmasked = load_stratified_skin_dataset(
        meta_dir=meta,
        cache_dirs=[thumbs],
        include_masked_mg=False,
        locked_test_list_path=tmp_path / "nonexistent.txt",
        min_real_samples=5,
    )
    total_unmasked = sum(len(v) for v in splits_unmasked.values())
    assert total_unmasked == 10

    # With include_masked_mg=True: the 5 unknown records have label == -1 (masked)
    splits_masked = load_stratified_skin_dataset(
        meta_dir=meta,
        cache_dirs=[thumbs],
        include_masked_mg=True,
        locked_test_list_path=tmp_path / "nonexistent.txt",
        min_real_samples=5,
    )
    all_masked = [s for v in splits_masked.values() for s in v]
    assert len(all_masked) == 15
    assert sum(1 for _, lbl, _ in all_masked if lbl == -1) == 5

    # 3. Sampling plan + freeze + card
    plan_splits = select_records_from_sampling_plan(meta, train_target=8, val_target=3, test_target=4, min_test_per_tone=1, min_test_per_modality=1)
    train_keys = {r.group_key for r in plan_splits["train"]}
    val_keys = {r.group_key for r in plan_splits["val"]}
    test_keys = {r.group_key for r in plan_splits["locked_test_v2"]}
    assert not (train_keys & val_keys)
    assert not (train_keys & test_keys)
    assert not (val_keys & test_keys)

    freeze = freeze_locked_test_v2(
        plan_splits["locked_test_v2"],
        out_path=tmp_path / "locked_test_v2.txt",
        log_path=tmp_path / "TEST_SET_LOG.md",
    )
    assert freeze["n_ids"] == len(plan_splits["locked_test_v2"])
    card = write_derm_v2_data_card(plan_splits, out_path=tmp_path / "CARD_V2.md")
    assert card.exists()
