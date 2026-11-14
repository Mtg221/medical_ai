"""Tests for Dermatology CV metadata catalog, leak-free splits, and RandomInitDermConvNet."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from medresearch_platform.dermatology.catalog import (
    generate_synthetic_derm_samples,
    iter_derm_catalog,
    summarize_derm_catalog,
)
from medresearch_platform.dermatology.vision_classifier import DermVisionTrainer


def test_derm_catalog_streaming_and_leak_free_splits(tmp_path: Path) -> None:
    """Verify gzip/plain JSON catalog streaming, filtering, and deterministic group splits."""
    rows_clinical = [
        {
            "i": "isic:ISIC_0001",
            "d": "isic",
            "t": "derived/thumbs/a.webp",
            "f": "raw/a.jpg",
            "st": "II",
            "mk": "case_100",
            "dx": "Benign",
            "mg": "benign",
            "bs": "back",
            "lc": "CC-BY",
            "ok": True,
            "cf": "histopathology",
        },
        {
            "i": "isic:ISIC_0002",
            "d": "isic",
            "t": "derived/thumbs/b.webp",
            "f": "raw/b.jpg",
            "st": "II",
            "mk": "case_100",
            "dx": "Benign",
            "mg": "benign",
            "bs": "back",
            "lc": "CC-BY",
            "ok": True,
            "cf": "histopathology",
        },
        {
            "i": "fitz:0003",
            "d": "fitzpatrick17k",
            "t": "derived/thumbs/c.webp",
            "f": "raw/c.jpg",
            "st": "V",
            "mk": None,
            "dx": "melanoma",
            "mg": "malignant",
            "bs": None,
            "lc": "CC-BY-NC-4.0",
            "ok": False,
            "cf": "atlas",
        },
    ]
    gz_path = tmp_path / "manifest-clinical-000.json"
    gz_path.write_bytes(gzip.compress(json.dumps({"count": 3, "items": rows_clinical}).encode("utf-8")))

    summary = summarize_derm_catalog(tmp_path)
    assert summary["total_records"] == 3
    assert summary["quality_ok_count"] == 2
    assert summary["modalities"]["clinical"] == 3
    assert summary["malignancy_group"]["benign"] == 2
    assert summary["malignancy_group"]["malignant"] == 1

    records = list(iter_derm_catalog(tmp_path, require_quality_ok=True))
    assert len(records) == 2
    # Same case_100 group_key must stay in the exact same split
    assert records[0].group_key == records[1].group_key == "isic:case_100"
    assert records[0].split_assignment == records[1].split_assignment


def test_derm_convnet_gradient_check() -> None:
    """Verify hand-derived Conv2D + AvgPool + GlobalMax/Avg + MLP backprop via finite differences."""
    trainer = DermVisionTrainer(conv_channels=4, hidden_dim=8, seed=7)
    rel_errors = trainer.check_gradients()
    assert rel_errors
    for name, err in rel_errors.items():
        assert err < 1e-4, f"DermConvNet gradient mismatch for {name}: {err}"


def test_derm_convnet_training_convergence_and_fairness() -> None:
    """Verify training from random init, image-only support, and subgroup fairness reporting."""
    samples = generate_synthetic_derm_samples(n_samples=48, image_size=16, seed=42)
    train_samples = [s for s in samples if s[2].split_assignment in {"train", "val"}]
    test_samples = [s for s in samples if s[2].split_assignment == "locked_test"]
    if not test_samples:
        test_samples = samples[-8:]

    trainer = DermVisionTrainer(conv_channels=6, hidden_dim=16, epochs=25, batch_size=8, lr=0.02, seed=42)
    train_info = trainer.fit(train_samples)
    assert train_info["pretrained_weights_used"] is False
    assert len(train_info["initial_weights_sha256"]) == 64
    assert train_info["final_loss"] < train_info["initial_loss"]

    eval_report = trainer.evaluate(test_samples)
    assert eval_report["pretrained_weights_used"] is False
    assert eval_report["accuracy"] >= 0.75
    assert "by_skin_tone" in eval_report and len(eval_report["by_skin_tone"]) >= 1
    assert "by_modality" in eval_report and len(eval_report["by_modality"]) >= 1

    # Verify image-only mode (missing metadata) also trains cleanly
    img_only = DermVisionTrainer(
        conv_channels=4, hidden_dim=12, epochs=15, batch_size=8, lr=0.02, seed=42, include_metadata=False
    )
    img_info = img_only.fit(train_samples)
    assert img_info["final_loss"] < img_info["initial_loss"]
