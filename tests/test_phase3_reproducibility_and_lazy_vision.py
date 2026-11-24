"""Tests for Phase 3 remediation: lazy on-disk Vision Dataset (F-09) and safe artifact unpacking (F-01)."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.cnn_torch import PyTorchDermResNet


def _make_record(idx: int, malignant: bool) -> DermCatalogRecord:
    return DermCatalogRecord(
        image_id=f"lazy_{idx}",
        dataset="unit_test",
        modality="clinical" if idx % 2 == 0 else "dermoscopic",
        thumb_path="",
        raw_path="",
        skin_tone="UNKNOWN",
        diagnosis_raw="melanoma" if malignant else "nevus",
        malignancy_group="malignant" if malignant else "benign",
        body_site="arm",
        license_name="MIT",
        quality_ok=True,
        confirmation="histopathology",
        group_key=f"grp_{idx}",
        split_assignment="train",
    )


def test_lazy_on_disk_vision_dataset_fit_and_predict(tmp_path: Path) -> None:
    """Verify PyTorchDermResNet.fit() and predict_batch() train & infer from disk Paths without RAM preloading."""
    samples: list[tuple[Path, int, DermCatalogRecord]] = []
    for i in range(8):
        malignant = i % 2 == 1
        arr = np.full((32, 32, 3), 210 if malignant else 45, dtype=np.uint8)
        img_path = tmp_path / f"img_{i}.png"
        Image.fromarray(arr).save(img_path)
        samples.append((img_path, 1 if malignant else 0, _make_record(i, malignant)))

    model = PyTorchDermResNet(image_size=32, include_metadata=True, seed=1, device="cpu", architecture_variant="resnet_v1")
    info = model.fit(samples[:6], samples[6:], epochs=1, batch_size=2, max_samples_per_epoch=6, num_workers=0)
    assert info["epochs_run"] == 1
    assert info["pretrained_weights_used"] is False

    p_mal, p_dx, p_mod = model.predict_batch(samples[6:], batch_size=2, tta=True)
    assert p_mal.shape == (2,)
    assert p_dx.shape[0] == 2
    assert p_mod.shape[0] == 2
    assert np.all((p_mal >= 0.0) & (p_mal <= 1.0))


def test_safe_unpack_archive_verifies_sha256_and_blocks_traversal(tmp_path: Path) -> None:
    from scripts.package_release_artifacts import safe_unpack_archive

    # 1. Valid archive extracts cleanly
    good_tar = tmp_path / "good.tar.gz"
    with tarfile.open(good_tar, "w:gz") as tf:
        data = b'{"ok": true}'
        info = tarfile.TarInfo(name="demo_model/manifest.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    good_sha = hashlib.sha256(good_tar.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        safe_unpack_archive(good_tar, expected_sha256="0" * 64, dest=tmp_path / "out_bad_sha")

    extracted = safe_unpack_archive(good_tar, expected_sha256=good_sha, dest=tmp_path / "out_good")
    assert "demo_model/manifest.json" in extracted
    assert (tmp_path / "out_good" / "demo_model" / "manifest.json").exists()

    # 2. Path traversal member is blocked
    evil_tar = tmp_path / "evil.tar.gz"
    with tarfile.open(evil_tar, "w:gz") as tf:
        data = b"pwned"
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    evil_sha = hashlib.sha256(evil_tar.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="path traversal"):
        safe_unpack_archive(evil_tar, expected_sha256=evil_sha, dest=tmp_path / "out_evil")
