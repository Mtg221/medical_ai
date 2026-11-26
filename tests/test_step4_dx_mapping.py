"""Step 4 (W2) tests: single shared honest diagnosis mapping, masking unspecified labels, <30 support guard."""

from __future__ import annotations

import numpy as np

from medresearch_platform.vision import classical_skin, cnn_multitask, cnn_torch, eval_vision
from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.dx_mapping import (
    CORE_DX_CLASSES,
    MASKED_DX_INDEX,
    map_record_to_core_dx_index,
)


def _rec(idx: int, *, dx: str, mg: str) -> DermCatalogRecord:
    return DermCatalogRecord(
        image_id=f"ISIC_{idx:07d}",
        dataset="isic",
        modality="dermoscopic",
        thumb_path=None,
        raw_path=None,
        skin_tone="II",
        diagnosis_raw=dx,
        malignancy_group=mg,
        body_site="back",
        license_name="CC-BY-4.0",
        quality_ok=True,
        confirmation="histopathology",
        group_key=f"g_{idx}",
        split_assignment="train",
        diagnosis_harmonised=dx,
        research_allowed=True,
    )


def test_w2_shared_dx_mapping_across_all_train_and_eval_modules() -> None:
    assert classical_skin.map_record_to_core_dx_index is map_record_to_core_dx_index
    assert cnn_torch.map_record_to_core_dx_index is map_record_to_core_dx_index
    assert cnn_multitask.map_record_to_core_dx_index is map_record_to_core_dx_index
    assert eval_vision.map_record_to_core_dx_index is map_record_to_core_dx_index
    assert classical_skin.CORE_DX_CLASSES is CORE_DX_CLASSES
    assert cnn_torch.CORE_DX_CLASSES is CORE_DX_CLASSES
    assert cnn_multitask.CORE_DX_CLASSES is CORE_DX_CLASSES
    assert eval_vision.CORE_DX_CLASSES is CORE_DX_CLASSES


def test_w2_unspecified_malignancy_never_fabricated_as_melanoma_or_nevus() -> None:
    mal_unspec = _rec(1, dx="malignant_unspecified", mg="malignant")
    ben_unspec = _rec(2, dx="benign_unspecified", mg="benign")
    other = _rec(3, dx="other", mg="benign")
    mel = _rec(4, dx="melanoma", mg="malignant")
    nev = _rec(5, dx="nevus", mg="benign")

    assert map_record_to_core_dx_index(mal_unspec) == MASKED_DX_INDEX
    assert map_record_to_core_dx_index(ben_unspec) == MASKED_DX_INDEX
    assert map_record_to_core_dx_index(other) == MASKED_DX_INDEX
    assert map_record_to_core_dx_index(mel) == CORE_DX_CLASSES.index("melanoma")
    assert map_record_to_core_dx_index(nev) == CORE_DX_CLASSES.index("nevus")


def test_w2_eval_drops_dx_classes_under_30_support() -> None:
    rng = np.random.default_rng(7)
    samples = []
    for i in range(40):
        mg = "malignant" if i < 20 else "benign"
        dx = "melanoma" if i < 5 else ("malignant_unspecified" if i < 20 else "benign_unspecified")
        samples.append((rng.random((32, 32, 3)), 1 if mg == "malignant" else 0, _rec(i, dx=dx, mg=mg)))
    p_mal = np.linspace(0.9, 0.1, len(samples))
    p_dx = np.full((len(samples), len(CORE_DX_CLASSES)), 1.0 / len(CORE_DX_CLASSES))
    metrics = eval_vision.compute_comprehensive_metrics(samples, p_mal, p_dx, compute_subgroups=False, n_boot=20)
    assert metrics["diagnosis_labelled_count"] == 5
    assert metrics["diagnosis_status"] == "not_enough_labelled_data"
    assert metrics["diagnosis_macro_f1"] is None
    assert metrics["diagnosis_per_class_f1"] == {}
    assert "too_few_to_conclude" in metrics["diagnosis_per_class_status"]["melanoma"]
