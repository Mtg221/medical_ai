"""Step 1 tests: dermatology vision imports, harmonised dx taxonomy, licence policy, leak-free splits, dHash dedup, and corpus card."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image

from medresearch_platform.data.cli import main as cli_main
from medresearch_platform.vision import (
    DermVisionTrainer,
    audit_derm_splits,
    compute_dhash64_from_array,
    deduplicate_and_assign_splits,
    evaluate_licence_policy,
    generate_derm_corpus_card,
    generate_synthetic_derm_samples,
    harmonise_diagnosis,
    iter_derm_catalog,
    summarize_derm_catalog,
    summarize_taxonomy_mapping,
)


def test_fixed_dermatology_imports_and_extras_archive() -> None:
    """Verify both medresearch_platform.vision and archived extras.dermatology import cleanly."""
    import extras.dermatology.catalog as ext_cat
    import extras.dermatology.pipeline as ext_pipe
    import extras.dermatology.vision_classifier as ext_vc

    assert hasattr(ext_cat, "iter_derm_catalog")
    assert hasattr(ext_pipe, "run_derm_benchmark")
    assert hasattr(ext_vc, "RandomInitDermConvNet")


def test_harmonised_diagnosis_taxonomy_and_mg_consistency() -> None:
    """Verify mapping of raw dx strings to the 9 harmonised classes and mg consistency check."""
    mel = harmonise_diagnosis("Superficial spreading melanoma", "malignant")
    assert mel.harmonised_dx == "melanoma"
    assert mel.mg_consistent is True
    assert mel.icd10 == "C43.9"

    # Inconsistent mg (melanoma marked benign) must be flagged
    mel_bad = harmonise_diagnosis("melanoma", "benign")
    assert mel_bad.harmonised_dx == "melanoma"
    assert mel_bad.mg_consistent is False

    bcc = harmonise_diagnosis("Basal cell carcinoma", "malignant")
    assert bcc.harmonised_dx == "basal_cell_carcinoma"
    assert bcc.mg_consistent is True

    scc = harmonise_diagnosis("Bowen's disease", "malignant")
    assert scc.harmonised_dx == "squamous_cell_carcinoma"

    ak = harmonise_diagnosis("Actinic keratosis", "benign")
    assert ak.harmonised_dx == "actinic_keratosis"

    nv = harmonise_diagnosis("Melanocytic nevus", "benign")
    assert nv.harmonised_dx == "nevus"

    sek = harmonise_diagnosis("Seborrheic keratosis", "benign")
    assert sek.harmonised_dx == "seborrheic_keratosis"

    df = harmonise_diagnosis("Dermatofibroma", "benign")
    assert df.harmonised_dx == "dermatofibroma"

    vasc = harmonise_diagnosis("Cherry angioma", "benign")
    assert vasc.harmonised_dx == "vascular_lesion"

    infl = harmonise_diagnosis("psoriasis vulgaris", "benign")
    assert infl.harmonised_dx == "inflammatory_infectious_dermatosis"

    scin_dict = harmonise_diagnosis("{'Melanocytic Nevus': 0.8, 'Melanoma': 0.2}", "unknown")
    assert scin_dict.harmonised_dx == "nevus"

    unk = harmonise_diagnosis("unclassified rare condition", "benign")
    assert unk.harmonised_dx == "other"

    # Rare class collapse (< 30) when corpus >= 30
    pairs = [("melanoma", "malignant")] * 35 + [("nevus", "benign")] * 35 + [("dermatofibroma", "benign")] * 4
    summary = summarize_taxonomy_mapping(pairs, min_class_count=30)
    assert "dermatofibroma" in summary["collapsed_to_other"]
    assert summary["harmonised_class_counts"]["melanoma"] == 35
    assert summary["harmonised_class_counts"]["nevus"] == 35
    assert summary["harmonised_class_counts"]["other"] == 4


def test_licence_policy_evaluation() -> None:
    """Verify CC0, CC-BY, CC-BY-NC, and proprietary licence classification."""
    cc0 = evaluate_licence_policy("CC0-1.0")
    assert cc0.research_allowed is True
    assert cc0.redistribution_allowed is True
    assert cc0.commercial_allowed is True
    assert cc0.docs_screenshot_safe is True

    cc_by_nc = evaluate_licence_policy("CC-BY-NC-4.0")
    assert cc_by_nc.research_allowed is True
    assert cc_by_nc.redistribution_allowed is False
    assert cc_by_nc.commercial_allowed is False
    assert cc_by_nc.docs_screenshot_safe is False

    forbidden = evaluate_licence_policy("ALL-RIGHTS-RESERVED")
    assert forbidden.research_allowed is False
    assert forbidden.docs_screenshot_safe is False


def test_catalog_streaming_splits_and_dhash_deduplication(tmp_path: Path) -> None:
    """Verify lesion-grouped splits (0 overlap) and cross-split perceptual hash deduplication."""
    meta_dir = tmp_path / "meta"
    thumbs_dir = tmp_path / "thumbs"
    meta_dir.mkdir()
    thumbs_dir.mkdir()

    # Create two identical images under different lesion keys that land in different splits
    rng = np.random.default_rng(123)
    pattern = (rng.uniform(0, 255, size=(32, 32, 3))).astype(np.uint8)
    unique_pattern = (rng.uniform(0, 255, size=(32, 32, 3))).astype(np.uint8)

    Image.fromarray(pattern).save(thumbs_dir / "dup1.webp")
    Image.fromarray(pattern).save(thumbs_dir / "dup2.webp")
    Image.fromarray(unique_pattern).save(thumbs_dir / "uniq3.webp")

    assert compute_dhash64_from_array(pattern) == compute_dhash64_from_array(pattern)

    rows = [
        {
            "i": "isic:ISIC_0001",
            "d": "isic",
            "t": "dup1.webp",
            "f": "raw/a.jpg",
            "st": "II",
            "mk": "case_100",
            "dx": "melanocytic nevus",
            "mg": "benign",
            "bs": "back",
            "lc": "CC-BY-4.0",
            "ok": True,
            "cf": "histopathology",
        },
        {
            "i": "isic:ISIC_0002",
            "d": "isic",
            "t": "uniq3.webp",
            "f": "raw/b.jpg",
            "st": "II",
            "mk": "case_100",
            "dx": "melanocytic nevus",
            "mg": "benign",
            "bs": "back",
            "lc": "CC-BY-4.0",
            "ok": True,
            "cf": "histopathology",
        },
        {
            "i": "fitz:0003",
            "d": "fitzpatrick17k",
            "t": "dup2.webp",
            "f": "raw/c.jpg",
            "st": "VI",
            "mk": "case_999",
            "dx": "melanoma",
            "mg": "malignant",
            "bs": "lower extremity",
            "lc": "CC-BY-NC-4.0",
            "ok": True,
            "cf": "histopathology",
        },
    ]
    gz_path = meta_dir / "manifest-dermoscopic-000.json"
    gz_path.write_bytes(gzip.compress(json.dumps({"count": 3, "items": rows}).encode("utf-8")))

    summary = summarize_derm_catalog(meta_dir)
    assert summary["total_records"] == 3
    assert summary["unique_lesion_groups"] == 2

    recs = list(iter_derm_catalog(meta_dir, require_quality_ok=True, require_research_licence=True))
    assert len(recs) == 3
    assert recs[0].group_key == recs[1].group_key == "isic:case_100"
    assert recs[0].split_assignment == recs[1].split_assignment

    clean_recs, stats = deduplicate_and_assign_splits(recs, cache_dirs=[thumbs_dir])
    if recs[0].split_assignment != recs[2].split_assignment:
        assert stats["cross_split_duplicates_excluded"] == 1
        assert len(clean_recs) == 2

    audit = audit_derm_splits(recs, cache_dirs=[thumbs_dir])
    assert audit["passed"] is True
    assert audit["lesion_group_overlap"]["train_val"] == 0
    assert audit["lesion_group_overlap"]["train_locked_test"] == 0
    assert audit["lesion_group_overlap"]["val_locked_test"] == 0


def test_derm_convnet_gradients_and_training() -> None:
    """Verify analytical Conv2D + pooling + MLP gradients (< 1e-4) and training convergence."""
    trainer = DermVisionTrainer(conv_channels=4, hidden_dim=8, seed=7)
    rel_errors = trainer.check_gradients()
    assert rel_errors
    for name, err in rel_errors.items():
        assert err < 1e-4, f"DermConvNet gradient mismatch for {name}: {err}"

    samples = generate_synthetic_derm_samples(n_samples=48, image_size=16, seed=42)
    train_samples = [s for s in samples if s[2].split_assignment in {"train", "val"}]
    test_samples = [s for s in samples if s[2].split_assignment == "locked_test"] or samples[-8:]

    trainer_fit = DermVisionTrainer(conv_channels=6, hidden_dim=16, epochs=25, batch_size=8, lr=0.02, seed=42)
    info = trainer_fit.fit(train_samples)
    assert info["pretrained_weights_used"] is False
    assert info["final_loss"] < info["initial_loss"]

    eval_res = trainer_fit.evaluate(test_samples)
    assert eval_res["accuracy"] >= 0.75
    assert "Research tool — not a diagnosis." in eval_res["disclaimer"]


def test_cli_derm_card_and_audit(tmp_path: Path) -> None:
    """Verify `medextract derm card` and `medextract derm audit-splits` CLI commands."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    out_card = tmp_path / "DERM_CORPUS_CARD.md"

    rows = [
        {
            "i": f"isic:ISIC_{idx:04d}",
            "d": "isic",
            "t": f"derived/thumbs/dataset=isic/ISIC_{idx:04d}.webp",
            "f": f"raw/ISIC_{idx:04d}.jpg",
            "st": ["I", "II", "III", "IV", "V", "VI"][idx % 6],
            "mk": f"lesion_{idx // 2:03d}",
            "dx": "melanoma" if idx % 3 == 0 else "melanocytic nevus",
            "mg": "malignant" if idx % 3 == 0 else "benign",
            "bs": "torso",
            "lc": "CC-BY-4.0",
            "ok": True,
            "cf": "histopathology",
        }
        for idx in range(24)
    ]
    (meta_dir / "manifest-dermoscopic-000.json").write_text(
        json.dumps({"items": rows}),
        encoding="utf-8",
    )

    rc_card = cli_main(["derm", "card", "--meta-dir", str(meta_dir), "--out", str(out_card)])
    assert rc_card == 0
    assert out_card.exists()
    content = out_card.read_text(encoding="utf-8")
    assert "DERM_CORPUS_CARD.md" in content
    assert "Research tool — not a diagnosis." in content

    rc_audit = cli_main(["derm", "audit-splits", "--meta-dir", str(meta_dir)])
    assert rc_audit == 0

    direct_report = generate_derm_corpus_card(meta_dir, out_md_path=out_card, log_test_set=False)
    assert direct_report["split_audit"]["passed"] is True
