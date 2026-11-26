"""Step 3 tests: Classical skin baseline, Multi-Task 4-Stage Residual CNN, Grad-CAM sanity test, calibration, abstention, and Holm paired bootstrap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from medresearch_platform.vision.catalog import generate_synthetic_derm_samples
from medresearch_platform.vision.classical_skin import (
    CORE_DX_CLASSES,
    ClassicalSkinClassifier,
    extract_classical_lesion_features,
)
from medresearch_platform.vision.cnn_multitask import (
    SAFETY_NOTICE_EN,
    SAFETY_NOTICE_FR,
    DermResNetMultiTask,
    verify_gradcam_lesion_mask_sanity,
)
from medresearch_platform.vision.eval_vision import (
    compute_comprehensive_metrics,
    paired_lesion_bootstrap_holm,
)


def test_classical_lesion_features_and_classifiers() -> None:
    """Verify RGB/HSV/Lab histograms, LBP/GLCM texture, Otsu shape features, and LogReg/SVM/GBDT baselines."""
    samples = generate_synthetic_derm_samples(n_samples=36, image_size=32, seed=42)
    feat = extract_classical_lesion_features(samples[0][0], size=32)
    assert feat.shape == (64,)
    assert np.all(np.isfinite(feat))

    train_s = [s for s in samples if s[2].split_assignment in ("train", "val")]
    test_s = [s for s in samples if s[2].split_assignment == "locked_test"] or samples[-8:]

    for variant in ("logreg", "svm", "gbdt"):
        clf = ClassicalSkinClassifier(variant=variant, include_metadata=True, seed=42)
        info = clf.fit(train_s)
        assert info["pretrained_weights_used"] is False
        p_mal, p_dx = clf.predict_proba(test_s)
        assert p_mal.shape == (len(test_s),)
        assert p_dx.shape == (len(test_s), len(CORE_DX_CLASSES))
        assert np.allclose(np.sum(p_dx, axis=1), 1.0, atol=1e-4)


def test_multitask_resnet_cnn_gradcam_calibration_and_abstention(tmp_path: Path) -> None:
    """Verify 4-stage residual CNN training from random init, Grad-CAM sanity, abstention, and save/load."""
    samples = generate_synthetic_derm_samples(n_samples=48, image_size=32, seed=42)
    train_s = [s for s in samples if s[2].split_assignment == "train"]
    val_s = [s for s in samples if s[2].split_assignment == "val"] or samples[:8]
    test_s = [s for s in samples if s[2].split_assignment == "locked_test"] or samples[-8:]

    cnn = DermResNetMultiTask(image_size=32, include_metadata=True, multi_task=True, seed=42)
    train_info = cnn.fit(train_s, val_s, epochs=30, batch_size=12, lr=0.018)
    assert train_info["pretrained_weights_used"] is False
    assert len(train_info["initial_weights_sha256"]) == 64
    assert train_info["final_loss"] < train_info["initial_loss"]

    # Grad-CAM sanity check: masking out the lesion must change the spatial heatmap
    sanity = verify_gradcam_lesion_mask_sanity(cnn, test_s[0][0])
    assert sanity["sanity_passed"] is True

    # Full prediction on a clean skin image + PNG heatmap overlay
    clean_pil = Image.fromarray((test_s[0][0] * 255.0).astype(np.uint8)).resize((128, 128), Image.Resampling.BICUBIC)
    pred = cnn.predict_image(clean_pil, rec=test_s[0][2])
    assert pred.pretrained_weights_used is False
    assert pred.safety_notice_en == SAFETY_NOTICE_EN
    assert pred.safety_notice_fr == SAFETY_NOTICE_FR
    assert len(pred.heatmap_grid) == 8 and len(pred.heatmap_grid[0]) == 8

    png_bytes = cnn.render_gradcam_overlay_png(clean_pil, np.asarray(pred.heatmap_grid, dtype=np.float64))
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    # Blurry image must abstain instead of guessing a risk score
    blur_pil = clean_pil.filter(ImageFilter.GaussianBlur(radius=6.0))
    pred_blur = cnn.predict_image(blur_pil)
    assert pred_blur.abstained is True
    assert pred_blur.risk_band == "abstained"
    assert pred_blur.malignancy_probability == -1.0
    assert pred_blur.needs_review is True
    assert pred_blur.abstain_reason_en is not None and pred_blur.abstain_reason_fr is not None

    # Save and load round-trip
    out_dir = tmp_path / "derm_cnn_seed1"
    manifest = cnn.save(out_dir, corpus_hash="test_hash", git_commit="312c9e9")
    assert manifest["pretrained_weights_used"] is False
    loaded = DermResNetMultiTask.load(out_dir)
    pred_loaded = loaded.predict_image(clean_pil, rec=test_s[0][2])
    assert pred_loaded.risk_band == pred.risk_band


def test_comprehensive_metrics_subgroups_and_holm_bootstrap() -> None:
    """Verify AUROC CI, sensitivity@90/95% spec, ECE, subgroup fairness flags, and Holm correction."""
    samples = generate_synthetic_derm_samples(n_samples=40, image_size=16, seed=7)
    y_true = np.asarray([s[1] for s in samples], dtype=np.float64)
    rng = np.random.default_rng(7)

    # Strong model vs weaker baseline
    p_mal_cnn = np.clip(0.15 + 0.70 * y_true + rng.normal(0, 0.08, size=len(samples)), 0.02, 0.98)
    p_mal_base = np.clip(0.35 + 0.30 * y_true + rng.normal(0, 0.18, size=len(samples)), 0.05, 0.95)
    p_dx = np.full((len(samples), len(CORE_DX_CLASSES)), 1.0 / len(CORE_DX_CLASSES), dtype=np.float64)
    for i, s in enumerate(samples):
        p_dx[i, 0 if s[1] == 1 else 4] = 0.75
    p_dx /= np.sum(p_dx, axis=1, keepdims=True)

    metrics = compute_comprehensive_metrics(samples, p_mal_cnn, p_dx, n_boot=100, seed=7)
    assert metrics["auroc"] >= 0.85
    assert len(metrics["auroc_ci95"]) == 2
    assert "sensitivity_at_90_specificity" in metrics
    assert "sensitivity_at_95_specificity" in metrics
    assert "by_skin_tone" in metrics["subgroups"]
    assert "by_modality" in metrics["subgroups"]
    assert "by_dataset" in metrics["subgroups"]
    assert "by_body_site" in metrics["subgroups"]

    # Subgroups with < 30 samples must carry "too few to conclude (< 30)"
    for tone_info in metrics["subgroups"]["by_skin_tone"].values():
        assert tone_info["small_sample_flag"] == "too few to conclude (< 30)"

    holm = paired_lesion_bootstrap_holm(
        samples,
        {"derm_cnn_multitask": p_mal_cnn, "classical_vision": p_mal_base},
        reference_model="derm_cnn_multitask",
        n_boot=200,
        seed=7,
    )
    assert len(holm) == 1
    assert "p_value_holm" in holm[0]
    assert holm[0]["delta_auroc"] > 0.0
