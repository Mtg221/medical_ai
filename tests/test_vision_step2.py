"""Step 2 tests: 5-class picture triage router, bilingual (EN/FR) image-quality gate, and zero-document-leakage guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from medresearch_platform.vision.quality import (
    QualityThresholds,
    assess_image_quality,
    calibrate_quality_thresholds_on_val,
)
from medresearch_platform.vision.router import (
    ROUTER_CLASSES,
    PictureTriageRouter,
    extract_router_features,
    generate_document_scan_image,
    generate_non_skin_other_image,
    generate_synthetic_skin_modality_image,
)


def test_bilingual_image_quality_checks_and_inclusive_skin_tones() -> None:
    """Verify blur, dark, overexposed, resolution, and inclusive Fitzpatrick I-VI skin area checks with EN/FR reasons."""
    # 1. Clean Fitzpatrick I and VI skin images must both pass quality check
    skin_tone_i = generate_synthetic_skin_modality_image("skin_dermoscopic", 0, seed=10)
    skin_tone_vi = generate_synthetic_skin_modality_image("skin_dermoscopic", 5, seed=10)

    rep_i = assess_image_quality(skin_tone_i)
    rep_vi = assess_image_quality(skin_tone_vi)
    assert rep_i.quality_ok is True, f"Light skin tone failed quality: {rep_i.reasons_en}"
    assert rep_vi.quality_ok is True, f"Dark skin tone VI failed quality: {rep_vi.reasons_en}"
    assert rep_i.reasons_code == []
    assert rep_vi.reasons_code == []

    # 2. Blurry image rejected with EN/FR reasons
    blurred = skin_tone_i.filter(ImageFilter.GaussianBlur(radius=6.0))
    rep_blur = assess_image_quality(blurred)
    assert rep_blur.quality_ok is False
    assert "BLURRY" in rep_blur.reasons_code
    assert any("blurry" in r.lower() or "out of focus" in r.lower() for r in rep_blur.reasons_en)
    assert any("floue" in r.lower() for r in rep_blur.reasons_fr)

    # 3. Too dark image rejected with EN/FR reasons
    dark_arr = (np.asarray(skin_tone_i, dtype=np.float32) * 0.08).astype(np.uint8)
    rep_dark = assess_image_quality(Image.fromarray(dark_arr))
    assert rep_dark.quality_ok is False
    assert "TOO_DARK" in rep_dark.reasons_code
    assert len(rep_dark.reasons_en) >= 1 and len(rep_dark.reasons_fr) >= 1

    # 4. Overexposed image rejected with EN/FR reasons
    bright_arr = np.clip(np.asarray(skin_tone_i, dtype=np.float32) * 1.8 + 200.0, 0, 255).astype(np.uint8)
    rep_bright = assess_image_quality(Image.fromarray(bright_arr))
    assert rep_bright.quality_ok is False
    assert "OVEREXPOSED" in rep_bright.reasons_code
    assert len(rep_bright.reasons_en) >= 1 and len(rep_bright.reasons_fr) >= 1

    # 5. Low resolution image rejected with EN/FR reasons
    tiny = skin_tone_i.resize((24, 24), Image.Resampling.BILINEAR)
    rep_tiny = assess_image_quality(tiny, thresholds=QualityThresholds(min_resolution_px=48))
    assert rep_tiny.quality_ok is False
    assert "LOW_RESOLUTION" in rep_tiny.reasons_code
    assert any("resolution" in r.lower() for r in rep_tiny.reasons_en)
    assert any("résolution" in r.lower() for r in rep_tiny.reasons_fr)

    # 6. Val threshold calibration
    cal = calibrate_quality_thresholds_on_val([skin_tone_i, skin_tone_vi], seed=42)
    assert cal["val_quality_f1"] >= 0.75


def test_picture_triage_router_training_and_document_routing(tmp_path: Path) -> None:
    """Verify 5-class router training from scratch, confusion matrix, save/load, and 0 document photos sent to skin model."""
    train_samples: list[tuple[np.ndarray, np.ndarray, int]] = []
    val_samples: list[tuple[np.ndarray, np.ndarray, int]] = []
    test_samples: list[tuple[np.ndarray, np.ndarray, int]] = []

    for cls_idx, cls_name in enumerate(ROUTER_CLASSES):
        for i in range(30):
            if cls_name in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile"):
                im = generate_synthetic_skin_modality_image(cls_name, i, seed=42)
            elif cls_name == "document_scan":
                im = generate_document_scan_image(i, seed=42)
            else:
                im = generate_non_skin_other_image(i, seed=42)
            p24, d28 = extract_router_features(im)
            if i < 20:
                train_samples.append((p24, d28, cls_idx))
            elif i < 25:
                val_samples.append((p24, d28, cls_idx))
            else:
                test_samples.append((p24, d28, cls_idx))

    router = PictureTriageRouter(seed=42)
    train_info = router.fit(train_samples, val_samples, epochs=35, batch_size=16, lr=0.018)
    assert train_info["pretrained_weights_used"] is False
    assert train_info["final_loss"] < train_info["initial_loss"]

    eval_res = router.evaluate(test_samples)
    assert eval_res["pretrained_weights_used"] is False
    assert eval_res["accuracy"] >= 0.84
    assert eval_res["document_photos_sent_to_skin_model"] == 0
    assert len(eval_res["confusion_matrix"]["matrix"]) == 5

    # Verify triage_image routing on a document scan and a non-skin image
    doc_img = generate_document_scan_image(99, seed=7)
    doc_decision = router.triage_image(doc_img)
    assert doc_decision.predicted_class == "document_scan"
    assert doc_decision.route_action == "ocr_text_pipeline"
    assert doc_decision.abstained is False

    other_img = generate_non_skin_other_image(0, seed=7)
    other_decision = router.triage_image(other_img)
    assert other_decision.predicted_class == "other"
    assert other_decision.route_action == "reject_unsupported"
    assert other_decision.abstained is True
    assert other_decision.abstain_reason_en is not None
    assert other_decision.abstain_reason_fr is not None

    # Save & load round-trip
    model_dir = tmp_path / "router_v1"
    manifest = router.save(model_dir)
    assert manifest["pretrained_weights_used"] is False
    loaded = PictureTriageRouter.load(model_dir)
    doc_decision_2 = loaded.triage_image(doc_img)
    assert doc_decision_2.predicted_class == "document_scan"
    assert doc_decision_2.route_action == "ocr_text_pipeline"
