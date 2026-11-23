"""Unit tests for Vision and NLP ensemble uncertainty guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from medresearch_platform.approaches.torch_ner import ensemble_torch_ner_predictions
from medresearch_platform.eval_harness.metrics import PredictedEntity
from medresearch_platform.vision.cnn_multitask import SkinPredictionResult
from medresearch_platform.vision.ensemble import CalibratedVisionEnsemble
from medresearch_platform.vision.quality import assess_image_quality
from medresearch_platform.vision.serving import load_vision_benchmark_results


@dataclass
class _StubSkinModel:
    prob: float
    image_size: int = 224
    abstain_min_confidence: float = 0.55

    def predict_image(self, image: Image.Image, **kwargs: Any) -> SkinPredictionResult:
        q = assess_image_quality(image)
        return SkinPredictionResult(
            malignancy_probability=self.prob,
            risk_band="high" if self.prob >= 0.65 else "low",
            top3_diagnoses=[{"rank": 1, "diagnosis": "melanoma", "probability": self.prob}],
            predicted_modality="dermoscopic",
            modality_probabilities={"dermoscopic": 0.95, "clinical": 0.04, "tbp_tile": 0.01},
            energy_score=-2.0,
            confidence=max(self.prob, 1.0 - self.prob),
            abstained=False,
            abstain_reason_en=None,
            abstain_reason_fr=None,
            needs_review=False,
            quality=q,
            heatmap_grid=[[0.5]],
            safety_notice_en="Research tool",
            safety_notice_fr="Outil de recherche",
            model_name="stub",
            pretrained_weights_used=False,
        )


@dataclass
class _StubNerModel:
    entities: list[PredictedEntity]

    def predict_entities(self, text: str) -> list[PredictedEntity]:
        return list(self.entities)


def test_vision_ensemble_uncertainty_guardrail() -> None:
    img = Image.new("RGB", (64, 64), color=(120, 90, 80))

    # Agreeing multi-scale models -> low uncertainty
    ens_agree = CalibratedVisionEnsemble(
        [_StubSkinModel(0.82, 224), _StubSkinModel(0.84, 224), _StubSkinModel(0.81, 256)],  # type: ignore[list-item]
        modality_temperatures={"dermoscopic": 1.0},
    )
    res_low = ens_agree.predict_image(img, modality_hint="dermoscopic")
    assert res_low.confidence_variance < 0.05
    assert res_low.high_uncertainty is False
    assert "MultiScale(224+256)" in res_low.model_name

    # Disagreeing models -> high uncertainty + needs_review
    ens_disagree = CalibratedVisionEnsemble(
        [_StubSkinModel(0.15, 224), _StubSkinModel(0.85, 224), _StubSkinModel(0.20, 256), _StubSkinModel(0.80, 256)],  # type: ignore[list-item]
        modality_temperatures={"dermoscopic": 1.0},
    )
    res_high = ens_disagree.predict_image(img, modality_hint="dermoscopic")
    assert res_high.confidence_variance > 0.12
    assert res_high.high_uncertainty is True
    assert res_high.needs_review is True


def test_nlp_ensemble_uncertainty_guardrail() -> None:
    ent_a = PredictedEntity(type="DISEASE", start=0, end=8, text="melanoma", confidence=0.88)
    ent_b = PredictedEntity(type="DISEASE", start=0, end=8, text="melanoma", confidence=0.86)

    # Unanimous agreement -> low variance
    unanimous = ensemble_torch_ner_predictions(
        [_StubNerModel([ent_a]), _StubNerModel([ent_b]), _StubNerModel([ent_a])],  # type: ignore[list-item]
        "melanoma lesion",
    )
    assert len(unanimous) == 1
    assert unanimous[0].confidence_variance < 0.05
    assert unanimous[0].high_uncertainty is False

    # Partial agreement (2 of 3) -> high variance > 0.12 -> flagged for review
    split = ensemble_torch_ner_predictions(
        [_StubNerModel([ent_a]), _StubNerModel([ent_b]), _StubNerModel([])],  # type: ignore[list-item]
        "melanoma lesion",
    )
    assert len(split) == 1
    assert split[0].confidence_variance > 0.12
    assert split[0].high_uncertainty is True
    assert split[0].needs_review is True


def test_vision_5seed_multiscale_benchmark_surfaced() -> None:
    summary = load_vision_benchmark_results()
    models = summary["skin_benchmark"]["models"]
    assert "pytorch_se_resnet_multiscale_5seed_tta_ensemble" in models
    lt = models["pytorch_se_resnet_multiscale_5seed_tta_ensemble"]["locked_test"]
    assert lt["auroc"] > 0.92
    assert "ensemble_uncertainty" in lt


def test_clinical_lesion_roi_cropper_and_retake_guidance() -> None:
    from medresearch_platform.vision.quality import build_retake_guidance, crop_clinical_lesion_roi

    # Create synthetic skin image with centered dark lesion
    im = Image.new("RGB", (200, 200), color=(210, 170, 150))
    for y in range(85, 115):
        for x in range(85, 115):
            im.putpixel((x, y), (70, 40, 35))
    cropped, box = crop_clinical_lesion_roi(im)
    assert cropped.size[0] >= 64 and cropped.size[1] >= 64
    assert 0.45 <= box["crop_frac"] <= 1.0

    # Uniform blurry dark image -> retake guidance recommended with bilingual tips
    bad_im = Image.new("RGB", (64, 64), color=(15, 15, 15))
    q_bad = assess_image_quality(bad_im)
    guide = build_retake_guidance(q_bad)
    assert guide["retake_recommended"] is True
    assert len(guide["tips_en"]) >= 1
    assert len(guide["tips_fr"]) >= 1
