"""Calibrated multi-seed PyTorchDermResNet ensemble with val-fitted per-modality temperature scaling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medresearch_platform.vision.cnn_torch import PyTorchDermResNet

# Fitted strictly on val_v3 (n=1,497) — never locked_test_v3 (docs/results/vision/vision_upgrades_audit.json)
DEFAULT_MODALITY_TEMPERATURES: dict[str, float] = {
    "dermoscopic": 0.90,
    "clinical": 1.20,
    "tbp_tile": 0.80,
}


def _apply_temperature(prob: float, temp: float) -> float:
    p = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    logit = float(np.log(p / (1.0 - p)))
    cal = float(1.0 / (1.0 + np.exp(-np.clip(logit / max(temp, 1e-3), -30.0, 30.0))))
    return round(cal, 4)


class CalibratedVisionEnsemble:
    """Drop-in multi-seed ensemble wrapper matching `PyTorchDermResNet.predict_image`."""

    def __init__(
        self,
        models: list[PyTorchDermResNet],
        *,
        modality_temperatures: dict[str, float] | None = None,
    ) -> None:
        if not models:
            raise ValueError("CalibratedVisionEnsemble requires at least one model.")
        self.models = models
        self.modality_temperatures = dict(modality_temperatures or DEFAULT_MODALITY_TEMPERATURES)
        self.model_version = f"PyTorchDermResNet4Stage-Ensemble{len(models)}-v1.0.0"
        self.backend = "pytorch_cpu"
        self.image_size = int(models[0].image_size)
        self.manifest = {
            **dict(getattr(models[0], "manifest", {})),
            "model_version": self.model_version,
            "ensemble_size": len(models),
            "modality_temperatures": self.modality_temperatures,
            "pretrained_weights_used": False,
        }

    @classmethod
    def load_dirs(
        cls,
        model_dirs: list[Path],
        *,
        modality_temperatures: dict[str, float] | None = None,
    ) -> CalibratedVisionEnsemble:
        loaded = [PyTorchDermResNet.load(d) for d in model_dirs]
        return cls(loaded, modality_temperatures=modality_temperatures)

    def predict_image(
        self,
        image: Image.Image,
        *,
        modality_hint: str | None = None,
        body_site: str | None = None,
    ) -> Any:
        results = [
            m.predict_image(image, modality_hint=modality_hint, body_site=body_site)
            for m in self.models
        ]
        lead = results[0]
        raw_probs = [float(r.malignancy_probability) for r in results]
        mean_prob = float(np.mean(raw_probs))
        mod_key = str(modality_hint or getattr(lead, "predicted_modality", "dermoscopic")).lower()
        temp = float(self.modality_temperatures.get(mod_key, 1.0))
        cal_prob = _apply_temperature(mean_prob, temp)

        # Aggregate diagnosis distributions across ensemble members
        dx_sums: dict[str, float] = {}
        for r in results:
            for item in getattr(r, "top3_diagnoses", []):
                name = str(item["diagnosis"])
                dx_sums[name] = dx_sums.get(name, 0.0) + float(item["probability"])
        top3_pairs = sorted(dx_sums.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3: list[dict[str, Any]] = [
            {"diagnosis": k, "probability": round(v / len(results), 4)} for k, v in top3_pairs
        ]

        if cal_prob >= 0.65:
            risk_band = "high"
        elif cal_prob <= 0.35:
            risk_band = "low"
        else:
            risk_band = "indeterminate"

        abstained = bool(lead.abstained) or (risk_band == "indeterminate" and mod_key == "dermoscopic")
        return replace(
            lead,
            malignancy_probability=cal_prob,
            malignancy_risk_band=risk_band if not abstained else "abstained",
            abstained=abstained,
            top3_diagnoses=top3 or lead.top3_diagnoses,
            model_version=self.model_version,
        )
