"""Calibrated multi-seed PyTorchDermResNet ensemble with val-fitted per-modality temperature scaling."""

from __future__ import annotations

import json
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
        self.architecture_name = f"PyTorchDermResNet4Stage-Ensemble{len(models)}-v1.0.0"
        self.model_version = self.architecture_name
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
        loaded = [PyTorchDermResNet.load(d, device="cpu") for d in model_dirs]
        temps = modality_temperatures
        if temps is None and model_dirs:
            cal_path = model_dirs[0].parent / "derm_ensemble_calibration.json"
            if cal_path.exists():
                try:
                    raw = json.loads(cal_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict) and isinstance(raw.get("modality_temperatures"), dict):
                        temps = {str(k): float(v) for k, v in raw["modality_temperatures"].items()}
                except Exception:
                    temps = None
        return cls(loaded, modality_temperatures=temps)

    def compute_gradcam_heatmap(self, image: Image.Image) -> tuple[np.ndarray, float]:
        return self.models[0].compute_gradcam_heatmap(image)

    def render_gradcam_overlay_png(self, image: Image.Image, cam_grid: np.ndarray) -> bytes:
        return self.models[0].render_gradcam_overlay_png(image, cam_grid)

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
        conf = round(max(cal_prob, 1.0 - cal_prob), 4)

        # Aggregate diagnosis distributions across ensemble members
        dx_sums: dict[str, float] = {}
        for r in results:
            for item in getattr(r, "top3_diagnoses", []):
                name = str(item["diagnosis"])
                dx_sums[name] = dx_sums.get(name, 0.0) + float(item["probability"])
        top3_pairs = sorted(dx_sums.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3: list[dict[str, Any]] = [
            {"rank": r_idx + 1, "diagnosis": k, "probability": round(v / len(results), 4)}
            for r_idx, (k, v) in enumerate(top3_pairs)
        ]

        abstained = bool(lead.abstained) or (conf < float(self.models[0].abstain_min_confidence))
        if abstained:
            risk_band = "abstained"
        elif cal_prob >= 0.65:
            risk_band = "high"
        elif cal_prob >= 0.35:
            risk_band = "medium"
        else:
            risk_band = "low"

        return replace(
            lead,
            malignancy_probability=cal_prob,
            confidence=conf,
            risk_band=risk_band,
            abstained=abstained,
            top3_diagnoses=(top3 or lead.top3_diagnoses) if not abstained else [],
            model_name=self.architecture_name,
        )
