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

UNCERTAINTY_HIGH_STD_THRESHOLD: float = 0.12


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
        use_tta: bool = False,
        uncertainty_high_std_threshold: float = UNCERTAINTY_HIGH_STD_THRESHOLD,
        specialist_indices: list[int] | None = None,
        modality_specialist_weights: dict[str, float] | None = None,
    ) -> None:
        if not models:
            raise ValueError("CalibratedVisionEnsemble requires at least one model.")
        self.models = models
        self.modality_temperatures = dict(modality_temperatures or DEFAULT_MODALITY_TEMPERATURES)
        self.use_tta = bool(use_tta)
        self.uncertainty_high_std_threshold = float(uncertainty_high_std_threshold)
        self.specialist_indices = sorted({i for i in (specialist_indices or []) if 0 <= i < len(models)})
        self.modality_specialist_weights = {
            str(k).lower(): float(v) for k, v in (modality_specialist_weights or {}).items()
        }
        unique_sizes = sorted({int(m.image_size) for m in models})
        if self.specialist_indices and len(unique_sizes) > 1:
            size_tag = "+".join(str(s) for s in unique_sizes)
            self.architecture_name = f"PyTorchDermSEResNetV2-ModalityAware({size_tag})-Ensemble{len(models)}-v2.1.0"
        elif len(unique_sizes) > 1:
            size_tag = "+".join(str(s) for s in unique_sizes)
            self.architecture_name = f"PyTorchDermSEResNetV2-MultiScale({size_tag})-Ensemble{len(models)}-v2.0.0"
        else:
            self.architecture_name = f"PyTorchDermResNet4Stage-Ensemble{len(models)}-v1.0.0"
        self.model_version = self.architecture_name
        self.backend = "pytorch_cpu"
        self.image_size = int(models[0].image_size)
        self.manifest = {
            **dict(getattr(models[0], "manifest", {})),
            "model_version": self.model_version,
            "ensemble_size": len(models),
            "image_sizes": unique_sizes,
            "modality_temperatures": self.modality_temperatures,
            "modality_specialist_weights": self.modality_specialist_weights,
            "specialist_count": len(self.specialist_indices),
            "use_tta": self.use_tta,
            "uncertainty_high_std_threshold": self.uncertainty_high_std_threshold,
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
        use_tta = False
        unc_thr = UNCERTAINTY_HIGH_STD_THRESHOLD
        spec_weights: dict[str, float] = {}
        spec_dir_names: set[str] = set()
        if model_dirs:
            cal_path = model_dirs[0].parent / "derm_ensemble_calibration.json"
            if cal_path.exists():
                try:
                    raw = json.loads(cal_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        if temps is None and isinstance(raw.get("modality_temperatures"), dict):
                            temps = {str(k): float(v) for k, v in raw["modality_temperatures"].items()}
                        use_tta = bool(raw.get("use_tta", False))
                        if "uncertainty_high_std_threshold" in raw:
                            unc_thr = float(raw["uncertainty_high_std_threshold"])
                        if isinstance(raw.get("modality_specialist_weights"), dict):
                            spec_weights = {
                                str(k).lower(): float(v) for k, v in raw["modality_specialist_weights"].items()
                            }
                        if isinstance(raw.get("clinical_specialist_dirs"), list):
                            spec_dir_names = {str(x) for x in raw["clinical_specialist_dirs"]}
                except Exception:
                    temps = temps
        spec_indices = [
            idx
            for idx, d in enumerate(model_dirs)
            if (d.name in spec_dir_names) or ("clinical" in d.name.lower())
        ]
        return cls(
            loaded,
            modality_temperatures=temps,
            use_tta=use_tta,
            uncertainty_high_std_threshold=unc_thr,
            specialist_indices=spec_indices,
            modality_specialist_weights=spec_weights,
        )

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
        mod_key = str(modality_hint or "dermoscopic").lower()
        n_total = len(self.models)
        spec_set = set(self.specialist_indices)
        gen_indices = [i for i in range(n_total) if i not in spec_set]

        if spec_set and gen_indices and self.modality_specialist_weights:
            alpha = float(np.clip(self.modality_specialist_weights.get(mod_key, 0.0), 0.0, 1.0))
            weights_all = [
                (alpha / len(spec_set)) if i in spec_set else ((1.0 - alpha) / len(gen_indices))
                for i in range(n_total)
            ]
        else:
            weights_all = [1.0 / n_total] * n_total

        active_pairs = [(idx, w) for idx, w in enumerate(weights_all) if w > 1e-6]
        if not active_pairs:
            active_pairs = [(i, 1.0 / n_total) for i in range(n_total)]

        use_clinical_roi = bool(self.use_tta and mod_key == "clinical")
        roi_image = None
        if use_clinical_roi:
            from medresearch_platform.vision.quality import crop_clinical_lesion_roi

            roi_image, _ = crop_clinical_lesion_roi(image)

        results: list[Any] = []
        active_weights: list[float] = []
        for idx, w in active_pairs:
            res_full = self.models[idx].predict_image(
                image, modality_hint=modality_hint, body_site=body_site, tta=self.use_tta
            )
            if roi_image is not None:
                res_roi = self.models[idx].predict_image(
                    roi_image, modality_hint=modality_hint, body_site=body_site, tta=self.use_tta
                )
                blended_p = round(0.5 * (float(res_full.malignancy_probability) + float(res_roi.malignancy_probability)), 4)
                res_full = replace(res_full, malignancy_probability=blended_p)
            results.append(res_full)
            active_weights.append(w)

        w_arr = np.asarray(active_weights, dtype=np.float64)
        w_arr = w_arr / float(np.sum(w_arr))
        lead = results[0]
        raw_probs = np.asarray([float(r.malignancy_probability) for r in results], dtype=np.float64)
        mean_prob = float(np.sum(w_arr * raw_probs))
        seed_std = round(float(np.sqrt(max(0.0, float(np.sum(w_arr * (raw_probs - mean_prob) ** 2))))), 4)
        is_high_uncertainty = bool(seed_std > self.uncertainty_high_std_threshold)
        eff_mod_key = str(modality_hint or getattr(lead, "predicted_modality", "dermoscopic")).lower()
        temp = float(self.modality_temperatures.get(eff_mod_key, 1.0))
        cal_prob = _apply_temperature(mean_prob, temp)
        conf = round(max(cal_prob, 1.0 - cal_prob), 4)

        # Aggregate diagnosis distributions across active ensemble members
        dx_sums: dict[str, float] = {}
        for w_i, r in zip(w_arr.tolist(), results, strict=True):
            for item in getattr(r, "top3_diagnoses", []):
                name = str(item["diagnosis"])
                dx_sums[name] = dx_sums.get(name, 0.0) + w_i * float(item["probability"])
        top3_pairs = sorted(dx_sums.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3: list[dict[str, Any]] = [
            {"rank": r_idx + 1, "diagnosis": k, "probability": round(v, 4)}
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
            needs_review=bool(lead.needs_review or abstained or is_high_uncertainty),
            top3_diagnoses=(top3 or lead.top3_diagnoses) if not abstained else [],
            model_name=self.architecture_name,
            confidence_variance=seed_std,
            high_uncertainty=is_high_uncertainty,
        )
