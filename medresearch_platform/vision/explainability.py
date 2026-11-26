"""Grad-CAM overlay rendering and Adebayo et al. (2018) randomization sanity checks (QUAL-001)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from medresearch_platform.vision.catalog import DermCatalogRecord


def verify_gradcam_adebayo_and_lesion_center(
    model: Any,
    samples: list[tuple[np.ndarray, int, DermCatalogRecord]],
    *,
    max_images: int = 220,
) -> dict[str, float]:
    """
    1. Adebayo et al. (2018) model-parameter randomisation sanity test: compare Grad-CAM of trained model vs random-init model.
    2. On dermoscopic images, compute fraction of Grad-CAM mass inside the central lesion region (r <= 0.35*H).
    """
    rand_model = type(model)(
        image_size=model.image_size,
        include_metadata=model.include_metadata,
        seed=model.seed + 999,
    )
    l1_shifts: list[float] = []
    center_shares: list[float] = []

    h = model.image_size
    yy, xx = np.ogrid[:h, :h]
    cy, cx = h / 2.0, h / 2.0
    center_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (0.35 * h) ** 2

    for arr, _, rec in samples[:max_images]:
        cam_tr = model.compute_gradcam(arr, rec)
        cam_rd = rand_model.compute_gradcam(arr, rec)
        l1_shifts.append(float(np.mean(np.abs(cam_tr - cam_rd))))
        if rec.modality == "dermoscopic":
            tot = float(np.sum(cam_tr))
            if tot > 1e-8:
                center_shares.append(float(np.sum(cam_tr[center_mask]) / tot))

    return {
        "adebayo_mean_l1_shift": round(float(np.mean(l1_shifts)) if l1_shifts else 0.0, 4),
        "dermoscopic_center_heat_share": round(float(np.mean(center_shares)) if center_shares else 0.0, 4),
        "n_dermoscopic_checked": float(len(center_shares)),
    }


def render_gradcam_overlay_png_from_cam(arr: np.ndarray, cam_grid: np.ndarray) -> bytes:
    rgb_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    base_im = Image.fromarray(rgb_u8, mode="RGB")
    cam = np.clip(cam_grid, 0.0, 1.0)
    heat_rgb = np.zeros((cam.shape[0], cam.shape[1], 3), dtype=np.uint8)
    heat_rgb[..., 0] = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    heat_rgb[..., 1] = np.clip((1.0 - np.abs(cam - 0.5) * 2.0) * 180.0, 0, 255).astype(np.uint8)
    heat_rgb[..., 2] = np.clip((1.0 - cam) * 200.0, 0, 255).astype(np.uint8)
    heat_im = Image.fromarray(heat_rgb, mode="RGB").resize(base_im.size, Image.Resampling.BILINEAR)
    blended = Image.blend(base_im, heat_im, alpha=0.45)
    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    return buf.getvalue()
