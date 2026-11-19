"""Zero-shot confirmation evaluation of `models/vision_router_v1` (+ text-line safety guard) on external real medical documents."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from medresearch_platform.vision.quality import detect_dense_text_lines
from medresearch_platform.vision.router_torch import (
    PyTorchPictureTriageRouter,
    _predict_with_text_safety_guard,
)


def evaluate_external_documents(
    image_dir: str | Path = "data/router_real/external_cord_v2_phone",
    model_dir: str | Path = "models/vision_router_v1",
    out_json: str | Path = "docs/results/vision/step3_router_external_cord_phone_confirm.json",
    dataset_label: str = "naver-clova-ix/cord-v2 (CC-BY-4.0, real smartphone photos of printed paper, zero-shot held-out)",
) -> dict[str, Any]:
    img_root = Path(image_dir)
    paths = sorted(
        p for p in img_root.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {img_root}")

    router = PyTorchPictureTriageRouter.load(model_dir)
    raw_counts: Counter[str] = Counter()
    served_counts: Counter[str] = Counter()
    guard_rescues = 0
    leaked_to_skin: list[dict[str, Any]] = []
    routed_to_other: list[dict[str, Any]] = []

    for path in paths:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            dec = router.triage_image(rgb)
            raw_cls, served_cls = _predict_with_text_safety_guard(router, rgb)
            text_diag = detect_dense_text_lines(rgb) if raw_cls.startswith("skin_") else None

        raw_counts[raw_cls] += 1
        served_counts[served_cls] += 1
        if raw_cls.startswith("skin_") and served_cls == "document_scan":
            guard_rescues += 1
        if served_cls.startswith("skin_"):
            leaked_to_skin.append(
                {
                    "filename": path.name,
                    "raw_class": raw_cls,
                    "served_class": served_cls,
                    "confidence": dec.confidence,
                    "probabilities": dec.probabilities,
                    "route_action": dec.route_action,
                    "abstained": dec.abstained,
                    "text_guard": text_diag,
                }
            )
        elif served_cls == "other":
            routed_to_other.append(
                {
                    "filename": path.name,
                    "confidence": dec.confidence,
                    "probabilities": dec.probabilities,
                }
            )

    n = len(paths)
    raw_skin = sum(raw_counts[c] for c in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile"))
    served_skin = sum(served_counts[c] for c in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile"))
    report: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": dataset_label,
        "model_dir": str(model_dir),
        "zero_shot_held_out": True,
        "owner_phone_photos": False,
        "total_images": n,
        "served_document_scan": int(served_counts["document_scan"]),
        "served_document_scan_rate": round(served_counts["document_scan"] / n, 4),
        "served_other": int(served_counts["other"]),
        "served_routed_to_skin": int(served_skin),
        "served_skin_leak_rate": round(served_skin / n, 4),
        "raw_router_routed_to_skin": int(raw_skin),
        "raw_router_skin_leak_rate": round(raw_skin / n, 4),
        "text_guard_rescues": int(guard_rescues),
        "served_predictions": dict(served_counts),
        "raw_predictions": dict(raw_counts),
        "skin_leaks_sample": leaked_to_skin[:20],
        "routed_to_other_count": len(routed_to_other),
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default="data/router_real/external_cord_v2_phone")
    parser.add_argument("--model-dir", default="models/vision_router_v1")
    parser.add_argument("--out-json", default="docs/results/vision/step3_router_external_cord_phone_confirm.json")
    parser.add_argument(
        "--dataset-label",
        default="naver-clova-ix/cord-v2 (CC-BY-4.0, real smartphone photos of printed paper, zero-shot held-out)",
    )
    args = parser.parse_args()
    report = evaluate_external_documents(args.image_dir, args.model_dir, args.out_json, args.dataset_label)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
