"""Run realistic multi-tier OCR stress benchmark on SYNTH-MED-V3 val documents (F6)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.documents.ocr_benchmark import (
    OCRBenchmarkRow,
    parse_and_benchmark_scan,
)

TIERS: dict[str, dict[str, Any]] = {
    "clean_scan": {"skew_deg": 0.5, "noise_std": 3.0, "blur_radius": 0.0, "jpeg_quality": None, "contrast": 1.0},
    "mild_scan": {"skew_deg": 2.5, "noise_std": 12.0, "blur_radius": 0.6, "jpeg_quality": 75, "contrast": 0.95},
    "heavy_scan": {"skew_deg": 4.5, "noise_std": 22.0, "blur_radius": 1.2, "jpeg_quality": 40, "contrast": 0.78},
}


def run_ocr_stress_benchmark(
    val_path: Path = Path("data/processed/v3/val.jsonl"),
    out_path: Path = Path("docs/results/ocr_stress_benchmark.json"),
    n_per_lang: int = 8,
) -> dict[str, Any]:
    docs = [AnnotatedDocument.from_dict(json.loads(line)) for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    en_docs = [d for d in docs if d.language == "en"][:n_per_lang]
    fr_docs = [d for d in docs if d.language == "fr"][:n_per_lang]
    sample_docs = en_docs + fr_docs

    tier_results: dict[str, Any] = {}
    for tier_name, params in TIERS.items():
        rows: list[OCRBenchmarkRow] = []
        for d in sample_docs:
            row = parse_and_benchmark_scan(
                d.doc_id,
                d.language,
                d.text,
                skew_deg=float(params["skew_deg"]),
                noise_std=float(params["noise_std"]),
                blur_radius=float(params["blur_radius"]),
                jpeg_quality=params["jpeg_quality"],
                contrast=float(params["contrast"]),
                as_pdf=True,
            )
            rows.append(row)

        by_lang: dict[str, dict[str, float]] = {}
        for lang in ("en", "fr"):
            l_rows = [r for r in rows if r.language == lang]
            by_lang[lang] = {
                "docs": float(len(l_rows)),
                "mean_cer": round(sum(r.cer for r in l_rows) / max(1, len(l_rows)), 4),
                "mean_wer": round(sum(r.wer for r in l_rows) / max(1, len(l_rows)), 4),
                "mean_confidence": round(sum(r.mean_confidence for r in l_rows) / max(1, len(l_rows)), 4),
            }

        tier_results[tier_name] = {
            "params": params,
            "docs": len(rows),
            "mean_cer": round(sum(r.cer for r in rows) / max(1, len(rows)), 4),
            "mean_wer": round(sum(r.wer for r in rows) / max(1, len(rows)), 4),
            "mean_confidence": round(sum(r.mean_confidence for r in rows) / max(1, len(rows)), 4),
            "by_language": by_lang,
            "rows": [asdict(r) for r in rows],
        }

    payload = {
        "corpus": "SYNTH-MED-V3",
        "split": "val",
        "docs_per_tier": len(sample_docs),
        "mean_words_per_doc": round(sum(len(d.text.split()) for d in sample_docs) / max(1, len(sample_docs)), 1),
        "tiers": tier_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    res = run_ocr_stress_benchmark()
    print(f"OCR Stress Benchmark ({res['docs_per_tier']} docs/tier, avg {res['mean_words_per_doc']} words/doc):")
    for tname, tdata in res["tiers"].items():
        print(
            f"  {tname:<12} | CER={tdata['mean_cer']:.4f} | WER={tdata['mean_wer']:.4f} "
            f"| conf={tdata['mean_confidence']:.4f} | EN CER={tdata['by_language']['en']['mean_cer']:.4f} "
            f"| FR CER={tdata['by_language']['fr']['mean_cer']:.4f}"
        )


if __name__ == "__main__":
    main()
