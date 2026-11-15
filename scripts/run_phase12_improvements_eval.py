"""Phase 12 evaluation: Hybrid v2 (CRF ensemble + ConText/NegEx), Realistic 80-note set, and OCR layout segmentation."""

from __future__ import annotations

import io
import json
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.data.realistic_v1 import save_realistic_eval_set
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.documents.ocr_benchmark import (
    compute_cer,
    compute_wer,
    render_noisy_scan_image,
    tesseract_available,
)
from medresearch_platform.documents.parsers import parse_document_bytes
from medresearch_platform.eval_harness.metrics import DocumentPrediction, Evaluator
from medresearch_platform.eval_harness.runner import APPROACH_REGISTRY
from scripts.run_v3_1_ocr_full_benchmark import TIERS, _load_split, _remap_prediction


def _eval_split_summary(
    docs: list[AnnotatedDocument],
    preds_by_seed: list[list[DocumentPrediction]],
) -> dict[str, Any]:
    evaluator = Evaluator()
    per_seed = [evaluator.evaluate(docs, preds) for preds in preds_by_seed]
    summary: dict[str, dict[str, float]] = {}
    for key in ("strict_f1", "lenient_f1", "assertion_macro_f1", "relation_f1", "timex_f1", "timex_norm_acc"):
        vals = [float(m.get(key, 0.0)) for m in per_seed]
        summary[key] = {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        }
    return {"summary": summary, "per_seed": per_seed}


def run_hybrid_v2_and_realistic() -> dict[str, Any]:
    realistic_path = save_realistic_eval_set("data/processed/realistic")
    v3_dir = Path("data/processed/v3_1")
    train_docs = _load_split(v3_dir / "train.jsonl")
    splits: dict[str, list[AnnotatedDocument]] = {
        "val": _load_split(v3_dir / "val.jsonl"),
        "test_iid": _load_split(v3_dir / "test_iid.jsonl"),
        "test_unseen_terms": _load_split(v3_dir / "test_unseen_terms.jsonl"),
        "test_unseen_templates": _load_split(v3_dir / "test_unseen_templates.jsonl"),
        "realistic_80": _load_split(realistic_path),
    }

    crf_cfg = yaml.safe_load(Path("configs/exp/v3_1/classical_crf.yaml").read_text(encoding="utf-8"))
    h1_cfg = yaml.safe_load(Path("configs/exp/v3_1/hybrid.yaml").read_text(encoding="utf-8"))
    h2_cfg = yaml.safe_load(Path("configs/exp/v3_2/hybrid_v2.yaml").read_text(encoding="utf-8"))

    models = [
        ("classical_crf", "classical_crf", crf_cfg),
        ("hybrid_v1", "hybrid", h1_cfg),
        ("hybrid_v2", "hybrid", h2_cfg),
    ]
    seeds = [1, 2, 3]
    all_eval_docs: list[AnnotatedDocument] = []
    slice_map: dict[str, tuple[int, int]] = {}
    offset = 0
    for sname, sdocs in splits.items():
        all_eval_docs.extend(sdocs)
        slice_map[sname] = (offset, offset + len(sdocs))
        offset += len(sdocs)

    results: dict[str, Any] = {str(s): {} for s in splits}
    per_model_preds: dict[str, list[list[DocumentPrediction]]] = {}

    for label, approach_key, cfg in models:
        fn = APPROACH_REGISTRY[approach_key]
        seed_preds: list[list[DocumentPrediction]] = []
        t_start = time.perf_counter()
        for seed in seeds:
            t0 = time.perf_counter()
            preds = fn(train_docs, all_eval_docs, seed, cfg)
            seed_preds.append(preds)
            print(f"[{label}][seed={seed}] {time.perf_counter() - t0:.1f}s", flush=True)
        per_model_preds[label] = seed_preds
        for sname, (lo, hi) in slice_map.items():
            results[sname][label] = _eval_split_summary(
                splits[sname], [sp[lo:hi] for sp in seed_preds]
            )
            results[sname][label]["mean_runtime_s"] = round((time.perf_counter() - t_start) / len(seeds), 2)

    out_path = Path("docs/results/phase12_hybrid_v2_and_realistic.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seeds": seeds, "splits": results}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_ocr_layout_benchmark() -> dict[str, Any]:
    ok, reason = tesseract_available(("eng", "fra"))
    if not ok:
        raise RuntimeError(f"Tesseract unavailable: {reason}")

    v3_dir = Path("data/processed/v3_1")
    train_docs = _load_split(v3_dir / "train.jsonl")
    test_iid = _load_split(v3_dir / "test_iid.jsonl")
    en_docs = [d for d in test_iid if d.language == "en"][:60]
    fr_docs = [d for d in test_iid if d.language == "fr"][:60]
    pages_docs = en_docs + fr_docs

    crf_cfg = yaml.safe_load(Path("configs/exp/v3_1/classical_crf.yaml").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    clean_docs: list[AnnotatedDocument] = []
    ocr_docs: list[AnnotatedDocument] = []
    cer_list: list[float] = []
    wer_list: list[float] = []
    tier_stats: dict[str, dict[str, Any]] = {name: {"cers": [], "wers": [], "idxs": []} for name, _ in TIERS}

    idx = 0
    for tier_idx, (tier_name, params) in enumerate(TIERS):
        slice_docs = en_docs[tier_idx * 10 : (tier_idx + 1) * 10] + fr_docs[tier_idx * 10 : (tier_idx + 1) * 10]
        for doc in slice_docs:
            img = render_noisy_scan_image(doc.text, seed=42 + tier_idx, **params)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            bundle = parse_document_bytes(buf.getvalue(), "PNG")
            ocr_text = "\n".join(p.raw_text for p in bundle.pages)
            cer = compute_cer(doc.text, ocr_text)
            wer = compute_wer(doc.text, ocr_text)
            cer_list.append(cer)
            wer_list.append(wer)
            tier_stats[tier_name]["cers"].append(cer)
            tier_stats[tier_name]["wers"].append(wer)
            tier_stats[tier_name]["idxs"].append(idx)
            clean_docs.append(doc)
            ocr_docs.append(
                AnnotatedDocument(
                    doc_id=f"{doc.doc_id}__ocr",
                    patient_id=doc.patient_id,
                    language=doc.language,
                    doc_type=doc.doc_type,
                    document_date=doc.document_date,
                    source="ocr_layout",
                    text=ocr_text,
                )
            )
            idx += 1
            if idx % 30 == 0:
                print(f"[OCR layout] processed {idx}/{len(pages_docs)} pages", flush=True)

    raw_ocr_preds = APPROACH_REGISTRY["classical_crf"](train_docs, ocr_docs, 42, crf_cfg)
    crf_ocr = [
        _remap_prediction(pred, ocr_doc.text, clean_doc)
        for pred, clean_doc, ocr_doc in zip(raw_ocr_preds, clean_docs, ocr_docs, strict=True)
    ]
    crf_clean = APPROACH_REGISTRY["classical_crf"](train_docs, clean_docs, 42, crf_cfg)
    ev = Evaluator()
    clean_metrics = ev.evaluate(clean_docs, crf_clean)
    ocr_metrics = ev.evaluate(clean_docs, crf_ocr)

    by_tier = {}
    for tier_name, st in tier_stats.items():
        idxs = st["idxs"]
        sub_clean_m = ev.evaluate([clean_docs[k] for k in idxs], [crf_clean[k] for k in idxs])
        sub_ocr_m = ev.evaluate([clean_docs[k] for k in idxs], [crf_ocr[k] for k in idxs])
        by_tier[tier_name] = {
            "pages": len(idxs),
            "mean_cer": round(sum(st["cers"]) / len(st["cers"]), 4),
            "mean_wer": round(sum(st["wers"]) / len(st["wers"]), 4),
            "classical_crf_clean_f1": round(sub_clean_m["strict_span"]["f1"], 4),
            "classical_crf_ocr_f1": round(sub_ocr_m["strict_span"]["f1"], 4),
            "classical_crf_f1_drop": round(sub_ocr_m["strict_span"]["f1"] - sub_clean_m["strict_span"]["f1"], 4),
        }

    out = {
        "total_pages": len(pages_docs),
        "runtime_s": round(time.perf_counter() - t0, 2),
        "overall": {
            "mean_cer": round(sum(cer_list) / len(cer_list), 4),
            "mean_wer": round(sum(wer_list) / len(wer_list), 4),
            "classical_crf_clean_f1": round(clean_metrics["strict_span"]["f1"], 4),
            "classical_crf_ocr_f1": round(ocr_metrics["strict_span"]["f1"], 4),
            "classical_crf_f1_drop": round(ocr_metrics["strict_span"]["f1"] - clean_metrics["strict_span"]["f1"], 4),
        },
        "by_tier": by_tier,
    }
    Path("docs/results/v3_2_ocr_layout_benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys

    if "--ocr-only" not in sys.argv:
        run_hybrid_v2_and_realistic()
    run_ocr_layout_benchmark()
