"""Full-size (>= 60 pages/language) OCR stress + end-to-end extraction benchmark on SYNTH-MED-V3.1."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.documents.ocr_benchmark import (
    compute_cer,
    compute_wer,
    map_ocr_spans_to_reference,
    render_noisy_scan_image,
    tesseract_available,
)
from medresearch_platform.documents.parsers import parse_document_bytes
from medresearch_platform.eval_harness.metrics import DocumentPrediction, Evaluator, PredictedEntity
from medresearch_platform.eval_harness.runner import APPROACH_REGISTRY
from medresearch_platform.hybrid.pipeline import HybridPipeline

TIERS: list[tuple[str, dict[str, Any]]] = [
    (
        "tier_1_300dpi_clean",
        {"dpi": 300, "skew_deg": 0.5, "noise_std": 2.0, "blur_radius": 0.0, "jpeg_quality": 90, "font_index": 0},
    ),
    (
        "tier_2_200dpi_mild_skew_jpeg",
        {"dpi": 200, "skew_deg": 2.5, "noise_std": 6.0, "blur_radius": 0.4, "jpeg_quality": 70, "font_index": 1},
    ),
    (
        "tier_3_phone_perspective_light",
        {
            "dpi": 200,
            "skew_deg": -3.5,
            "noise_std": 8.0,
            "blur_radius": 0.3,
            "jpeg_quality": 60,
            "perspective": 0.35,
            "uneven_lighting": 0.35,
            "font_index": 2,
        },
    ),
    (
        "tier_4_two_col_table_stamps",
        {
            "dpi": 200,
            "skew_deg": 1.5,
            "noise_std": 5.0,
            "jpeg_quality": 75,
            "stamps_stains": True,
            "two_columns": True,
            "font_index": 0,
        },
    ),
    (
        "tier_5_table_multifont_skew5",
        {
            "dpi": 200,
            "skew_deg": 5.0,
            "noise_std": 10.0,
            "blur_radius": 0.5,
            "jpeg_quality": 55,
            "table_mode": True,
            "stamps_stains": True,
            "font_index": 1,
        },
    ),
    (
        "tier_6_150dpi_heavy_blur_noise",
        {
            "dpi": 150,
            "skew_deg": -5.0,
            "noise_std": 18.0,
            "blur_radius": 1.0,
            "jpeg_quality": 35,
            "contrast": 0.82,
            "uneven_lighting": 0.25,
            "font_index": 2,
        },
    ),
]


def _load_split(path: Path) -> list[AnnotatedDocument]:
    with path.open("r", encoding="utf-8") as fh:
        return [AnnotatedDocument.from_dict(json.loads(line)) for line in fh if line.strip()]


def _remap_prediction(
    pred: DocumentPrediction, ocr_text: str, clean_doc: AnnotatedDocument
) -> DocumentPrediction:
    spans = [(e.start, e.end) for e in pred.entities]
    mapped = map_ocr_spans_to_reference(ocr_text, clean_doc.text, spans)
    remapped_ents: list[PredictedEntity] = []
    for ent, m in zip(pred.entities, mapped):
        if m is None:
            continue
        rs, re_ = m
        remapped_ents.append(
            PredictedEntity(
                type=ent.type,
                start=rs,
                end=re_,
                text=clean_doc.text[rs:re_],
                assertion=ent.assertion,
                confidence=ent.confidence,
            )
        )
    return DocumentPrediction(
        doc_id=clean_doc.doc_id,
        language=clean_doc.language,
        doc_type=pred.doc_type,
        entities=remapped_ents,
        relations=pred.relations,
        timex=pred.timex,
    )


def main() -> None:
    ok, reason = tesseract_available(("eng", "fra"))
    if not ok:
        raise RuntimeError(f"Tesseract prerequisite missing: {reason}")

    t_start = time.perf_counter()
    data_dir = Path("data/processed/v3_1")
    train_docs = _load_split(data_dir / "train.jsonl")
    test_docs = _load_split(data_dir / "test_iid.jsonl")

    en_docs = [d for d in test_docs if d.language == "en"][:60]
    fr_docs = [d for d in test_docs if d.language == "fr"][:60]
    assert len(en_docs) == 60 and len(fr_docs) == 60

    # Assign 10 EN + 10 FR pages to each of the 6 degradation tiers (120 pages total)
    page_records: list[dict[str, Any]] = []
    ocr_docs: list[AnnotatedDocument] = []
    clean_docs: list[AnnotatedDocument] = []
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
            boxes = sum(len(p.word_boxes) for p in bundle.pages)

            ocr_doc = AnnotatedDocument(
                doc_id=f"{doc.doc_id}__ocr",
                patient_id=doc.patient_id,
                document_date=doc.document_date,
                source="ocr",
                language=doc.language,
                doc_type=doc.doc_type,
                text=ocr_text,
            )
            clean_docs.append(doc)
            ocr_docs.append(ocr_doc)
            page_records.append(
                {
                    "doc_id": doc.doc_id,
                    "language": doc.language,
                    "tier": tier_name,
                    "cer": cer,
                    "wer": wer,
                    "word_boxes": boxes,
                    "ocr_text": ocr_text,
                }
            )
        tier_cer = sum(r["cer"] for r in page_records[-20:]) / 20.0
        tier_wer = sum(r["wer"] for r in page_records[-20:]) / 20.0
        print(f"[{tier_name}] 20 pages (10 EN + 10 FR): mean CER={tier_cer:.4f}, mean WER={tier_wer:.4f}", flush=True)

    # Train best single (classical_crf) and hybrid on V3.1 train (seed=42)
    crf_cfg = yaml.safe_load(Path("configs/exp/v3_1/classical_crf.yaml").read_text(encoding="utf-8"))
    hybrid_cfg = yaml.safe_load(Path("configs/exp/v3_1/hybrid.yaml").read_text(encoding="utf-8"))

    crf_preds_all = APPROACH_REGISTRY["classical_crf"](train_docs, clean_docs + ocr_docs, 42, crf_cfg)
    crf_clean_preds = crf_preds_all[: len(clean_docs)]
    crf_ocr_preds = [
        _remap_prediction(p, rec["ocr_text"], cdoc)
        for p, rec, cdoc in zip(crf_preds_all[len(clean_docs) :], page_records, clean_docs)
    ]

    pipe = HybridPipeline(seed=42, cfg=hybrid_cfg).fit(train_docs)
    hyb_clean_preds = [pipe.extract_document(d)[0] for d in clean_docs]
    hyb_ocr_preds = [
        _remap_prediction(pipe.extract_document(odoc)[0], rec["ocr_text"], cdoc)
        for odoc, rec, cdoc in zip(ocr_docs, page_records, clean_docs)
    ]

    ev = Evaluator()

    def _score_subset(indices: list[int]) -> dict[str, Any]:
        sub_gold = [clean_docs[i] for i in indices]
        crf_c = float(ev.evaluate(sub_gold, [crf_clean_preds[i] for i in indices])["strict_span"]["f1"])
        crf_o = float(ev.evaluate(sub_gold, [crf_ocr_preds[i] for i in indices])["strict_span"]["f1"])
        hyb_c = float(ev.evaluate(sub_gold, [hyb_clean_preds[i] for i in indices])["strict_span"]["f1"])
        hyb_o = float(ev.evaluate(sub_gold, [hyb_ocr_preds[i] for i in indices])["strict_span"]["f1"])
        cers = [page_records[i]["cer"] for i in indices]
        wers = [page_records[i]["wer"] for i in indices]
        boxes = [page_records[i]["word_boxes"] for i in indices]
        return {
            "pages": len(indices),
            "mean_cer": round(sum(cers) / len(cers), 4),
            "mean_wer": round(sum(wers) / len(wers), 4),
            "mean_word_boxes": round(sum(boxes) / len(boxes), 1),
            "classical_crf_clean_f1": round(crf_c, 4),
            "classical_crf_ocr_f1": round(crf_o, 4),
            "classical_crf_f1_drop": round(crf_o - crf_c, 4),
            "hybrid_clean_f1": round(hyb_c, 4),
            "hybrid_ocr_f1": round(hyb_o, 4),
            "hybrid_f1_drop": round(hyb_o - hyb_c, 4),
        }

    by_tier: dict[str, Any] = {}
    for tier_name, _ in TIERS:
        idxs = [i for i, r in enumerate(page_records) if r["tier"] == tier_name]
        by_tier[tier_name] = _score_subset(idxs)

    by_language: dict[str, Any] = {}
    for lang in ("en", "fr"):
        idxs = [i for i, r in enumerate(page_records) if r["language"] == lang]
        by_language[lang] = _score_subset(idxs)

    overall = _score_subset(list(range(len(page_records))))

    payload = {
        "dataset": "SYNTH-MED-V3.1",
        "split": "test_iid",
        "pages_per_language": {"en": 60, "fr": 60},
        "total_pages": len(page_records),
        "runtime_s": round(time.perf_counter() - t_start, 2),
        "overall": overall,
        "by_language": by_language,
        "by_tier": by_tier,
        "pages": [
            {k: v for k, v in r.items() if k != "ocr_text"} for r in page_records
        ],
    }

    out_path = Path("docs/results/v3_1_ocr_full_benchmark.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved full OCR benchmark ({len(page_records)} pages) to {out_path} in {payload['runtime_s']}s")
    print("Overall:", json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
