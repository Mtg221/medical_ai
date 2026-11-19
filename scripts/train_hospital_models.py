"""Train & benchmark unified hospital NER model: From-Scratch PyTorch Char-CNN + BiLSTM + Partial-Label Marginal CRF + Dual-Expert Ensemble."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.approaches.context_negex import apply_context_and_relations
from medresearch_platform.approaches.torch_ner import PyTorchCharCNNBiLSTMCRF
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import DocumentPrediction, Evaluator, PredictedEntity
from medresearch_platform.eval_harness.runner import _load_jsonl


def _merge_hospital_ensemble(
    doc: AnnotatedDocument,
    torch_ents: list[PredictedEntity],
    crf_ents: list[PredictedEntity],
) -> list[PredictedEntity]:
    """High-precision confidence-weighted union of PyTorch Char-CNN-BiLSTM-CRF and upgraded CRF."""
    merged: dict[tuple[str, int, int], PredictedEntity] = {}
    for e in torch_ents:
        merged[(e.type, e.start, e.end)] = e
    for ce in crf_ents:
        key = (ce.type, ce.start, ce.end)
        if key in merged:
            prev = merged[key]
            merged[key] = PredictedEntity(
                type=prev.type,
                start=prev.start,
                end=prev.end,
                text=prev.text,
                assertion=prev.assertion,
                confidence=round(min(0.99, max(prev.confidence, ce.confidence) + 0.05), 4),
            )
        else:
            overlapping = [
                k for k, m in merged.items() if max(ce.start, m.start) < min(ce.end, m.end)
            ]
            if not overlapping and ce.confidence >= 0.72:
                merged[key] = ce
            elif overlapping and all(ce.confidence > merged[k].confidence + 0.12 for k in overlapping):
                for k in overlapping:
                    del merged[k]
                merged[key] = ce
    return sorted(merged.values(), key=lambda x: x.start)


def _eval_split(
    ev: Evaluator,
    docs: list[AnnotatedDocument],
    preds: list[DocumentPrediction],
) -> dict[str, Any]:
    res = ev.evaluate(docs, preds)
    out: dict[str, Any] = {
        "num_docs": len(docs),
        "strict_f1": res["strict_span"]["f1"],
        "lenient_f1": res["lenient_span"]["f1"],
        "precision": res["strict_span"]["precision"],
        "recall": res["strict_span"]["recall"],
        "assertion_macro_f1": res["assertion"]["macro_f1"],
        "doc_acc": res["doc_classification"]["accuracy"],
        "en_f1": res["per_language"].get("en", {}).get("f1"),
        "fr_f1": res["per_language"].get("fr", {}).get("f1"),
    }
    sources = sorted({d.source for d in docs})
    if len(sources) > 1:
        pmap = {p.doc_id: p for p in preds}
        by_src = {}
        for src in sources:
            sd = [d for d in docs if d.source == src]
            sm = ev.evaluate(sd, [pmap[d.doc_id] for d in sd])
            by_src[src] = {
                "strict_f1": sm["strict_span"]["f1"],
                "lenient_f1": sm["lenient_span"]["f1"],
            }
        out["by_corpus"] = by_src
    return out


def main() -> None:
    t0 = time.perf_counter()
    ev = Evaluator()
    synth_train = _load_jsonl(Path("data/processed/v3_1/train.jsonl"))
    synth_val = _load_jsonl(Path("data/processed/v3_1/val.jsonl"))
    synth_unseen = _load_jsonl(Path("data/processed/v3_1/test_unseen_terms.jsonl"))
    open_train = _load_jsonl(Path("data/processed/open_clinical/train.jsonl"))
    open_test = _load_jsonl(Path("data/processed/open_clinical/test.jsonl"))
    llm_80 = _load_jsonl(Path("data/processed/realistic_llm/eval.jsonl"))

    unified_train = synth_train + open_train
    print(f"Training unified hospital Char-CNN-BiLSTM-PartialCRF on {len(unified_train)} EN+FR docs...", flush=True)

    t_tr = time.perf_counter()
    torch_model = PyTorchCharCNNBiLSTMCRF(
        word_emb_dim=48,
        char_emb_dim=16,
        char_filters=16,
        cluster_emb_dim=16,
        hidden_dim=96,
        num_layers=2,
        dropout=0.25,
        lr=0.003,
        epochs=14,
        batch_size=16,
        use_self_supervised=True,
        ocr_augment_ratio=0.15,
        seed=1,
    ).fit(unified_train, verbose=True)
    torch_train_s = round(time.perf_counter() - t_tr, 2)
    saved_dir = torch_model.save(Path("models/hospital_char_cnn_bilstm_crf_seed1"))
    print(f"Saved {saved_dir} in {torch_train_s}s", flush=True)

    # Also fit upgraded full-schema CRF (with E3C Layer 3 clusters + OCR aug) for ensemble comparison
    t_crf = time.perf_counter()
    crf_synth = ClassicalCRFModel(
        seed=1, cfg={"use_self_supervised": True, "ocr_augment_ratio": 0.30}
    ).fit(synth_train)
    crf_open = ClassicalCRFModel(
        seed=1, cfg={"use_self_supervised": True, "ocr_augment_ratio": 0.15}
    ).fit(open_train)
    crf_train_s = round(time.perf_counter() - t_crf, 2)

    splits = {
        "synth_val": synth_val,
        "synth_unseen_terms": synth_unseen,
        "open_clinical_test": open_test,
        "realistic_llm_80": llm_80,
    }

    report: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "post-hoc (test seen) — Hospital unified Partial-Label Marginal Char-CNN-BiLSTM-CRF trained from scratch on SYNTH-MED-V3.1 (600) + open_clinical (1,645)",
        "pretrained_weights_used": False,
        "initial_weights_sha256": torch_model.initial_weights_sha256,
        "trained_weights_sha256": torch_model.trained_weights_sha256,
        "torch_train_seconds": torch_train_s,
        "crf_train_seconds": crf_train_s,
        "splits": {},
    }

    for sp_name, sp_docs in splits.items():
        preds_torch: list[DocumentPrediction] = []
        preds_ens: list[DocumentPrediction] = []
        for d in sp_docs:
            crf_pred = (crf_open if sp_name == "open_clinical_test" else crf_synth).predict_document(d)
            t_ents = torch_model.predict_entities(d.text)
            t_enr, t_rel, t_tx = apply_context_and_relations(d, t_ents)
            preds_torch.append(
                DocumentPrediction(
                    doc_id=d.doc_id,
                    language=d.language,
                    doc_type=d.doc_type,
                    predicted_doc_type=crf_pred.predicted_doc_type,
                    entities=t_enr,
                    relations=t_rel,
                    timex=t_tx,
                )
            )
            e_ents = _merge_hospital_ensemble(d, t_ents, crf_pred.entities)
            e_enr, e_rel, e_tx = apply_context_and_relations(d, e_ents)
            preds_ens.append(
                DocumentPrediction(
                    doc_id=d.doc_id,
                    language=d.language,
                    doc_type=d.doc_type,
                    predicted_doc_type=crf_pred.predicted_doc_type,
                    entities=e_enr,
                    relations=e_rel,
                    timex=e_tx,
                )
            )
        m_torch = _eval_split(ev, sp_docs, preds_torch)
        m_ens = _eval_split(ev, sp_docs, preds_ens)
        report["splits"][sp_name] = {
            "torch_partial_crf": m_torch,
            "hospital_ensemble_torch_plus_crf": m_ens,
        }
        print(
            f"[{sp_name:20s}] torch_partial_crf={m_torch['strict_f1']:.4f} (len={m_torch['lenient_f1']:.4f}) | "
            f"ensemble={m_ens['strict_f1']:.4f} (len={m_ens['lenient_f1']:.4f})",
            flush=True,
        )

    report["total_elapsed_s"] = round(time.perf_counter() - t0, 2)
    out = Path("docs/results/hospital_partial_crf_benchmark.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out} in {report['total_elapsed_s']}s")


if __name__ == "__main__":
    main()
