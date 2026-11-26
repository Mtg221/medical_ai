"""Train & benchmark unified hospital NER model: From-Scratch PyTorch Char-CNN + BiLSTM + Partial-Label Marginal CRF + Multi-Seed Dual-Expert Ensemble.

Runs automatically on CUDA GPU when available (e.g., Kaggle T4/P100) and saves CPU-loadable checkpoints with SHA-256 verification.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.approaches.context_negex import apply_context_and_relations
from medresearch_platform.approaches.torch_ner import (
    PyTorchCharCNNBiLSTMCRF,
    ensemble_torch_ner_predictions,
)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train from-scratch hospital Partial-Label Char-CNN-BiLSTM-CRF (CPU or CUDA GPU).")
    p.add_argument("--seeds", type=str, default="1", help="Comma-separated random seeds (e.g. 1,2,3 on GPU).")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--word-emb-dim", type=int, default=48)
    p.add_argument("--char-emb-dim", type=int, default=16)
    p.add_argument("--char-filters", type=int, default=16)
    p.add_argument("--cluster-emb-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--ocr-augment-ratio", type=float, default=0.15)
    p.add_argument("--device", type=str, default="auto", help="'auto' (CUDA if available else CPU), 'cuda', or 'cpu'.")
    p.add_argument("--open-dir", type=Path, default=Path("data/processed/open_clinical"))
    p.add_argument("--out-models-dir", type=Path, default=Path("models"))
    p.add_argument("--out-report", type=Path, default=Path("docs/results/hospital_partial_crf_benchmark.json"))
    p.add_argument(
        "--clinical-eval-only",
        action="store_true",
        help="Evaluate open_clinical_test only on clinical sources (maccrobat2020, pxcorpus, e3c_layer1) if biomedical corpora were added.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    t0 = time.perf_counter()
    ev = Evaluator()

    synth_train = _load_jsonl(Path("data/processed/v3_1/train.jsonl"))
    synth_val = _load_jsonl(Path("data/processed/v3_1/val.jsonl"))
    synth_unseen = _load_jsonl(Path("data/processed/v3_1/test_unseen_terms.jsonl"))
    open_train = _load_jsonl(args.open_dir / "train.jsonl")
    open_test_all = _load_jsonl(args.open_dir / "test.jsonl")
    llm_80 = _load_jsonl(Path("data/processed/realistic_llm/eval.jsonl"))

    open_test = (
        [d for d in open_test_all if d.source in {"maccrobat2020", "pxcorpus", "e3c_layer1"}]
        if args.clinical_eval_only
        else open_test_all
    )

    unified_train = synth_train + open_train
    src_counts: dict[str, int] = {}
    for d in unified_train:
        src_counts[d.source] = src_counts.get(d.source, 0) + 1
    print(
        f"Training {len(seeds)}-seed hospital Char-CNN-BiLSTM-PartialCRF (device={args.device}) "
        f"on {len(unified_train)} EN+FR docs: {src_counts}",
        flush=True,
    )

    torch_models: list[PyTorchCharCNNBiLSTMCRF] = []
    seed_meta: list[dict[str, Any]] = []
    t_tr_total = time.perf_counter()
    for sd in seeds:
        t_sd = time.perf_counter()
        m = PyTorchCharCNNBiLSTMCRF(
            word_emb_dim=args.word_emb_dim,
            char_emb_dim=args.char_emb_dim,
            char_filters=args.char_filters,
            cluster_emb_dim=args.cluster_emb_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_self_supervised=True,
            ocr_augment_ratio=args.ocr_augment_ratio,
            seed=sd,
            device=args.device,
        ).fit(unified_train, verbose=True)
        sd_sec = round(time.perf_counter() - t_sd, 2)
        saved_dir = m.save(args.out_models_dir / f"hospital_char_cnn_bilstm_crf_seed{sd}")
        print(f"Saved {saved_dir} (device={m.training_device}) in {sd_sec}s", flush=True)
        torch_models.append(m)
        seed_meta.append(
            {
                "seed": sd,
                "training_device": m.training_device,
                "train_seconds": sd_sec,
                "initial_weights_sha256": m.initial_weights_sha256,
                "trained_weights_sha256": m.trained_weights_sha256,
                "saved_dir": str(saved_dir),
            }
        )
    torch_train_s = round(time.perf_counter() - t_tr_total, 2)

    # Also fit upgraded full-schema CRF (with E3C Layer 3 clusters + OCR aug) for ensemble comparison
    t_crf = time.perf_counter()
    crf_synth = ClassicalCRFModel(
        seed=1, cfg={"use_self_supervised": True, "ocr_augment_ratio": 0.30}
    ).fit(synth_train)
    open_clinical_only_train = [
        d for d in open_train if d.source in {"maccrobat2020", "pxcorpus", "e3c_layer1"}
    ] or open_train
    crf_open = ClassicalCRFModel(
        seed=1, cfg={"use_self_supervised": True, "ocr_augment_ratio": 0.15}
    ).fit(open_clinical_only_train)
    crf_train_s = round(time.perf_counter() - t_crf, 2)

    splits = {
        "synth_val": synth_val,
        "synth_unseen_terms": synth_unseen,
        "open_clinical_test": open_test,
        "realistic_llm_80": llm_80,
    }

    report: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            f"post-hoc (test seen) — Hospital {len(seeds)}-seed Partial-Label Marginal Char-CNN-BiLSTM-CRF "
            f"trained from scratch on {len(unified_train)} EN+FR docs"
        ),
        "pretrained_weights_used": False,
        "seeds": seeds,
        "train_source_counts": src_counts,
        "hyperparams": {
            "word_emb_dim": args.word_emb_dim,
            "char_emb_dim": args.char_emb_dim,
            "char_filters": args.char_filters,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        "models": seed_meta,
        "initial_weights_sha256": seed_meta[0]["initial_weights_sha256"],
        "trained_weights_sha256": seed_meta[0]["trained_weights_sha256"],
        "torch_train_seconds": torch_train_s,
        "crf_train_seconds": crf_train_s,
        "splits": {},
    }

    for sp_name, sp_docs in splits.items():
        preds_torch: list[DocumentPrediction] = []
        preds_ens: list[DocumentPrediction] = []
        for d in sp_docs:
            crf_pred = (crf_open if sp_name == "open_clinical_test" else crf_synth).predict_document(d)
            t_ents = ensemble_torch_ner_predictions(torch_models, d.text)
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
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_report} in {report['total_elapsed_s']}s")


if __name__ == "__main__":
    main()
