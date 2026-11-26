"""Reproducible CPU/Kaggle-GPU training & evaluation entry point for existing models on the EN/FR conversation dataset.

Strictly from-scratch (0 pretrained weights), preserves existing model architectures (`HybridPipeline`,
`ClassicalCRFModel`, `TorchCharCNNBiLSTMCRF`), and saves new versioned checkpoints without overwriting baselines.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.data.conversation_loader import load_conversation_dataset_from_zip
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import Evaluator
from medresearch_platform.hybrid.pipeline import HybridPipeline


def _load_jsonl(path: Path) -> list[AnnotatedDocument]:
    if not path.exists():
        return []
    return [AnnotatedDocument.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _eval_grouped(ev: Evaluator, docs: list[AnnotatedDocument], preds: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": ev.evaluate(docs, preds), "by_language": {}, "by_noise": {}}
    for lang in ("en", "fr", "mixed"):
        idxs = [i for i, d in enumerate(docs) if d.metadata.get("original_language", d.language) == lang]
        if idxs:
            out["by_language"][lang] = ev.evaluate([docs[i] for i in idxs], [preds[i] for i in idxs])
    for ns in ("clean", "moderate", "high"):
        idxs = [i for i, d in enumerate(docs) if d.metadata.get("noise_severity") == ns]
        if idxs:
            out["by_noise"][ns] = ev.evaluate([docs[i] for i in idxs], [preds[i] for i in idxs])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train existing models on bilingual EN/FR medical conversation dataset.")
    parser.add_argument("--zip-path", type=Path, default=Path("medical_conversation_dataset_en_fr.zip"))
    parser.add_argument("--synth-train", type=Path, default=Path("data/processed/v3_1/train.jsonl"))
    parser.add_argument("--synth-val", type=Path, default=Path("data/locked/synth_med_v3_1/val.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/eval/conversation_post_training_eval.json"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-torch", action="store_true", help="Also train existing PyTorch Char-CNN-BiLSTM-CRF (recommended on Kaggle GPU).")
    parser.add_argument("--torch-epochs", type=int, default=12)
    parser.add_argument("--torch-batch-size", type=int, default=16)
    parser.add_argument("--torch-lr", type=float, default=1e-3)
    args = parser.parse_args()

    ds = load_conversation_dataset_from_zip(args.zip_path)
    synth_train = _load_jsonl(args.synth_train)
    synth_val = _load_jsonl(args.synth_val)
    combined_train = list(synth_train) + list(ds.train)
    print(
        f"Loaded conversation dataset: train={len(ds.train)}, val={len(ds.validation)}, test={len(ds.test)}; "
        f"combined_train={len(combined_train)} (incl. {len(synth_train)} SYNTH-MED-V3.1 train)",
        flush=True,
    )

    ev = Evaluator()
    results: dict[str, Any] = {
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "pretrained_weights_used": False,
        "dataset_audit": ds.audit,
        "combined_train_n": len(combined_train),
        "models": {},
    }

    # 1. Train existing HybridPipeline (hybrid_v2) -> new versioned checkpoint
    cfg_path = Path("configs/exp/v3_2/hybrid_v2.yaml")
    cfg = dict(yaml.safe_load(cfg_path.read_text(encoding="utf-8"))) if cfg_path.exists() else {"epochs": 8, "aux_epochs": 5}
    t0 = time.perf_counter()
    hyb = HybridPipeline(seed=args.seed, cfg=cfg)
    hyb.fit(combined_train)
    hyb_train_s = time.perf_counter() - t0
    hyb_out = args.out_dir / f"hybrid_v2_conv_en_fr_seed{args.seed}"
    hyb.save(hyb_out, corpus_id="SYNTH-MED-V3.1+CONV-EN-FR-v1")

    t_load = time.perf_counter()
    hyb_reloaded = HybridPipeline.load(hyb_out)
    hyb_load_ms = (time.perf_counter() - t_load) * 1000.0
    t_inf = time.perf_counter()
    hyb_val_preds = [hyb_reloaded.extract_document(d)[0] for d in ds.validation]
    hyb_test_preds = [hyb_reloaded.extract_document(d)[0] for d in ds.test]
    hyb_inf_s = time.perf_counter() - t_inf
    hyb_synth_val_preds = [hyb_reloaded.extract_document(d)[0] for d in synth_val] if synth_val else []

    results["models"]["hybrid_v2_conv_en_fr"] = {
        "checkpoint_dir": str(hyb_out),
        "train_seconds": round(hyb_train_s, 2),
        "cpu_load_ms": round(hyb_load_ms, 1),
        "cpu_ms_per_doc": round(hyb_inf_s / (len(ds.validation) + len(ds.test)) * 1000.0, 1),
        "validation_122": _eval_grouped(ev, ds.validation, hyb_val_preds),
        "test_118": _eval_grouped(ev, ds.test, hyb_test_preds),
        "synth_med_v3_1_val_160": ev.evaluate(synth_val, hyb_synth_val_preds) if synth_val else {},
    }
    print(
        f"hybrid_v2_conv_en_fr trained in {hyb_train_s:.1f}s | TEST_118 Strict F1="
        f"{results['models']['hybrid_v2_conv_en_fr']['test_118']['overall']['strict_span']['f1']:.4f} | "
        f"Assert Acc={results['models']['hybrid_v2_conv_en_fr']['test_118']['overall']['assertion']['accuracy']:.4f} | "
        f"Rel F1={results['models']['hybrid_v2_conv_en_fr']['test_118']['overall']['relations']['f1']:.4f}",
        flush=True,
    )

    # 2. Train existing ClassicalCRFModel -> new versioned checkpoint
    t1 = time.perf_counter()
    crf = ClassicalCRFModel(seed=args.seed)
    crf.fit(combined_train)
    crf_train_s = time.perf_counter() - t1
    crf_out = args.out_dir / f"classical_crf_conv_en_fr_seed{args.seed}"
    crf.save(crf_out, corpus_id="SYNTH-MED-V3.1+CONV-EN-FR-v1")

    t_cload = time.perf_counter()
    crf_reloaded = ClassicalCRFModel.load(crf_out)
    crf_load_ms = (time.perf_counter() - t_cload) * 1000.0
    t_cinf = time.perf_counter()
    crf_val_preds = [crf_reloaded.predict_document(d) for d in ds.validation]
    crf_test_preds = [crf_reloaded.predict_document(d) for d in ds.test]
    crf_inf_s = time.perf_counter() - t_cinf
    crf_synth_val_preds = [crf_reloaded.predict_document(d) for d in synth_val] if synth_val else []

    results["models"]["classical_crf_conv_en_fr"] = {
        "checkpoint_dir": str(crf_out),
        "train_seconds": round(crf_train_s, 2),
        "cpu_load_ms": round(crf_load_ms, 1),
        "cpu_ms_per_doc": round(crf_inf_s / (len(ds.validation) + len(ds.test)) * 1000.0, 1),
        "validation_122": _eval_grouped(ev, ds.validation, crf_val_preds),
        "test_118": _eval_grouped(ev, ds.test, crf_test_preds),
        "synth_med_v3_1_val_160": ev.evaluate(synth_val, crf_synth_val_preds) if synth_val else {},
    }
    print(
        f"classical_crf_conv_en_fr trained in {crf_train_s:.1f}s | TEST_118 Strict F1="
        f"{results['models']['classical_crf_conv_en_fr']['test_118']['overall']['strict_span']['f1']:.4f} | "
        f"Assert Acc={results['models']['classical_crf_conv_en_fr']['test_118']['overall']['assertion']['accuracy']:.4f}",
        flush=True,
    )

    # 3. Optional GPU/CPU training of existing PyTorch Char-CNN-BiLSTM-CRF (uses existing torch_ner.py)
    if args.train_torch:
        import torch

        from medresearch_platform.approaches.torch_ner import PyTorchCharCNNBiLSTMCRF

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training existing PyTorchCharCNNBiLSTMCRF on device={device} (epochs={args.torch_epochs})...", flush=True)
        t2 = time.perf_counter()
        tmodel = PyTorchCharCNNBiLSTMCRF(
            seed=args.seed,
            epochs=args.torch_epochs,
            batch_size=args.torch_batch_size,
            lr=args.torch_lr,
        )
        tmodel.fit(combined_train)
        torch_train_s = time.perf_counter() - t2
        torch_out = args.out_dir / f"hospital_char_cnn_bilstm_crf_conv_en_fr_seed{args.seed}"
        tmodel.save(torch_out)
        t_reloaded = PyTorchCharCNNBiLSTMCRF.load(torch_out)
        from medresearch_platform.eval_harness.metrics import DocumentPrediction

        t_val_preds = [
            DocumentPrediction(doc_id=d.doc_id, language=d.language, doc_type=d.doc_type, entities=t_reloaded.predict_entities(d.text))
            for d in ds.validation
        ]
        t_test_preds = [
            DocumentPrediction(doc_id=d.doc_id, language=d.language, doc_type=d.doc_type, entities=t_reloaded.predict_entities(d.text))
            for d in ds.test
        ]
        results["models"]["torch_char_cnn_bilstm_crf_conv_en_fr"] = {
            "checkpoint_dir": str(torch_out),
            "device": device,
            "train_seconds": round(torch_train_s, 2),
            "validation_122": _eval_grouped(ev, ds.validation, t_val_preds),
            "test_118": _eval_grouped(ev, ds.test, t_test_preds),
        }

    results["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved post-training evaluation report to {args.report_out}", flush=True)


if __name__ == "__main__":
    main()
