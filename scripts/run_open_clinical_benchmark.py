"""Multi-approach benchmark on real open-license EN/FR clinical corpora (MACCROBAT2020, PxCorpus, E3C)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel, predict_classical_ml
from medresearch_platform.approaches.neural_seq import predict_bilstm_crf
from medresearch_platform.approaches.rule_based import predict_rule_based
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import DocumentPrediction, Evaluator
from medresearch_platform.eval_harness.runner import _load_jsonl
from medresearch_platform.hybrid.pipeline import HybridPipeline


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eval_by_corpus(
    evaluator: Evaluator,
    test_docs: list[AnnotatedDocument],
    preds: list[DocumentPrediction],
) -> dict[str, Any]:
    pred_by_id = {p.doc_id: p for p in preds}
    overall = evaluator.evaluate(test_docs, preds)
    corpora = sorted({d.source for d in test_docs})
    by_corpus: dict[str, Any] = {}
    for src in corpora:
        sub_docs = [d for d in test_docs if d.source == src]
        sub_preds = [pred_by_id[d.doc_id] for d in sub_docs]
        m = evaluator.evaluate(sub_docs, sub_preds)
        by_corpus[src] = {
            "num_docs": len(sub_docs),
            "strict_span": m["strict_span"],
            "lenient_span": m["lenient_span"],
            "timex": m["timex"],
            "per_language": m["per_language"],
        }
    return {
        "overall": {
            "num_docs": len(test_docs),
            "strict_span": overall["strict_span"],
            "lenient_span": overall["lenient_span"],
            "per_language": overall["per_language"],
            "per_entity_type": overall["per_entity_type"],
            "timex": overall["timex"],
        },
        "by_corpus": by_corpus,
    }


def run_benchmark(
    *,
    data_dir: Path = Path("data/processed/open_clinical"),
    out_path: Path = Path("docs/results/open_clinical_benchmark.json"),
    seed: int = 1,
    epochs: int = 6,
    aux_epochs: int = 4,
) -> dict[str, Any]:
    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing open clinical splits in {data_dir}. Run: python3 scripts/download_open_clinical_corpora.py"
        )

    train_docs = _load_jsonl(train_path)
    test_docs = _load_jsonl(test_path)
    evaluator = Evaluator()
    results: dict[str, Any] = {}

    # 1. Rule-based
    t0 = time.perf_counter()
    preds_rb = predict_rule_based(train_docs, test_docs, seed, {})
    results["rule_based"] = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        **_eval_by_corpus(evaluator, test_docs, preds_rb),
    }
    print(f"[rule_based]      strict_f1={results['rule_based']['overall']['strict_span']['f1']:.4f} ({results['rule_based']['elapsed_seconds']}s)", flush=True)

    # 2. TF-IDF + Classical ML
    t0 = time.perf_counter()
    preds_ml = predict_classical_ml(train_docs, test_docs, seed, {})
    results["classical_ml"] = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        **_eval_by_corpus(evaluator, test_docs, preds_ml),
    }
    print(f"[classical_ml]    strict_f1={results['classical_ml']['overall']['strict_span']['f1']:.4f} ({results['classical_ml']['elapsed_seconds']}s)", flush=True)

    # 3. Classical CRF
    t0 = time.perf_counter()
    crf = ClassicalCRFModel(seed=seed).fit(train_docs)
    preds_crf = [crf.predict_document(d) for d in test_docs]
    results["classical_crf"] = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        **_eval_by_corpus(evaluator, test_docs, preds_crf),
    }
    print(f"[classical_crf]   strict_f1={results['classical_crf']['overall']['strict_span']['f1']:.4f} ({results['classical_crf']['elapsed_seconds']}s)", flush=True)

    # 4. From-scratch BiLSTM-CRF
    t0 = time.perf_counter()
    preds_bilstm_crf = predict_bilstm_crf(
        train_docs, test_docs, seed, {"hidden_dim": 32, "epochs": epochs, "lr": 0.018}
    )
    results["bilstm_crf"] = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        **_eval_by_corpus(evaluator, test_docs, preds_bilstm_crf),
    }
    print(f"[bilstm_crf]      strict_f1={results['bilstm_crf']['overall']['strict_span']['f1']:.4f} ({results['bilstm_crf']['elapsed_seconds']}s)", flush=True)

    # 5. Hybrid v2 (CRF ensemble + ConText + CNN Rel + CRF TIMEX, no Stage 5 reranker)
    t0 = time.perf_counter()
    cfg_v2: dict[str, Any] = {
        "hidden_dim": 32,
        "epochs": epochs,
        "aux_epochs": aux_epochs,
        "lr": 0.018,
        "use_crf_ensemble": True,
        "use_rules_layer": False,
        "use_assertion_bilstm": False,
        "use_relation_cnn": True,
        "use_temporal_crf": True,
        "use_transformer_fusion": False,
    }
    pipe_v2 = HybridPipeline(seed=seed, cfg=cfg_v2).fit(train_docs)
    preds_v2 = [pipe_v2.extract_document(d, use_rules_layer=False, use_transformer_fusion=False)[0] for d in test_docs]
    results["hybrid_v2"] = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        **_eval_by_corpus(evaluator, test_docs, preds_v2),
    }
    print(f"[hybrid_v2]       strict_f1={results['hybrid_v2']['overall']['strict_span']['f1']:.4f} ({results['hybrid_v2']['elapsed_seconds']}s)", flush=True)

    # 6. Hybrid v2 + OOF Stage 5 Transformer Reranker
    t0 = time.perf_counter()
    cfg_oof: dict[str, Any] = {
        **cfg_v2,
        "use_transformer_fusion": True,
        "stage5_oof_folds": 2,
        "stage5_oof_max_docs": 80,
    }
    pipe_oof = HybridPipeline(seed=seed, cfg=cfg_oof).fit(train_docs)
    preds_oof = [pipe_oof.extract_document(d, use_rules_layer=False, use_transformer_fusion=True)[0] for d in test_docs]
    results["hybrid_v2_oof_stage5"] = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "calibrated_rerank_threshold": pipe_oof.reranker.threshold,
        "calibrated_temperature": pipe_oof.scaler.temperature,
        **_eval_by_corpus(evaluator, test_docs, preds_oof),
    }
    print(
        f"[hybrid_v2_oof]   strict_f1={results['hybrid_v2_oof_stage5']['overall']['strict_span']['f1']:.4f} "
        f"(thresh={pipe_oof.reranker.threshold}, {results['hybrid_v2_oof_stage5']['elapsed_seconds']}s)",
        flush=True,
    )

    payload: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": str(data_dir),
        "train_docs": len(train_docs),
        "test_docs": len(test_docs),
        "train_sha256": _sha256_file(train_path),
        "test_sha256": _sha256_file(test_path),
        "seed": seed,
        "epochs": epochs,
        "aux_epochs": aux_epochs,
        "pretrained_weights_used": False,
        "approaches": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark from-scratch approaches on open-license EN/FR clinical corpora.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/open_clinical"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/open_clinical_benchmark.json"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--aux-epochs", type=int, default=4)
    args = parser.parse_args()
    run_benchmark(
        data_dir=args.data_dir,
        out_path=args.out,
        seed=args.seed,
        epochs=args.epochs,
        aux_epochs=args.aux_epochs,
    )


if __name__ == "__main__":
    main()
