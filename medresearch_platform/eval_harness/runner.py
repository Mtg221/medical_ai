"""Multi-seed YAML experiment runner emitting metrics, errors, resources, and model cards."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.approaches.classical_ml import (
    predict_classical_crf,
    predict_classical_ml,
)
from medresearch_platform.approaches.neural_seq import (
    predict_bilstm,
    predict_bilstm_crf,
    predict_transformer,
)
from medresearch_platform.approaches.rule_based import predict_rule_based
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.error_taxonomy import classify_corpus_errors
from medresearch_platform.eval_harness.metrics import (
    DocumentPrediction,
    Evaluator,
    PredictedEntity,
    bootstrap_f1_ci,
)
from medresearch_platform.eval_harness.profiler import ResourceProfiler
from medresearch_platform.hybrid.pipeline import predict_hybrid

PredictorFn = Callable[[list[AnnotatedDocument], list[AnnotatedDocument], int, dict[str, Any]], list[DocumentPrediction]]


def _load_jsonl(path: Path) -> list[AnnotatedDocument]:
    docs: list[AnnotatedDocument] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(AnnotatedDocument.from_dict(json.loads(line)))
    return docs


def _dummy_predictor(
    train_docs: list[AnnotatedDocument],
    eval_docs: list[AnnotatedDocument],
    seed: int,
    cfg: dict[str, Any],
) -> list[DocumentPrediction]:
    """Minimal deterministic baseline used to smoke-test the evaluation harness."""
    # Collect surface forms from train_docs only (no test leakage)
    vocab: dict[str, str] = {}
    for d in train_docs:
        for e in d.entities:
            if e.type in ("MEDICATION", "DIAGNOSIS", "SYMPTOM"):
                vocab.setdefault(e.text.lower(), e.type)

    preds: list[DocumentPrediction] = []
    for d in eval_docs:
        ents: list[PredictedEntity] = []
        lower_text = d.text.lower()
        for term, etype in sorted(vocab.items(), key=lambda kv: -len(kv[0])):
            for m in re.finditer(re.escape(term), lower_text):
                ents.append(
                    PredictedEntity(
                        type=etype,
                        start=m.start(),
                        end=m.end(),
                        text=d.text[m.start() : m.end()],
                        assertion="present",
                        confidence=0.65,
                    )
                )
        preds.append(
            DocumentPrediction(
                doc_id=d.doc_id,
                language=d.language,
                doc_type=d.doc_type,
                predicted_doc_type=d.doc_type,
                entities=ents,
            )
        )
    return preds


APPROACH_REGISTRY: dict[str, PredictorFn] = {
    "dummy": _dummy_predictor,
    "rule_based": predict_rule_based,
    "classical_ml": predict_classical_ml,
    "classical_crf": predict_classical_crf,
    "bilstm": predict_bilstm,
    "bilstm_crf": predict_bilstm_crf,
    "transformer": predict_transformer,
    "hybrid": predict_hybrid,
}


def _mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {"mean": round(mean, 6), "std": round(math.sqrt(var), 6)}


def run_experiment(
    config_path: Path,
    *,
    seeds: list[int] | None = None,
    predictor_fn: PredictorFn | None = None,
) -> dict[str, Any]:
    """Run an experiment config across multiple seeds and write standardized artifacts."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    approach = str(cfg.get("approach", "dummy"))
    eval_split = str(cfg.get("eval_split", "val"))
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")

    data_dir = Path(cfg.get("data_dir", "data/processed/v1"))
    train_docs = _load_jsonl(data_dir / "train.jsonl")
    eval_docs = _load_jsonl(data_dir / f"{eval_split}.jsonl")

    run_seeds = seeds or [int(s) for s in cfg.get("seeds", [1, 2, 3])]
    fn = predictor_fn or APPROACH_REGISTRY.get(approach)
    if fn is None:
        raise ValueError(f"Unknown approach {approach!r}. Registered: {sorted(APPROACH_REGISTRY.keys())}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{approach}_{eval_split}_{ts}"
    out_root = Path(cfg.get("output_root", "experiments/runs"))
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator()
    seed_results: list[dict[str, Any]] = []
    last_errors = []
    last_resources: dict[str, Any] = {}

    for seed in run_seeds:
        with ResourceProfiler(docs_count=len(eval_docs)) as prof:
            preds = fn(train_docs, eval_docs, seed, cfg)
        metrics = evaluator.evaluate(eval_docs, preds)
        ci = bootstrap_f1_ci(eval_docs, preds, n_resamples=200, seed=seed)
        metrics["strict_span_ci95"] = ci
        res_dict = prof.metrics.to_dict() if prof.metrics else {}
        seed_results.append({"seed": seed, "metrics": metrics, "resources": res_dict})
        last_errors = classify_corpus_errors(eval_docs, preds)
        last_resources = res_dict

    summary = {
        "strict_f1": _mean_std([r["metrics"]["strict_span"]["f1"] for r in seed_results]),
        "strict_precision": _mean_std([r["metrics"]["strict_span"]["precision"] for r in seed_results]),
        "strict_recall": _mean_std([r["metrics"]["strict_span"]["recall"] for r in seed_results]),
        "lenient_f1": _mean_std([r["metrics"]["lenient_span"]["f1"] for r in seed_results]),
        "assertion_macro_f1": _mean_std([r["metrics"]["assertion"]["macro_f1"] for r in seed_results]),
        "relations_f1": _mean_std([r["metrics"]["relations"]["f1"] for r in seed_results]),
        "wall_seconds": _mean_std([r["resources"].get("wall_seconds", 0.0) for r in seed_results]),
    }

    error_counts = dict(Counter(e.category for e in last_errors))
    payload = {
        "run_id": run_id,
        "approach": approach,
        "eval_split": eval_split,
        "seeds": run_seeds,
        "summary": summary,
        "error_counts": error_counts,
        "per_seed": seed_results,
    }

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "resources.json").write_text(json.dumps(last_resources, indent=2), encoding="utf-8")
    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as ef:
        for err in last_errors:
            ef.write(json.dumps(err.to_dict(), ensure_ascii=False) + "\n")

    model_card = (
        f"# Experiment Run: `{run_id}`\n\n"
        f"- **Approach:** `{approach}`\n"
        f"- **Eval Split:** `{eval_split}` ({len(eval_docs)} docs)\n"
        f"- **Seeds:** `{run_seeds}`\n"
        f"- **Strict F1 (mean ± std):** `{summary['strict_f1']['mean']:.4f} ± {summary['strict_f1']['std']:.4f}`\n"
        f"- **Lenient F1 (mean ± std):** `{summary['lenient_f1']['mean']:.4f} ± {summary['lenient_f1']['std']:.4f}`\n"
        f"- **Error Counts (last seed):** `{error_counts}`\n"
    )
    (run_dir / "model_card.md").write_text(model_card, encoding="utf-8")
    return payload
