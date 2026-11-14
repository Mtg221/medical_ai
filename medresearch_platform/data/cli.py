"""CLI entrypoint for `medextract`."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from medresearch_platform.data.builder import build_corpus_from_config
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.data.synthetic_v2 import generate_synthetic_corpus
from medresearch_platform.documents.parsers import parse_document_bytes
from medresearch_platform.eval_harness.runner import _load_jsonl, run_experiment
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.reports import generate_report_bytes


def _print_stats_table(manifest: dict[str, Any]) -> None:
    stats = manifest["stats"]
    print(f"Dataset: {manifest['dataset_name']} v{manifest['version']} (seed={manifest['seed']})")
    print(f"Total documents: {stats['total_documents']} | Splits: {stats['by_split']}")
    print(f"Languages: {stats['by_language']} | Tokens (lang:split): {stats['tokens_by_language_split']}")
    print(f"Relations: {stats['total_relations']} | TIMEX: {stats['total_timex']}")
    print(f"Leakage check passed: {manifest['leakage_check']['passed']}")
    print("\nEntities per type & language:")
    print(f"{'Entity Type':<18} {'EN':>8} {'FR':>8}")
    print("-" * 36)
    for etype, counts in stats["entities_by_type_language"].items():
        print(f"{etype:<18} {counts.get('en', 0):>8} {counts.get('fr', 0):>8}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medextract", description="Multilingual Medical Extraction CLI")
    sub = parser.add_subparsers(dest="group", required=True)

    data_parser = sub.add_parser("data", help="Dataset operations")
    data_sub = data_parser.add_subparsers(dest="action", required=True)

    build_parser = data_sub.add_parser("build", help="Build dataset splits and manifest")
    build_parser.add_argument("--config", required=True, help="Path to data YAML config")

    exp_parser = sub.add_parser("experiment", help="Experiment operations")
    exp_sub = exp_parser.add_subparsers(dest="action", required=True)
    run_parser = exp_sub.add_parser("run", help="Run a multi-seed experiment")
    run_parser.add_argument("config", help="Path to experiment YAML config")
    run_parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 1,2,3")

    ext_parser = sub.add_parser("extract", help="Extract structured medical JSON from a document")
    ext_parser.add_argument("file", help="Path to input document (TXT, PDF, DOCX, PNG)")
    ext_parser.add_argument("--pipeline", default="hybrid", help="Pipeline name (default: hybrid)")
    ext_parser.add_argument("--train-jsonl", default="data/processed/v1/train.jsonl", help="Training corpus JSONL")

    rep_parser = sub.add_parser("report", help="Generate a traceable EN/FR human-readable report")
    rep_parser.add_argument("file", help="Path to input document (TXT, PDF, DOCX, PNG)")
    rep_parser.add_argument("--lang", choices=["en", "fr"], default="en", help="Report language")
    rep_parser.add_argument("--format", choices=["markdown", "html", "pdf"], default="markdown", help="Output format")
    rep_parser.add_argument("-o", "--output", default=None, help="Output file path (optional)")
    rep_parser.add_argument("--train-jsonl", default="data/processed/v1/train.jsonl", help="Training corpus JSONL")

    args = parser.parse_args(argv)
    if args.group == "data" and args.action == "build":
        manifest = build_corpus_from_config(Path(args.config))
        _print_stats_table(manifest)
        return 0
    if args.group == "experiment" and args.action == "run":
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else None
        res = run_experiment(Path(args.config), seeds=seeds)
        sf1 = res["summary"]["strict_f1"]
        print(f"Run {res['run_id']} | Strict F1: {sf1['mean']:.4f} ± {sf1['std']:.4f}")
        print(f"Error counts: {res['error_counts']}")
        return 0
    if args.group == "extract":
        fpath = Path(args.file)
        fmt = fpath.suffix.lstrip(".").upper() or "TXT"
        parsed = parse_document_bytes(fpath.read_bytes(), file_format=fmt)
        text = "\n".join(p.raw_text for p in parsed.pages)
        train_path = Path(args.train_jsonl)
        train_docs = (
            _load_jsonl(train_path)
            if train_path.exists()
            else generate_synthetic_corpus(en_count=24, fr_count=24, seed=42)
        )
        doc = AnnotatedDocument(
            doc_id=fpath.stem,
            patient_id="PAT-CLI",
            language=parsed.detected_language if parsed.detected_language in ("en", "fr") else "en",
            doc_type="discharge_summary",
            document_date=date.today().isoformat(),
            source=fpath.name,
            text=text,
        )
        pipe = HybridPipeline(seed=42, cfg={"epochs": 6, "aux_epochs": 4, "max_train_docs": 60}).fit(train_docs)
        _, payload = pipe.extract_document(doc)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if args.group == "report":
        fpath = Path(args.file)
        fmt = fpath.suffix.lstrip(".").upper() or "TXT"
        parsed = parse_document_bytes(fpath.read_bytes(), file_format=fmt)
        text = "\n".join(p.raw_text for p in parsed.pages)
        train_path = Path(args.train_jsonl)
        train_docs = (
            _load_jsonl(train_path)
            if train_path.exists()
            else generate_synthetic_corpus(en_count=24, fr_count=24, seed=42)
        )
        doc = AnnotatedDocument(
            doc_id=fpath.stem,
            patient_id="PAT-CLI",
            language=parsed.detected_language if parsed.detected_language in ("en", "fr") else "en",
            doc_type="discharge_summary",
            document_date=date.today().isoformat(),
            source=fpath.name,
            text=text,
        )
        pipe = HybridPipeline(seed=42, cfg={"epochs": 6, "aux_epochs": 4, "max_train_docs": 60}).fit(train_docs)
        _, payload = pipe.extract_document(doc)
        doc_meta = {
            "document_id": fpath.stem,
            "original_filename": fpath.name,
            "detected_language": parsed.detected_language,
            "extraction_mode": parsed.extraction_mode,
        }
        report_bytes, _ = generate_report_bytes(
            doc_meta,
            payload.get("records", []),
            [],
            lang=args.lang,
            fmt=args.format,
        )
        if args.output:
            Path(args.output).write_bytes(report_bytes)
            print(f"Wrote {args.format} ({args.lang}) report to {args.output}")
        else:
            print(report_bytes.decode("utf-8", errors="replace"))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
