"""Unified `medextract` CLI for dataset generation, experiment execution, training, extraction, reports, and user management."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.config import PlatformConfig
from medresearch_platform.data.builder import build_corpus_from_config
from medresearch_platform.documents.parsers import detect_document_language, parse_document_bytes
from medresearch_platform.eval_harness.runner import _load_jsonl, run_experiment
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.hybrid.stage4_temporal import resolve_document_date
from medresearch_platform.reports.generator import generate_report_bytes
from medresearch_platform.serving.extractor import ExtractionService


def _resolve_input_path(args: argparse.Namespace) -> Path:
    p = getattr(args, "input_opt", None) or getattr(args, "input_pos", None)
    if p is None:
        raise ValueError("An input file path is required (positional or --input).")
    return Path(p)


def _extract_file_with_saved_model(
    input_path: Path,
    *,
    model_dir: Path,
    approach: str = "hybrid_v2",
    document_date: str | None = None,
) -> tuple[dict[str, Any], str, str, str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    raw_bytes = input_path.read_bytes()
    ext = input_path.suffix.lower().lstrip(".")
    fmt_map = {
        "txt": "TXT",
        "pdf": "PDF",
        "docx": "DOCX",
        "png": "PNG",
        "jpg": "JPEG",
        "jpeg": "JPEG",
        "tif": "TIFF",
        "tiff": "TIFF",
        "webp": "WEBP",
    }
    fmt = fmt_map.get(ext, "TXT")
    if fmt == "TXT":
        text = raw_bytes.decode("utf-8", errors="replace")
        lang = detect_document_language(text)
        page_map: list[dict[str, Any]] | None = None
    else:
        parsed = parse_document_bytes(raw_bytes, fmt)
        text = "\n".join(p.raw_text for p in parsed.pages)
        lang = parsed.detected_language
        page_map = []
        cursor = 0
        for p in parsed.pages:
            if page_map:
                cursor += 1
            start_off = cursor
            cursor += len(p.raw_text)
            page_map.append({
                "page_number": p.page_number,
                "char_start": start_off,
                "char_end": cursor,
                "word_boxes": getattr(p, "word_boxes", []),
            })

    resolved_date, date_source = resolve_document_date(text, fallback_date=document_date)
    svc = ExtractionService(
        model_dir=model_dir if approach == "hybrid_v2" else None,
        classical_crf_dir=model_dir if approach == "classical_crf" else None,
    )
    res = svc.extract(
        text,
        language=lang if lang in ("en", "fr") else "en",
        document_date=resolved_date,
        page_map=page_map,
        document_id=input_path.stem,
        source_record_id=input_path.stem,
        approach=approach,
    )
    payload = res.to_dict()
    payload["schema_version"] = "1.0"
    payload["document_date"] = resolved_date
    payload["document_date_source"] = date_source
    payload["detected_language"] = lang
    payload["entities"] = [
        {
            "id": f"e{i+1}",
            "type": r.get("entity_type"),
            "text": r.get("original_text"),
            "assertion": r.get("assertion_status"),
            "confidence": r.get("confidence_score"),
            "char_start": r.get("char_start"),
            "char_end": r.get("char_end"),
            "source": {"page": r.get("page_number", 1), "start": r.get("char_start"), "end": r.get("char_end")},
        }
        for i, r in enumerate(payload.get("records", []))
    ]
    return payload, text, lang, resolved_date


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medextract", description="Multilingual Medical Extraction CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # medextract data build
    p_data = sub.add_parser("data", help="Dataset build and validation commands")
    sub_data = p_data.add_subparsers(dest="data_cmd", required=True)
    p_build = sub_data.add_parser("build", help="Build synthetic corpus from YAML config")
    p_build.add_argument("--config", type=Path, required=True, help="Path to configs/data/v2.yaml or v3.yaml")

    # medextract experiment
    p_exp = sub.add_parser("experiment", help="Run a reproducible multi-seed experiment")
    p_exp.add_argument("--config", type=Path, required=True, help="Path to configs/exp/*.yaml")
    p_exp.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Random seeds")
    p_exp.add_argument("--split", choices=["val", "test"], default="val", help="Evaluation split")
    p_exp.add_argument("--unlock-test", action="store_true", help="Required flag when --split test is used")
    p_exp.add_argument("--reason", type=str, default="CLI evaluation", help="Reason logged when unlocking test")

    # medextract train
    p_train = sub.add_parser("train", help="Train and persist a production model artifact directory")
    p_train.add_argument("--config", type=Path, required=True, help="Path to configs/exp/*.yaml")
    p_train.add_argument("--seed", type=int, default=1, help="Random seed")
    p_train.add_argument("--out", type=Path, required=True, help="Output directory (e.g. models/hybrid_v2_seed1)")

    # medextract extract
    p_ext = sub.add_parser("extract", help="Extract clinical entities, assertions, relations, and TIMEX using saved model")
    p_ext.add_argument("input_pos", nargs="?", type=Path, default=None, help="Input file (.txt, .pdf, .docx, .png)")
    p_ext.add_argument("--input", "-i", dest="input_opt", type=Path, default=None, help="Input file (.txt, .pdf, .docx, .png)")
    p_ext.add_argument("--model", type=Path, default=Path(os.environ.get("MEDRESEARCH_MODEL_DIR", "models/hybrid_v2_seed1")), help="Saved model directory")
    p_ext.add_argument("--approach", choices=["hybrid_v2", "classical_crf"], default="hybrid_v2", help="Serving approach")
    p_ext.add_argument("--pipeline", type=str, default=None, help="Deprecated alias for --approach")
    p_ext.add_argument("--train-jsonl", type=Path, default=None, help="Deprecated ignored argument")
    p_ext.add_argument("--document-date", type=str, default=None, help="Optional fallback document date (YYYY-MM-DD)")
    p_ext.add_argument("--output", "--out", "-o", dest="output", type=Path, default=None, help="Output JSON path (stdout if omitted)")

    # medextract report
    p_rep = sub.add_parser("report", help="Generate a bilingual structured clinical report using saved model")
    p_rep.add_argument("input_pos", nargs="?", type=Path, default=None, help="Input file (.txt, .pdf, .docx, .png)")
    p_rep.add_argument("--input", "-i", dest="input_opt", type=Path, default=None, help="Input file (.txt, .pdf, .docx, .png)")
    p_rep.add_argument("--model", type=Path, default=Path(os.environ.get("MEDRESEARCH_MODEL_DIR", "models/hybrid_v2_seed1")), help="Saved model directory")
    p_rep.add_argument("--approach", choices=["hybrid_v2", "classical_crf"], default="hybrid_v2", help="Serving approach")
    p_rep.add_argument("--pipeline", type=str, default=None, help="Deprecated alias for --approach")
    p_rep.add_argument("--train-jsonl", type=Path, default=None, help="Deprecated ignored argument")
    p_rep.add_argument("--format", choices=["markdown", "html", "pdf"], default="markdown")
    p_rep.add_argument("--lang", choices=["en", "fr"], default="en")
    p_rep.add_argument("--document-date", type=str, default=None, help="Optional fallback document date (YYYY-MM-DD)")
    p_rep.add_argument("--output", "--out", "-o", dest="output", type=Path, required=True, help="Output report file")

    # medextract users
    p_users = sub.add_parser("users", help="Initialize or reset local platform users")
    sub_users = p_users.add_subparsers(dest="users_cmd", required=True)
    p_u_init = sub_users.add_parser("init", help="Initialize default demo users and set their password")
    p_u_init.add_argument("--password", type=str, default=None, help="Password (defaults to MEDRESEARCH_DEMO_PASSWORD)")
    p_u_set = sub_users.add_parser("set-password", help="Set password for a specific username")
    p_u_set.add_argument("username", type=str, help="Username (e.g. admin_dakar)")
    p_u_set.add_argument("--password", type=str, required=True, help="New password (min 8 chars)")

    args = parser.parse_args(argv)

    if args.cmd == "data" and args.data_cmd == "build":
        summary = build_corpus_from_config(args.config)
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd == "experiment":
        res = run_experiment(args.config, seeds=args.seeds)
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "train":
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        data_dir = Path(cfg.get("data_dir", "data/processed/v3_1"))
        train_docs = _load_jsonl(data_dir / "train.jsonl")
        approach = str(cfg.get("approach", "hybrid"))
        corpus_id = str(cfg.get("corpus_id", data_dir.name))
        if approach == "classical_crf":
            crf = ClassicalCRFModel(seed=args.seed, cfg=cfg).fit(train_docs)
            crf.save(args.out, corpus_id=corpus_id)
            print(json.dumps(crf.manifest, indent=2))
        else:
            pipe = HybridPipeline(seed=args.seed, cfg=cfg).fit(train_docs)
            pipe.save(args.out, corpus_id=corpus_id)
            print(json.dumps(pipe.manifest, indent=2))
        return 0

    if args.cmd == "extract":
        input_path = _resolve_input_path(args)
        approach = "classical_crf" if getattr(args, "pipeline", None) == "classical_crf" else args.approach
        payload, _, _, _ = _extract_file_with_saved_model(
            input_path,
            model_dir=args.model,
            approach=approach,
            document_date=args.document_date,
        )
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0

    if args.cmd == "report":
        input_path = _resolve_input_path(args)
        approach = "classical_crf" if getattr(args, "pipeline", None) == "classical_crf" else args.approach
        payload, _, lang, resolved_date = _extract_file_with_saved_model(
            input_path,
            model_dir=args.model,
            approach=approach,
            document_date=args.document_date,
        )
        assert_map = {
            "present": "AFFIRMED",
            "absent": "NEGATED",
            "uncertain": "UNCERTAIN",
            "hypothetical": "HYPOTHETICAL",
            "family_history": "FAMILY_HISTORY",
        }
        entities_dicts: list[dict[str, Any]] = []
        relations_dicts: list[dict[str, Any]] = []
        span_to_id: dict[tuple[int, int], str] = {}
        for i, rec in enumerate(payload.get("records", [])):
            eid = f"e{i+1}"
            cs = int(rec.get("char_start") or 0)
            ce = int(rec.get("char_end") or 0)
            span_to_id[(cs, ce)] = eid
            prov = rec.get("provenance") or {}
            entities_dicts.append({
                "entity_id": eid,
                "entity_type": rec.get("entity_type"),
                "source_text": rec.get("original_text"),
                "normalized_text": rec.get("normalized_concept") or rec.get("extracted_value"),
                "assertion_status": assert_map.get(str(rec.get("assertion_status")), "AFFIRMED"),
                "confidence": rec.get("confidence_score", 0.9),
                "source_stage": prov.get("source_stage", payload.get("model_name", "hybrid_v2")),
                "review_status": rec.get("review_status", "unreviewed"),
                "review_reason": prov.get("review_reason", ""),
                "char_start": cs,
                "char_end": ce,
            })
        r_idx = 0
        for rec in payload.get("records", []):
            prov = rec.get("provenance") or {}
            for r in prov.get("relations", []):
                r_idx += 1
                relations_dicts.append({
                    "relation_id": f"r{r_idx}",
                    "relation_type": r.get("type"),
                    "source_entity_id": span_to_id.get((int(r.get("head_start", 0)), int(r.get("head_end", 0)))),
                    "target_entity_id": span_to_id.get((int(r.get("tail_start", 0)), int(r.get("tail_end", 0)))),
                    "confidence": r.get("confidence", 0.85),
                })

        doc_meta = {
            "document_id": input_path.stem,
            "original_filename": input_path.name,
            "detected_language": lang,
            "document_type": "CLINICAL_NOTE",
            "created_at": resolved_date,
        }
        report_bytes, _ = generate_report_bytes(
            doc_meta,
            entities_dicts,
            relations_dicts,
            lang=args.lang,
            fmt=args.format,  # type: ignore[arg-type]
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(report_bytes)
        return 0

    if args.cmd == "users":
        from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService

        cfg = PlatformConfig.from_env()
        svc = MedicalResearchPlatformService(cfg)
        if args.users_cmd == "init":
            pw = args.password or os.environ.get("MEDRESEARCH_DEMO_PASSWORD") or secrets.token_urlsafe(16)
            svc.init_or_reset_default_users(pw)
            print(json.dumps({"status": "ok", "db": str(cfg.sqlite_path), "users_initialized": 5}, indent=2))
            return 0
        if args.users_cmd == "set-password":
            svc.set_user_password(args.username, args.password)
            print(json.dumps({"status": "ok", "username": args.username, "db": str(cfg.sqlite_path)}, indent=2))
            return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
