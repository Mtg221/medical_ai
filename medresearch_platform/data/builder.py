"""Reproducible corpus builder and statistics generator."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.data.splits import assign_stratified_splits, verify_no_split_leakage
from medresearch_platform.data.synthetic_v2 import (
    generate_synthetic_corpus,
    render_document_artifacts,
)
from medresearch_platform.data.synthetic_v3 import generate_v3_flat_list


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_corpus_stats(docs: list[AnnotatedDocument]) -> dict[str, Any]:
    """Compute docs, tokens, and entities per type, per language, and per split."""
    by_split: dict[str, int] = Counter()
    by_lang: dict[str, int] = Counter()
    by_doc_type: dict[str, int] = Counter()
    tokens_by_lang_split: dict[str, int] = Counter()
    entities_by_type_lang: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    assertions_by_class: dict[str, int] = Counter()
    total_relations = 0
    total_timex = 0

    for d in docs:
        sp = d.split or "unassigned"
        by_split[sp] += 1
        by_lang[d.language] += 1
        by_doc_type[d.doc_type] += 1
        tokens_by_lang_split[f"{d.language}:{sp}"] += len(d.text.split())
        total_relations += len(d.relations)
        total_timex += len(d.timex)
        for e in d.entities:
            entities_by_type_lang[e.type][d.language] += 1
            assertions_by_class[e.assertion] += 1

    return {
        "total_documents": len(docs),
        "by_split": dict(by_split),
        "by_language": dict(by_lang),
        "by_doc_type": dict(by_doc_type),
        "tokens_by_language_split": dict(tokens_by_lang_split),
        "entities_by_type_language": {k: dict(v) for k, v in sorted(entities_by_type_lang.items())},
        "assertions_by_class": dict(assertions_by_class),
        "total_relations": total_relations,
        "total_timex": total_timex,
    }


def build_corpus_from_config(config_path: Path) -> dict[str, Any]:
    """Build dataset splits, run leakage verification, and emit manifest + stats."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    langs = cfg.get("languages", {"en": 400, "fr": 400})
    splits_cfg = cfg.get("splits", {"train": 0.70, "val": 0.15, "test": 0.15})
    leak_cfg = cfg.get("leakage_check", {})
    render_cfg = cfg.get("render_samples", {})

    out_dir = Path(cfg.get("output_dir", "data/processed/v1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    generator_ver = str(cfg.get("generator", "v2"))
    if generator_ver in {"v3", "v3_1"}:
        v3_counts = {str(k): int(v) for k, v in cfg.get("v3_splits_per_language", {}).items()}
        docs = generate_v3_flat_list(
            split_counts_per_lang=v3_counts or None,
            seed=seed,
            version=generator_ver,
        )
        split_names = sorted({d.split for d in docs if d.split})
    else:
        docs = generate_synthetic_corpus(
            en_count=int(langs.get("en", 400)),
            fr_count=int(langs.get("fr", 400)),
            seed=seed,
            typo_rate=float(render_cfg.get("typo_rate", 0.05)),
        )
        assign_stratified_splits(
            docs,
            train_ratio=float(splits_cfg.get("train", 0.70)),
            val_ratio=float(splits_cfg.get("val", 0.15)),
            test_ratio=float(splits_cfg.get("test", 0.15)),
            seed=seed,
        )
        split_names = ["train", "val", "test"]

    leakage = verify_no_split_leakage(
        docs,
        minhash_threshold=float(leak_cfg.get("minhash_threshold", 0.85)),
        num_perm=int(leak_cfg.get("num_perm", 64)),
        shingle_size=int(leak_cfg.get("shingle_size", 5)),
    )
    if not leakage["passed"]:
        raise RuntimeError(f"Split leakage detected: {leakage}")

    split_files: dict[str, str] = {}
    split_hashes: dict[str, str] = {}
    for sp in split_names:
        sp_path = out_dir / f"{sp}.jsonl"
        sp_docs = [d for d in docs if d.split == sp]
        with sp_path.open("w", encoding="utf-8", newline="\n") as f:
            for d in sp_docs:
                f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
        digest = _sha256_file(sp_path)
        if generator_ver == "v3_1" and sp in {"val", "test_iid", "test_unseen_terms", "test_unseen_templates"}:
            locked_file = Path("data/locked/synth_med_v3_1") / f"{sp}.jsonl"
            if locked_file.exists():
                import hashlib

                locked_sha = hashlib.sha256(locked_file.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                if digest != locked_sha:
                    raise RuntimeError(
                        f"Locked SYNTH-MED-V3.1 split '{sp}' hash mismatch: generated {digest} != locked {locked_sha}"
                    )
        split_files[sp] = str(sp_path)
        split_hashes[sp] = digest

    rendered_samples: list[dict[str, str]] = []
    if render_cfg.get("enabled", False):
        n_per_lang = int(render_cfg.get("count_per_language", 6))
        samples_dir = out_dir / "rendered_samples"
        for lang in ("en", "fr"):
            lang_docs = [d for d in docs if d.language == lang][:n_per_lang]
            for idx, d in enumerate(lang_docs):
                rendered_samples.append(render_document_artifacts(d, samples_dir / lang, seed=seed + idx))

    stats = compute_corpus_stats(docs)
    manifest = {
        "dataset_name": cfg.get("dataset_name", "SYNTH-MED-V2"),
        "version": cfg.get("version", "1.0.0"),
        "seed": seed,
        "synthetic_only": True,
        "test_split_locked": True,
        "split_files": split_files,
        "sha256": split_hashes,
        "leakage_check": leakage,
        "rendered_sample_count": len(rendered_samples),
        "stats": stats,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
