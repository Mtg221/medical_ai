"""CLI entrypoint for `medextract`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from medresearch_platform.data.builder import build_corpus_from_config


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

    args = parser.parse_args(argv)
    if args.group == "data" and args.action == "build":
        manifest = build_corpus_from_config(Path(args.config))
        _print_stats_table(manifest)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
