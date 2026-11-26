"""Command-Line Interface for the Medical Information Intelligence & Federated Research Platform."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from medresearch_platform.config import PlatformConfig
from medresearch_platform.diagnostics import generate_readiness_report
from medresearch_platform.security.audit import verify_audit_chain
from medresearch_platform.workflows.pipeline import MedicalResearchPlatformService


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="medresearch-cli",
        description="Bilingual (EN/FR) Medical Information Intelligence & Federated Research Platform CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("diagnostics", help="Run hardware, OS, and dependency readiness diagnostics")
    sub.add_parser("init-db", help="Initialize database schema and bootstrap institutions/users/terminology")
    sub.add_parser("seed", help="Seed synthetic bilingual clinical corpus (SYNTH-MED-EN-FR v1)")
    sub.add_parser("run-experiment", help="Run full CPU-only Phase 1 OCR, NLP, Fairness, and Federated benchmark")
    sub.add_parser("verify-audit", help="Verify cryptographic hash chain of audit logs")

    args = parser.parse_args(argv)
    config = PlatformConfig.from_env()
    service = MedicalResearchPlatformService(config=config)

    if args.command == "diagnostics":
        report = generate_readiness_report(config.workspace_dir)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "init-db":
        res = service.initialize_platform()
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "seed":
        res = service.seed_synthetic_corpus()
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "run-experiment":
        manifest = service.run_phase1_experiments()
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if args.command == "verify-audit":
        service.initialize_platform()
        with service.db.connect() as conn:
            res = verify_audit_chain(conn)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0 if res.get("valid") else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
