"""Verify, package, and safely unpack SHA-256-verified NLP & Vision model release artifacts.

Addresses reproducibility (Audit F-01) by:
  1. `verify`: Auditing every served NLP & Vision checkpoint against its manifest.json and
     writing a version-controlled `docs/results/ARTIFACTS_MANIFEST.json`.
  2. `pack`: Creating reproducible `.tar.gz` release bundles + `checksums.sha256`.
  3. `unpack`: Verifying SHA-256 + tar path-traversal safety before extracting into `models/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NLP_SERVED_DIRS: tuple[str, ...] = (
    "hybrid_v2_seed1",
    "classical_crf_seed1",
    *(f"hospital_char_cnn_bilstm_crf_seed{i}" for i in range(1, 19)),
)

VISION_SERVED_DIRS: tuple[str, ...] = (
    "vision_router_v1",
    "derm_cnn_bsdrop40_seed1",
    "derm_cnn_seed2",
    "derm_cnn_seed3",
    "derm_cnn_seed4",
    "derm_cnn_seed5",
    "derm_cnn_clinical_seed6",
    "derm_cnn_clinical_seed7",
    "derm_convnext_seed10",
    "derm_convnext_seed11",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_single_checkpoint(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        return {"dir": model_dir.name, "status": "MISSING_MANIFEST", "valid": False}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Case 1: Multi-file manifest (hybrid_v2, classical_crf)
    if isinstance(manifest.get("files"), dict):
        mismatches: list[str] = []
        total_bytes = manifest_path.stat().st_size
        for rel, expected_sha in manifest["files"].items():
            target = model_dir / rel
            if not target.exists():
                mismatches.append(f"missing:{rel}")
                continue
            total_bytes += target.stat().st_size
            if _sha256_file(target) != expected_sha:
                mismatches.append(f"sha_mismatch:{rel}")
        return {
            "dir": model_dir.name,
            "model_type": str(manifest.get("model_type", "unknown")),
            "pretrained_weights_used": bool(manifest.get("pretrained_weights_used", False)),
            "manifest_sha256": _sha256_file(manifest_path),
            "size_bytes": total_bytes,
            "valid": len(mismatches) == 0,
            "status": "OK" if not mismatches else ";".join(mismatches),
        }

    # Case 2: Single-weight PyTorch checkpoint (NLP or Vision)
    weight_candidates = ("weights_torch.pt", "char_cnn_bilstm_crf.pt", "router_torch.pt", "model.pt")
    weight_file = next((model_dir / w for w in weight_candidates if (model_dir / w).exists()), None)
    expected_sha = str(manifest.get("weights_file_sha256") or manifest.get("trained_weights_sha256") or "")
    if weight_file is None:
        return {"dir": model_dir.name, "status": "MISSING_WEIGHTS", "valid": False}
    actual_sha = _sha256_file(weight_file)
    valid = bool(expected_sha and actual_sha == expected_sha)
    return {
        "dir": model_dir.name,
        "model_type": str(manifest.get("architecture_variant") or manifest.get("encoder_type") or "pytorch"),
        "pretrained_weights_used": bool(manifest.get("pretrained_weights_used", False)),
        "weight_file": weight_file.name,
        "trained_weights_sha256": actual_sha,
        "manifest_sha256": _sha256_file(manifest_path),
        "size_bytes": weight_file.stat().st_size + manifest_path.stat().st_size,
        "valid": valid,
        "status": "OK" if valid else "SHA_MISMATCH",
    }


def verify_served_artifacts(
    models_root: Path = Path("models"),
    out_manifest: Path = Path("docs/results/ARTIFACTS_MANIFEST.json"),
) -> dict[str, Any]:
    nlp_entries = [_verify_single_checkpoint(models_root / name) for name in NLP_SERVED_DIRS if (models_root / name).exists()]
    vision_entries = [_verify_single_checkpoint(models_root / name) for name in VISION_SERVED_DIRS if (models_root / name).exists()]
    all_entries = nlp_entries + vision_entries
    invalid = [e for e in all_entries if not e["valid"]]
    pretrained = [e for e in all_entries if e.get("pretrained_weights_used")]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "nlp_checkpoints_found": len(nlp_entries),
            "vision_checkpoints_found": len(vision_entries),
            "total_verified_ok": sum(1 for e in all_entries if e["valid"]),
            "total_invalid": len(invalid),
            "total_pretrained_used": len(pretrained),
            "total_size_mb": round(sum(int(e.get("size_bytes", 0)) for e in all_entries) / (1024 * 1024), 2),
        },
        "nlp_artifacts": nlp_entries,
        "vision_artifacts": vision_entries,
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _pack_group(models_root: Path, names: tuple[str, ...], out_tar: Path) -> str:
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tf:
        for name in names:
            d = models_root / name
            if not d.exists():
                continue
            tf.add(d, arcname=name)
    return _sha256_file(out_tar)


def safe_unpack_archive(archive: Path, *, expected_sha256: str, dest: Path) -> list[str]:
    actual = _sha256_file(archive)
    if actual.lower() != expected_sha256.strip().lower():
        raise ValueError(f"Archive SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    dest_resolved = dest.resolve()
    dest_resolved.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            member_path = (dest_resolved / member.name).resolve()
            if not str(member_path).startswith(str(dest_resolved)):
                raise ValueError(f"Blocked unsafe tar path traversal member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Blocked symlink/hardlink in release archive: {member.name}")
        tf.extractall(path=dest_resolved, filter="data")
        extracted = [m.name for m in tf.getmembers()]
    return extracted


def main() -> None:
    p = argparse.ArgumentParser(description="Verify, pack, or unpack SHA-256-verified model artifacts.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("verify", help="Verify served checkpoints and write docs/results/ARTIFACTS_MANIFEST.json")
    pv.add_argument("--models-dir", type=Path, default=Path("models"))
    pv.add_argument("--out", type=Path, default=Path("docs/results/ARTIFACTS_MANIFEST.json"))

    pp = sub.add_parser("pack", help="Package verified NLP & Vision release tarballs + checksums.sha256")
    pp.add_argument("--models-dir", type=Path, default=Path("models"))
    pp.add_argument("--out-dir", type=Path, default=Path("dist/release"))

    pu = sub.add_parser("unpack", help="Verify SHA-256 and safely unpack a release tarball")
    pu.add_argument("--archive", type=Path, required=True)
    pu.add_argument("--sha256", type=str, required=True)
    pu.add_argument("--dest", type=Path, default=Path("models"))

    args = p.parse_args()
    if args.cmd == "verify":
        rep = verify_served_artifacts(args.models_dir, args.out)
        print(json.dumps(rep["summary"], indent=2))
        if rep["summary"]["total_invalid"] > 0:
            raise SystemExit(1)
    elif args.cmd == "pack":
        rep = verify_served_artifacts(args.models_dir)
        if rep["summary"]["total_invalid"] > 0:
            raise SystemExit("Aborting pack: invalid checkpoint(s) found.")
        nlp_tar = args.out_dir / "models_nlp_18seed.tar.gz"
        vis_tar = args.out_dir / "models_vision_9seed.tar.gz"
        nlp_sha = _pack_group(args.models_dir, NLP_SERVED_DIRS, nlp_tar)
        vis_sha = _pack_group(args.models_dir, VISION_SERVED_DIRS, vis_tar)
        checksums = f"{nlp_sha}  {nlp_tar.name}\n{vis_sha}  {vis_tar.name}\n"
        (args.out_dir / "checksums.sha256").write_text(checksums, encoding="utf-8")
        print(checksums, end="")
    elif args.cmd == "unpack":
        files = safe_unpack_archive(args.archive, expected_sha256=args.sha256, dest=args.dest)
        print(f"Verified and extracted {len(files)} entries into {args.dest}")


if __name__ == "__main__":
    main()
