"""Shared model persistence, SHA-256 manifest verification, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn_crfsuite

from medresearch_platform.data.schema import AnnotatedDocument


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def compute_corpus_hash(docs: list[AnnotatedDocument]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x.doc_id):
        h.update(d.doc_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(d.text.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def save_crf_model(crf: sklearn_crfsuite.CRF, dst: Path) -> None:
    src = getattr(crf.modelfile, "name", crf.modelfile)
    if not src or not Path(src).exists():
        raise RuntimeError("CRF modelfile does not exist on disk; fit the CRF before saving.")
    shutil.copy2(src, dst)


def load_crf_model(src: Path) -> sklearn_crfsuite.CRF:
    if not src.exists():
        raise FileNotFoundError(f"Missing CRF model file: {src}")
    return sklearn_crfsuite.CRF(model_filename=str(src))


def save_npz_params(dst: Path, params: dict[str, np.ndarray], meta: dict[str, int | float | str]) -> None:
    payload: dict[str, Any] = {f"p__{k}": v for k, v in params.items()}
    for k, v in meta.items():
        payload[f"m__{k}"] = np.array(v)
    np.savez(dst, **payload)


def load_npz_params(src: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    params: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    with np.load(src, allow_pickle=False) as data:
        for key in data.files:
            if key.startswith("p__"):
                params[key[3:]] = np.array(data[key], copy=True)
            elif key.startswith("m__"):
                val = data[key]
                meta[key[3:]] = val.item() if val.ndim == 0 else val.tolist()
    return params, meta


def write_manifest(out_dir: Path, metadata: dict[str, Any]) -> Path:
    files: dict[str, str] = {}
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name != "manifest.json":
            files[p.name] = sha256_file(p)
    manifest = {
        **metadata,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def verify_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in model directory: {model_dir}")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    for rel, expected_sha in files.items():
        target = model_dir / rel
        if not target.exists():
            raise FileNotFoundError(f"Model artifact missing: {target}")
        actual = sha256_file(target)
        if actual != expected_sha:
            raise ValueError(f"SHA-256 mismatch for {target}: expected {expected_sha}, got {actual}")
    return manifest
