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


def _serialize_tfidf_vectorizer(vec: Any, prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    params = vec.get_params(deep=False)
    is_fitted = hasattr(vec, "vocabulary_") and vec.vocabulary_ is not None
    meta: dict[str, Any] = {
        "fitted": is_fitted,
        "analyzer": params.get("analyzer", "word"),
        "ngram_range": list(params.get("ngram_range", (1, 1))),
        "max_features": params.get("max_features"),
        "sublinear_tf": bool(params.get("sublinear_tf", False)),
        "lowercase": bool(params.get("lowercase", True)),
        "norm": params.get("norm", "l2"),
        "use_idf": bool(params.get("use_idf", True)),
        "smooth_idf": bool(params.get("smooth_idf", True)),
        "token_pattern": params.get("token_pattern", r"(?u)\b\w\w+\b"),
    }
    if is_fitted:
        meta["vocabulary"] = {str(k): int(v) for k, v in vec.vocabulary_.items()}
        if hasattr(vec, "idf_"):
            arrays[f"{prefix}__idf"] = np.asarray(vec.idf_, dtype=np.float64)
        if hasattr(vec, "fixed_vocabulary_"):
            meta["fixed_vocabulary"] = bool(vec.fixed_vocabulary_)
    return meta


def _deserialize_tfidf_vectorizer(meta: dict[str, Any], prefix: str, arrays: dict[str, np.ndarray]) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        analyzer=meta.get("analyzer", "word"),
        ngram_range=tuple(meta.get("ngram_range", [1, 1])),
        max_features=meta.get("max_features"),
        sublinear_tf=bool(meta.get("sublinear_tf", False)),
        lowercase=bool(meta.get("lowercase", True)),
        norm=meta.get("norm", "l2"),
        use_idf=bool(meta.get("use_idf", True)),
        smooth_idf=bool(meta.get("smooth_idf", True)),
        token_pattern=meta.get("token_pattern", r"(?u)\b\w\w+\b"),
    )
    if meta.get("fitted"):
        vec.vocabulary_ = {str(k): int(v) for k, v in meta.get("vocabulary", {}).items()}
        vec.fixed_vocabulary_ = bool(meta.get("fixed_vocabulary", False))
        idf_key = f"{prefix}__idf"
        if idf_key in arrays:
            vec.idf_ = np.asarray(arrays[idf_key], dtype=np.float64)
    return vec


def _serialize_linear_classifier(clf: Any, prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    cls_name = type(clf).__name__
    is_fitted = hasattr(clf, "classes_") and hasattr(clf, "coef_")
    meta: dict[str, Any] = {
        "class_name": cls_name,
        "fitted": is_fitted,
        "random_state": getattr(clf, "random_state", None),
        "max_iter": getattr(clf, "max_iter", 1000),
        "class_weight": getattr(clf, "class_weight", None),
    }
    if is_fitted:
        meta["classes"] = [str(c) for c in clf.classes_]
        arrays[f"{prefix}__coef"] = np.asarray(clf.coef_, dtype=np.float64)
        arrays[f"{prefix}__intercept"] = np.asarray(clf.intercept_, dtype=np.float64)
        if hasattr(clf, "n_features_in_"):
            meta["n_features_in"] = int(clf.n_features_in_)
    return meta


def _deserialize_linear_classifier(meta: dict[str, Any], prefix: str, arrays: dict[str, np.ndarray]) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    cls_name = str(meta.get("class_name", "LinearSVC"))
    if cls_name == "LogisticRegression":
        clf: Any = LogisticRegression(
            max_iter=int(meta.get("max_iter", 300)),
            random_state=meta.get("random_state"),
            class_weight=meta.get("class_weight"),
        )
    else:
        clf = LinearSVC(
            random_state=meta.get("random_state"),
            max_iter=int(meta.get("max_iter", 1000)),
            class_weight=meta.get("class_weight"),
        )
    if meta.get("fitted"):
        clf.classes_ = np.asarray(meta["classes"], dtype=object)
        clf.coef_ = np.asarray(arrays[f"{prefix}__coef"], dtype=np.float64)
        clf.intercept_ = np.asarray(arrays[f"{prefix}__intercept"], dtype=np.float64)
        if "n_features_in" in meta:
            clf.n_features_in_ = int(meta["n_features_in"])
    return clf


def save_sklearn_text_bundle(
    json_path: Path,
    npz_path: Path,
    *,
    vectorizers: dict[str, Any],
    classifiers: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Serialize scikit-learn TfidfVectorizer/FeatureUnion + LinearSVC/LogisticRegression without pickle."""
    from sklearn.pipeline import FeatureUnion

    arrays: dict[str, Any] = {}
    vec_meta: dict[str, Any] = {}
    for name, vec in vectorizers.items():
        if isinstance(vec, FeatureUnion):
            vec_meta[name] = {
                "kind": "FeatureUnion",
                "transformers": [
                    {
                        "name": sub_name,
                        "spec": _serialize_tfidf_vectorizer(sub_vec, f"v__{name}__{sub_name}", arrays),
                    }
                    for sub_name, sub_vec in vec.transformer_list
                ],
            }
        else:
            vec_meta[name] = {
                "kind": "TfidfVectorizer",
                "spec": _serialize_tfidf_vectorizer(vec, f"v__{name}", arrays),
            }

    clf_meta = {
        name: _serialize_linear_classifier(clf, f"c__{name}", arrays)
        for name, clf in classifiers.items()
    }
    payload = {
        "format": "safe_sklearn_text_bundle_v1",
        "vectorizers": vec_meta,
        "classifiers": clf_meta,
        "extra": extra or {},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(npz_path, **arrays)


def load_sklearn_text_bundle(
    json_path: Path,
    npz_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Deserialize safe JSON+NPZ (allow_pickle=False) sklearn text bundle."""
    from sklearn.pipeline import FeatureUnion

    if not json_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"Safe sklearn bundle missing: {json_path} / {npz_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("format") != "safe_sklearn_text_bundle_v1":
        raise ValueError(f"Unsupported sklearn bundle format: {payload.get('format')!r}")

    arrays: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for k in data.files:
            arrays[k] = np.array(data[k], copy=True)

    vectorizers: dict[str, Any] = {}
    for name, spec in payload.get("vectorizers", {}).items():
        if spec.get("kind") == "FeatureUnion":
            subs = [
                (
                    str(item["name"]),
                    _deserialize_tfidf_vectorizer(item["spec"], f"v__{name}__{item['name']}", arrays),
                )
                for item in spec.get("transformers", [])
            ]
            vectorizers[name] = FeatureUnion(subs)
        else:
            vectorizers[name] = _deserialize_tfidf_vectorizer(spec["spec"], f"v__{name}", arrays)

    classifiers = {
        name: _deserialize_linear_classifier(spec, f"c__{name}", arrays)
        for name, spec in payload.get("classifiers", {}).items()
    }
    return vectorizers, classifiers, dict(payload.get("extra", {}))
