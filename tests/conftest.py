"""Session-wide pytest fixtures ensuring fresh-clone test independence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.data.builder import build_corpus_from_config
from medresearch_platform.eval_harness.runner import _load_jsonl
from medresearch_platform.hybrid.pipeline import HybridPipeline


def ensure_corpora_and_test_models(
    model_root: Path,
    *,
    force_build_models: bool = False,
) -> tuple[Path, Path]:
    """Build v3/v3_1 corpora if missing, and ensure tiny hybrid_v2 + classical_crf test models exist."""
    v3_train = Path("data/processed/v3/train.jsonl")
    v3_val = Path("data/processed/v3/val.jsonl")
    if not v3_train.exists() or not v3_val.exists():
        build_corpus_from_config(Path("configs/data/v3.yaml"))

    v3_1_train = Path("data/processed/v3_1/train.jsonl")
    v3_1_val = Path("data/processed/v3_1/val.jsonl")
    if not v3_1_train.exists() or not v3_1_val.exists():
        build_corpus_from_config(Path("configs/data/v3_1.yaml"))

    hybrid_dir = model_root / "hybrid_v2_test"
    crf_dir = model_root / "classical_crf_test"

    need_hybrid = force_build_models or (not Path("models/hybrid_v2_seed1").exists() and not hybrid_dir.exists())
    need_crf = force_build_models or (not Path("models/classical_crf_seed1").exists() and not crf_dir.exists())

    if need_hybrid or need_crf:
        docs = _load_jsonl(v3_1_train)[:60]

        if need_hybrid:
            cfg = yaml.safe_load(Path("configs/exp/v3_2/hybrid_v2.yaml").read_text(encoding="utf-8"))
            cfg = {**cfg, "epochs": 6, "max_train_docs": 60}
            pipe = HybridPipeline(seed=1, cfg=cfg)
            pipe.fit(docs)
            pipe.save(hybrid_dir, corpus_id="SYNTH-MED-V3.1-test-subset")

        if need_crf:
            crf = ClassicalCRFModel(seed=1)
            crf.fit(docs)
            crf.save(crf_dir, corpus_id="SYNTH-MED-V3.1-test-subset")

    resolved_hybrid = hybrid_dir if (force_build_models or not Path("models/hybrid_v2_seed1").exists()) else Path("models/hybrid_v2_seed1")
    resolved_crf = crf_dir if (force_build_models or not Path("models/classical_crf_seed1").exists()) else Path("models/classical_crf_seed1")
    return resolved_hybrid, resolved_crf


@pytest.fixture(scope="session", autouse=True)
def ensure_v3_and_v3_1_corpora(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Ensure corpora, fallback test models, and isolated temp DB/storage exist for test runs."""
    session_runtime = tmp_path_factory.mktemp("test_runtime")
    if "MEDRESEARCH_DB_PATH" not in os.environ and "MEDRESEARCH_SQLITE_PATH" not in os.environ:
        temp_db = session_runtime / "pytest_medresearch.sqlite3"
        os.environ["MEDRESEARCH_SQLITE_PATH"] = str(temp_db)
    if "MEDRESEARCH_STORAGE_DIR" not in os.environ:
        os.environ["MEDRESEARCH_STORAGE_DIR"] = str(session_runtime / "storage")
    if "MEDRESEARCH_TEST_SET_LOG_PATH" not in os.environ:
        os.environ["MEDRESEARCH_TEST_SET_LOG_PATH"] = str(session_runtime / "TEST_SET_LOG.md")

    model_root = tmp_path_factory.mktemp("test_models")
    hybrid_dir, crf_dir = ensure_corpora_and_test_models(model_root)
    if "MEDRESEARCH_MODEL_DIR" not in os.environ and not Path("models/hybrid_v2_seed1").exists():
        os.environ["MEDRESEARCH_MODEL_DIR"] = str(hybrid_dir)
    if "MEDRESEARCH_CLASSICAL_CRF_DIR" not in os.environ and not Path("models/classical_crf_seed1").exists():
        os.environ["MEDRESEARCH_CLASSICAL_CRF_DIR"] = str(crf_dir)


@pytest.fixture(scope="session")
def isolated_test_models(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Isolated session temp models (hybrid_v2 + classical_crf) independent of repo models/."""
    model_root = tmp_path_factory.mktemp("isolated_api_models")
    return ensure_corpora_and_test_models(model_root, force_build_models=True)
