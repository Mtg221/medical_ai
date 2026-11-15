"""Session-wide pytest fixtures ensuring fresh-clone test independence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from medresearch_platform.data.builder import build_corpus_from_config


@pytest.fixture(scope="session", autouse=True)
def ensure_v3_and_v3_1_corpora(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Build SYNTH-MED-V3 and SYNTH-MED-V3.1 automatically if missing (e.g. fresh clone)."""
    v3_train = Path("data/processed/v3/train.jsonl")
    v3_val = Path("data/processed/v3/val.jsonl")
    if not v3_train.exists() or not v3_val.exists():
        build_corpus_from_config(Path("configs/data/v3.yaml"))

    v3_1_train = Path("data/processed/v3_1/train.jsonl")
    v3_1_val = Path("data/processed/v3_1/val.jsonl")
    if not v3_1_train.exists() or not v3_1_val.exists():
        build_corpus_from_config(Path("configs/data/v3_1.yaml"))

    default_model = Path("models/hybrid_v2_seed1")
    if not default_model.exists() and "MEDRESEARCH_MODEL_DIR" not in os.environ:
        import json

        from medresearch_platform.data.schema import AnnotatedDocument
        from medresearch_platform.hybrid.pipeline import HybridConfig, HybridPipeline

        tmp_model_dir = tmp_path_factory.mktemp("hybrid_v2_test_model")

        docs: list[AnnotatedDocument] = []
        with v3_1_train.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(AnnotatedDocument(**json.loads(line)))
                    if len(docs) >= 60:
                        break
        pipe = HybridPipeline(HybridConfig(seed=1, ner_epochs=6, use_temporal_stage=True, use_relation_stage=True))
        pipe.fit(docs)
        pipe.save(tmp_model_dir, corpus_id="SYNTH-MED-V3.1-test-subset")
        os.environ["MEDRESEARCH_MODEL_DIR"] = str(tmp_model_dir)
