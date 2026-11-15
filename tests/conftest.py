"""Session-wide pytest fixtures ensuring fresh-clone test independence."""

from __future__ import annotations

from pathlib import Path

import pytest

from medresearch_platform.data.builder import build_corpus_from_config


@pytest.fixture(scope="session", autouse=True)
def ensure_v3_and_v3_1_corpora() -> None:
    """Build SYNTH-MED-V3 and SYNTH-MED-V3.1 automatically if missing (e.g. fresh clone)."""
    v3_train = Path("data/processed/v3/train.jsonl")
    v3_val = Path("data/processed/v3/val.jsonl")
    if not v3_train.exists() or not v3_val.exists():
        build_corpus_from_config(Path("configs/data/v3.yaml"))

    v3_1_train = Path("data/processed/v3_1/train.jsonl")
    v3_1_val = Path("data/processed/v3_1/val.jsonl")
    if not v3_1_train.exists() or not v3_1_val.exists():
        build_corpus_from_config(Path("configs/data/v3_1.yaml"))
