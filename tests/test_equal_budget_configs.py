"""Tests for Phase 11B Step 2: equal 600-doc training budget across all configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from medresearch_platform.approaches.classical_ml import _select_train_docs
from medresearch_platform.eval_harness.runner import _load_jsonl


def test_v3_1_configs_use_equal_full_budget() -> None:
    cfg_dir = Path("configs/exp/v3_1")
    yamls = sorted(cfg_dir.glob("*.yaml"))
    assert len(yamls) == 7
    for p in yamls:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        max_docs = cfg.get("max_train_docs", "MISSING")
        assert max_docs is None or int(max_docs) >= 600, (
            f"{p} must set max_train_docs: null (or >= 600), got {max_docs!r}"
        )


def test_select_train_docs_none_returns_all() -> None:
    docs = _load_jsonl(Path("data/processed/v3_1/train.jsonl"))[:20]
    out = _select_train_docs(docs, seed=1, cfg={"max_train_docs": None})
    assert len(out) == 20
    sub = _select_train_docs(docs, seed=1, cfg={"max_train_docs": 8})
    assert len(sub) == 8
