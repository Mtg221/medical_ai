"""Direct test of tests/conftest.py fresh-clone model builder helper (K1)."""

from __future__ import annotations

from pathlib import Path

from conftest import ensure_corpora_and_test_models

from medresearch_platform.serving.extractor import ExtractionService


def test_conftest_helper_builds_valid_loadable_models(tmp_path: Path) -> None:
    hybrid_dir, crf_dir = ensure_corpora_and_test_models(tmp_path, force_build_models=True)
    assert (hybrid_dir / "manifest.json").exists()
    assert (crf_dir / "manifest.json").exists()

    svc = ExtractionService(model_dir=hybrid_dir, classical_crf_dir=crf_dir)
    loaded = svc.list_loaded_models()
    assert len(loaded) == 2
