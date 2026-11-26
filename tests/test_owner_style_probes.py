"""L1 owner-style free-text probes recorded as known xfail failures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medresearch_platform.serving.extractor import ExtractionService

_PROBE_PATH = Path(__file__).resolve().parent / "data" / "owner_style_probe.jsonl"
_PROBES = [json.loads(line) for line in _PROBE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.xfail(reason="Known OOD owner-style free-text failures on SYNTH-MED-V3.1 model (L1)", strict=False)
@pytest.mark.parametrize("probe", _PROBES, ids=[p["probe_id"] for p in _PROBES])
def test_owner_style_probe_known_failures(probe: dict) -> None:
    model_dir = Path("models/hybrid_v2_seed1")
    if not (model_dir / "manifest.json").exists():
        pytest.skip("requires trained models/hybrid_v2_seed1")

    svc = ExtractionService(model_dir=model_dir)
    res = svc.extract(probe["text"], document_id=probe["probe_id"], language=probe["language"])
    by_text = {r.original_text: r for r in res.records}

    for must in probe.get("expected_must_extract", []):
        assert must in by_text, f"Missed entity {must!r}"

    for rule in probe.get("expected_must_not_mislabel", []):
        rec = by_text.get(rule["text"])
        if "forbidden_type" in rule and rec is not None:
            assert rec.entity_type != rule["forbidden_type"], f"{rule['text']} mislabeled as {rec.entity_type}"
        if "expected_type" in rule:
            assert rec is not None and rec.entity_type == rule["expected_type"], (
                f"{rule['text']} expected {rule['expected_type']}, got {rec.entity_type if rec else None}"
            )
