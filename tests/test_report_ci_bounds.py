"""Ensure every `estimate [low, high]` CI in docs/RESEARCH_REPORT.md contains its estimate."""

from __future__ import annotations

import re
from pathlib import Path

_CI_PATTERN = re.compile(r"\*{0,2}(\d+\.\d+)\*{0,2}`?\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]")


def test_research_report_ci_bounds_contain_estimate() -> None:
    report_path = Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_REPORT.md"
    text = report_path.read_text(encoding="utf-8")
    matches = _CI_PATTERN.findall(text)
    assert len(matches) >= 10, f"Expected >= 10 CI entries in RESEARCH_REPORT.md, found {len(matches)}"
    for est_s, low_s, high_s in matches:
        est, low, high = float(est_s), float(low_s), float(high_s)
        assert low <= est <= high, f"CI [{low}, {high}] does not contain estimate {est}"
