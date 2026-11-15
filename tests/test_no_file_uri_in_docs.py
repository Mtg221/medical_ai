"""Guard against machine-local file:/// links in Markdown documentation (H9)."""

from __future__ import annotations

import re
from pathlib import Path


def test_no_file_uri_links_in_docs() -> None:
    md_files = [
        Path("README.md"),
        Path("PROJECT_STATUS.md"),
        Path("CHANGELOG.md"),
        *sorted(Path("docs").rglob("*.md")),
        *sorted(Path("DECISIONS").glob("*.md")),
    ]
    pattern = re.compile(r"(\]\(|<|href=[\"'])file:///")
    offenders: list[str] = []
    for p in md_files:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            offenders.append(str(p))
    assert not offenders, f"Found machine-local file:/// links in docs: {offenders}"
