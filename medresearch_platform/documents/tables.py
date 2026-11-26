"""Structured table extraction utilities for PDF lab reports and DOCX clinical tables."""

from __future__ import annotations

import io
from dataclasses import dataclass

import pdfplumber


@dataclass(frozen=True)
class ExtractedTable:
    """Structured table extracted from a PDF or DOCX clinical document."""

    page_number: int
    headers: list[str]
    rows: list[list[str]]
    bbox: dict[str, int]

    def to_lines(self) -> list[str]:
        """Render table rows into readable pipe-delimited lines."""
        lines: list[str] = []
        if self.headers:
            lines.append(" | ".join(self.headers))
        for row in self.rows:
            if any(cell.strip() for cell in row):
                lines.append(" | ".join(row))
        return lines


def extract_pdf_tables(pdf_bytes: bytes) -> dict[int, list[ExtractedTable]]:
    """Extract structured tables per 1-indexed page from digital PDF bytes using pdfplumber."""
    tables_by_page: dict[int, list[ExtractedTable]] = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            found = page.find_tables()
            if not found:
                continue
            page_tables: list[ExtractedTable] = []
            for tbl in found:
                raw_rows = tbl.extract() or []
                cleaned_rows = [
                    [str(cell or "").replace("\n", " ").strip() for cell in row]
                    for row in raw_rows
                    if row and any(str(c or "").strip() for c in row)
                ]
                if not cleaned_rows:
                    continue
                headers = cleaned_rows[0]
                body_rows = cleaned_rows[1:] if len(cleaned_rows) > 1 else []
                x0, top, x1, bottom = tbl.bbox
                page_tables.append(
                    ExtractedTable(
                        page_number=page_idx,
                        headers=headers,
                        rows=body_rows,
                        bbox={
                            "x": int(round(x0)),
                            "y": int(round(top)),
                            "width": max(1, int(round(x1 - x0))),
                            "height": max(1, int(round(bottom - top))),
                        },
                    )
                )
            if page_tables:
                tables_by_page[page_idx] = page_tables
    return tables_by_page
