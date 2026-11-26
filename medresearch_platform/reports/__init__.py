"""Traceable bilingual (EN/FR) human-readable clinical extraction report generators."""

from medresearch_platform.reports.generator import (
    ReportFormat,
    ReportLanguage,
    generate_report_bytes,
    render_html_report,
    render_markdown_report,
    render_pdf_report,
)

__all__ = [
    "ReportFormat",
    "ReportLanguage",
    "generate_report_bytes",
    "render_html_report",
    "render_markdown_report",
    "render_pdf_report",
]
