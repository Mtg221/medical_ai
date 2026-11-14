"""Tests for native PDF/DOCX/TXT/PNG parsing, language detection, sectioning, backup/restore, and leak-free splits."""

from __future__ import annotations

from pathlib import Path
import pytest

from medresearch_platform.config import PlatformConfig
from medresearch_platform.db.engine import DatabaseEngine
from medresearch_platform.documents.parsers import (
    detect_document_language,
    parse_document_bytes,
    segment_page_sections,
)
from medresearch_platform.documents.synthetic_generator import (
    SYNTHETIC_CLINICAL_CORPUS,
    build_synthetic_file_bytes,
    verify_leak_free_splits,
)


def test_language_detection_en_fr_and_mixed() -> None:
    en_txt = "Patient presents with chest pain and shortness of breath. Prescribed metformin daily."
    fr_txt = "Le patient présente une douleur thoracique et une dyspnée sans fièvre. Traitement prescrit."
    mixed_txt = "Patient presents with chest pain and dyspnée; traitement prescribed daily."

    assert detect_document_language(en_txt) == "en"
    assert detect_document_language(fr_txt) == "fr"
    assert detect_document_language(mixed_txt) == "mixed_en_fr"


def test_native_pdf_docx_txt_parsing_and_sections() -> None:
    for spec in SYNTHETIC_CLINICAL_CORPUS:
        raw_bytes, _ = build_synthetic_file_bytes(spec)
        bundle = parse_document_bytes(
            raw_bytes,
            spec.file_format,
            document_type=spec.document_type,
            specialty=spec.specialty,
        )
        assert bundle.page_count == 1
        assert bundle.detected_language == spec.language
        assert len(bundle.pages[0].sections) >= 3
        section_types = {s.section_type for s in bundle.pages[0].sections}
        assert "CHIEF_COMPLAINT" in section_types
        assert "MEDICATIONS" in section_types
        assert "ASSESSMENT_PLAN" in section_types


def test_leak_free_patient_and_document_family_splits() -> None:
    report = verify_leak_free_splits(SYNTHETIC_CLINICAL_CORPUS)
    assert report["leak_detected"] is False
    assert report["subject_overlaps"]["train_val"] == []
    assert report["subject_overlaps"]["train_locked_test"] == []
    assert report["subject_overlaps"]["val_locked_test"] == []


def test_database_backup_and_restore_integrity(tmp_path: Path) -> None:
    cfg = PlatformConfig(
        profile="test",
        workspace_dir=tmp_path,
        storage_dir=tmp_path / "storage",
        sqlite_path=tmp_path / "db" / "main.sqlite3",
        postgres_url=None,
        max_cpu_threads=2,
        max_memory_mb=1024,
        max_upload_bytes=1024 * 1024,
        max_document_pages=20,
        default_random_seed=42,
        token_secret_key="sec1",
        pseudonym_vault_key="sec2",
    )
    db = DatabaseEngine(cfg)
    db.apply_migrations()

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO institutions (institution_id, name, country_code, governance_status, created_at) "
            "VALUES ('inst_test', 'Test Inst', 'SN', 'ACTIVE', '2026-11-12T00:00:00Z')"
        )

    backup_file = tmp_path / "backups" / "snapshot.sqlite3"
    sha256_hash = db.backup_to(backup_file)
    assert backup_file.exists()

    # Delete the institution in live DB, then restore from verified backup
    with db.transaction() as conn:
        conn.execute("DELETE FROM institutions WHERE institution_id = 'inst_test'")

    with pytest.raises(ValueError, match="Backup integrity check failed"):
        db.restore_from(backup_file, expected_sha256="0" * 64)

    db.restore_from(backup_file, expected_sha256=sha256_hash)
    with db.connect() as conn:
        row = conn.execute("SELECT name FROM institutions WHERE institution_id = 'inst_test'").fetchone()
        assert row is not None
        assert str(row["name"]) == "Test Inst"
