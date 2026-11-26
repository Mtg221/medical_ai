"""Step 1 regression tests: W12 log isolation, W6 canonical split counts, W14 length-scaled timeout."""

from __future__ import annotations

from pathlib import Path

from medresearch_platform.config import PlatformConfig
from medresearch_platform.vision.card import generate_derm_corpus_card
from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.sync import (
    compute_split_counts,
    freeze_locked_test_v2,
    write_derm_v2_data_card,
)


def _sample_record(idx: int, *, mg: str = "benign", modality: str = "dermoscopic", tone: str = "II") -> DermCatalogRecord:
    return DermCatalogRecord(
        image_id=f"ISIC_{idx:07d}",
        dataset="isic",
        modality=modality,
        thumb_path=None,
        raw_path=None,
        skin_tone=tone,
        diagnosis_raw="nevus" if mg == "benign" else "melanoma",
        malignancy_group=mg,
        body_site="back",
        license_name="CC-BY-4.0",
        quality_ok=True,
        confirmation="histopathology",
        group_key=f"lesion_{idx}",
        split_assignment="train",
        research_allowed=True,
    )


def test_w12_audit_and_freeze_do_not_modify_real_test_set_log(tmp_path: Path) -> None:
    real_log = Path("docs/results/TEST_SET_LOG.md")
    before = real_log.read_bytes()

    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    # Minimal manifest json so generate_derm_corpus_card runs audit & logging
    (meta_dir / "manifest-dermoscopic-000.json").write_text(
        '{"records": [{"id": "ISIC_0000001", "ds": "isic", "mod": "dermoscopic", "dx": "nevus", '
        '"mg": "benign", "cf": "histopathology", "tone": "II", "ok": true, "lic": "CC-BY-4.0"}]}',
        encoding="utf-8",
    )
    generate_derm_corpus_card(meta_dir, out_md_path=tmp_path / "CARD.md", log_test_set=True)
    freeze_locked_test_v2([_sample_record(1)], out_path=tmp_path / "locked_test_v2.txt")

    after = real_log.read_bytes()
    assert after == before, "Pytest audit/freeze must never mutate tracked docs/results/TEST_SET_LOG.md"


def test_w6_canonical_split_counts_match_data_card(tmp_path: Path) -> None:
    split_records = {
        "train": [_sample_record(1, mg="benign"), _sample_record(2, mg="malignant"), _sample_record(3, mg="unknown")],
        "val": [_sample_record(4, mg="benign", modality="clinical", tone="IV")],
        "locked_test_v2": [_sample_record(5, mg="malignant", modality="tbp_tile", tone="VI")],
    }
    counts = compute_split_counts(split_records)
    card_path = write_derm_v2_data_card(split_records, out_path=tmp_path / "CARD_V2.md", dropped_near_dups=0)
    card_text = card_path.read_text(encoding="utf-8")

    assert counts["train"]["total"] == 3
    assert counts["train"]["labelled_mg"] == 2
    assert counts["train"]["masked_mg"] == 1
    assert "| `train` | 3 | 1 | 1 | 1 | 0 | 3 | 0 |" in card_text
    assert "| `val` | 1 | 1 | 0 | 0 | 1 | 0 | 0 |" in card_text
    assert "| `locked_test_v2` | 1 | 0 | 1 | 0 | 0 | 0 | 1 |" in card_text


def test_w14_document_job_timeout_scales_with_length(tmp_path: Path) -> None:
    cfg = PlatformConfig.from_env(workspace_dir=tmp_path)
    t_small = cfg.compute_document_job_timeout(1_000)
    t_300k = cfg.compute_document_job_timeout(300_000)
    assert t_small < t_300k
    assert t_300k >= 300.0
