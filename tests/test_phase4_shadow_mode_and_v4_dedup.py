"""Unit tests for Phase 4 deliverables: shadow-mode clinical gate harness & external derm ingestion."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.eval_shadow_mode import evaluate_shadow_mode_records
from scripts.ingest_external_clinical_derm import (
    ingest_external_derm_csv,
    normalize_fitzpatrick,
    normalize_malignancy_label,
)


def test_shadow_mode_gates_pass_and_detect_fitzpatrick_disparity() -> None:
    # Build a synthetic shadow-mode log with strong calibrated performance across I-II, III-IV, V-VI
    records: list[dict[str, object]] = []
    idx = 0
    for mod in ("dermoscopic", "clinical"):
        for fitz in ("I-II", "III-IV", "V-VI"):
            for lbl in (0, 1):
                for rep in range(10):
                    idx += 1
                    prob = 0.97 if lbl == 1 else 0.03
                    records.append(
                        {
                            "case_id": f"C{idx}",
                            "modality": mod,
                            "fitzpatrick": fitz,
                            "predicted_prob_malignant": prob,
                            "abstained": False,
                            "ground_truth_malignant": lbl,
                            "clinician_agreed": True,
                            "latency_ms": 450.0 + rep,
                        }
                    )

    res_ok = evaluate_shadow_mode_records(records)
    assert res_ok["verdict"] == "READY_FOR_SUPERVISED_PILOT"
    assert res_ok["fitzpatrick_sens_gap"] == 0.0
    assert all(g["status"] == "PASS" for g in res_ok["promotion_gates"].values())

    # Now degrade V-VI sensitivity -> must fail Fitzpatrick parity gate and block promotion
    degraded = [
        (
            {**r, "predicted_prob_malignant": 0.01}
            if r["fitzpatrick"] == "V-VI" and r["ground_truth_malignant"] == 1
            else r
        )
        for r in records
    ]
    res_fail = evaluate_shadow_mode_records(degraded)
    assert res_fail["verdict"] == "BLOCKED_FAILED_GATES"
    assert res_fail["promotion_gates"]["fitzpatrick_sens_gap"]["status"] == "FAIL"


def test_external_derm_ingestion_deduplicates_against_heldout_sha256(tmp_path: Path) -> None:
    assert normalize_malignancy_label("MEL") == "malignant"
    assert normalize_malignancy_label("Basal Cell Carcinoma") == "malignant"
    assert normalize_malignancy_label("NEV") == "benign"
    assert normalize_fitzpatrick("5") == "V-VI"
    assert normalize_fitzpatrick("II") == "I-II"

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    p1 = img_dir / "case_1.png"
    p2 = img_dir / "case_2.png"
    p3 = img_dir / "case_3_dup.png"
    Image.fromarray(np.full((16, 16, 3), 120, dtype=np.uint8)).save(p1)
    Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8)).save(p2)
    p3.write_bytes(p2.read_bytes())  # exact internal duplicate of p2

    heldout_sha = hashlib.sha256(p1.read_bytes()).hexdigest()

    csv_path = tmp_path / "meta.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["img_id", "diagnostic", "fitspatrick", "patient_id", "region", "biopsed"],
        )
        w.writeheader()
        w.writerow({"img_id": "case_1.png", "diagnostic": "MEL", "fitspatrick": "5", "patient_id": "P1", "region": "arm", "biopsed": "True"})
        w.writerow({"img_id": "case_2.png", "diagnostic": "BCC", "fitspatrick": "4", "patient_id": "P2", "region": "face", "biopsed": "True"})
        w.writerow({"img_id": "case_3_dup.png", "diagnostic": "BCC", "fitspatrick": "4", "patient_id": "P3", "region": "face", "biopsed": "True"})

    out_manifest = tmp_path / "manifest.json"
    rep = ingest_external_derm_csv(
        csv_path=csv_path,
        images_dir=img_dir,
        dataset_name="pad_ufes_20",
        license_name="CC-BY-4.0",
        heldout_shas={heldout_sha},
        out_manifest=out_manifest,
    )
    assert rep["summary"]["ingested_items"] == 1
    assert rep["summary"]["skipped_heldout_sha256_collision"] == 1
    assert rep["summary"]["skipped_internal_duplicate"] == 1
    assert rep["items"][0]["st"] == "III-IV"
    assert rep["items"][0]["mg"] == "malignant"
    assert rep["items"][0]["cf"] == "histopathology"
