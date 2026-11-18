"""Correction Prompt 18 — Step 1: End-to-end served-path evaluation on locked_test_v3 via real API."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from medresearch_platform.config import PlatformConfig
from medresearch_platform.interfaces.api import create_app
from medresearch_platform.vision.catalog import DermCatalogRecord
from medresearch_platform.vision.eval_vision import (
    compute_comprehensive_metrics,
    load_stratified_skin_dataset,
)
from medresearch_platform.vision.splits import log_locked_test_access
from scripts.run_step5_v3_benchmark import _compute_corpus_modality_prevalence, _reweighted_view


def main() -> None:
    t0 = time.perf_counter()
    splits = load_stratified_skin_dataset(
        meta_dir="var/cache/derm/meta",
        image_size=96,
        include_masked_mg=True,
        include_fitzpatrick17k_labelled=True,
        locked_test_list_path="configs/vision/locked_test_v3.txt",
        val_list_path="configs/vision/val_v3.txt",
        lesion_groups_path="configs/vision/lesion_groups_v3.json",
        min_real_samples=100,
    )
    test_lab = [s for s in splits["locked_test"] if s[2].malignancy_group in ("benign", "malignant")]
    print(f"[P18 Served-Path] Evaluating {len(test_lab)} locked_test_v3 images through POST /api/v1/images/upload ...", flush=True)

    repo_root = Path.cwd().resolve()
    os.environ["MEDRESEARCH_VISION_ROUTER_DIR"] = str(repo_root / "models" / "vision_router_v1")
    os.environ["MEDRESEARCH_VISION_SKIN_DIR"] = str(repo_root / "models" / "derm_cnn_seed1")

    routing_counts: Counter[str] = Counter()
    abstention_counts: Counter[str] = Counter()
    y_true_all: list[int] = []
    p_mal_all: list[float] = []
    abstained_all: list[bool] = []
    recs_all: list[DermCatalogRecord] = []
    dx_logits_dummy = np.zeros((len(test_lab), 7), dtype=np.float64)

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        cfg = PlatformConfig.from_env(workspace_dir=ws)
        app = create_app(cfg)
        app.state.service.init_or_reset_default_users("TestPass_2026!")
        now_iso = datetime.now(timezone.utc).isoformat()
        with app.state.service.db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO datasets ("
                "dataset_id, institution_id, name, version, source_type, license, consent_basis, "
                "permitted_purpose, deidentification_status, languages_json, modalities_json, "
                "record_count, manifest_sha256, retention_policy, created_at"
                ") VALUES ('ds_synth_bilingual_v1', 'inst_dakar_primary', 'Test', '3.1', 'SYNTHETIC', 'MIT', "
                "'RESEARCH', 'BENCHMARK', 'SYNTHETIC_NO_PHI', '[\"en\",\"fr\"]', '[\"TXT\"]', 0, 'abc', 'RETAIN', ?)",
                (now_iso,),
            )
            conn.execute(
                "INSERT OR IGNORE INTO subjects (subject_id, institution_id, dataset_id, pseudonym_code, language_primary, consent_status, created_at) "
                "VALUES ('subj_synth_001', 'inst_dakar_primary', 'ds_synth_bilingual_v1', 'PAT-SYNTH-001', 'en', 'SYNTHETIC_EXEMPT', ?)",
                (now_iso,),
            )

        client = TestClient(app, raise_server_exceptions=False)
        tok = client.post(
            "/api/v1/auth/token",
            json={"username": "admin_dakar", "password": "TestPass_2026!"},
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {tok}"}

        for idx, (_arr, lbl, rec) in enumerate(test_lab):
            rel = str(rec.thumb_path or rec.raw_path or "")
            candidates = [Path(rel), Path("var/cache/derm/thumbs") / rel, Path("var/cache/derm") / rel]
            img_path = next((p for p in candidates if p.exists()), candidates[1])
            raw_bytes = img_path.read_bytes()
            mime = "image/webp" if img_path.suffix.lower() == ".webp" else "image/jpeg"
            data: dict[str, str] = {"subject_id": "subj_synth_001", "force_new": "true"}
            if rec.body_site:
                data["body_site"] = str(rec.body_site)

            resp = client.post(
                "/api/v1/images/upload",
                headers=headers,
                data=data,
                files={"file": (img_path.name, raw_bytes, mime)},
            )
            if resp.status_code in (200, 201):
                payload = resp.json()
                routed_to = str(payload.get("routed_to") or "")
                if routed_to == "ocr_text_pipeline":
                    routing_counts["document_scan"] += 1
                    p_mal = 0.5
                    is_abstained = True
                    abstention_counts["routed_document_scan"] += 1
                else:
                    pic_type = str(payload.get("picture_type") or "")
                    routing_counts[pic_type or "skin_unknown"] += 1
                    if pic_type in ("other", "document_scan"):
                        p_mal = 0.5
                        is_abstained = True
                        abstention_counts[f"routed_{pic_type}"] += 1
                    else:
                        p_mal = float(payload.get("malignancy_risk_probability") or 0.5)
                        is_abstained = bool(payload.get("abstained", False))
                        if is_abstained:
                            abstention_counts["skin_abstained"] += 1
                        else:
                            abstention_counts["skin_scored"] += 1
            else:
                # e.g., document_scan routed to OCR where Tesseract is absent on macOS
                routing_counts["document_scan_or_ocr_error"] += 1
                p_mal = 0.5
                is_abstained = True
                abstention_counts["routed_document_scan_ocr_err"] += 1

            y_true_all.append(int(lbl))
            p_mal_all.append(p_mal)
            abstained_all.append(is_abstained)
            recs_all.append(rec)

            if (idx + 1) % 400 == 0:
                print(f"  processed {idx + 1}/{len(test_lab)} ...", flush=True)

    p_arr = np.asarray(p_mal_all, dtype=np.float64)
    abs_arr = np.asarray(abstained_all, dtype=bool)

    overall_metrics = compute_comprehensive_metrics(
        test_lab,
        p_arr,
        dx_logits_dummy,
        abstained=abs_arr,
    )

    by_modality: dict[str, Any] = {}
    for mod in ("dermoscopic", "clinical", "tbp_tile"):
        mask = np.asarray([r.modality == mod for r in recs_all])
        sub_samples = [s for s, m in zip(test_lab, mask) if m]
        by_modality[mod] = compute_comprehensive_metrics(
            sub_samples,
            p_arr[mask],
            dx_logits_dummy[mask],
            abstained=abs_arr[mask],
        )

    corpus_prev = _compute_corpus_modality_prevalence()
    view_b = _reweighted_view(test_lab, p_arr, abs_arr, corpus_prev)

    log_locked_test_access(
        dataset_name="locked_test_v3",
        model_name="served_path_derm_cnn_bsdrop40_seed1",
        reason="served-path evaluation X1",
        sample_count=len(test_lab),
    )

    out: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "endpoint": "POST /api/v1/images/upload",
        "router_dir": "models/vision_router_v1",
        "skin_model_dir": "models/derm_cnn_seed1",
        "skin_model_manifest": json.loads(Path("models/derm_cnn_seed1/manifest.json").read_text(encoding="utf-8")),
        "n_locked_test_v3": len(test_lab),
        "routing_counts": dict(routing_counts),
        "abstention_counts": dict(abstention_counts),
        "served_path_overall": overall_metrics,
        "served_path_by_modality": by_modality,
        "view_b_corpus_prevalence_reweighted": view_b,
    }
    out_path = Path("docs/results/vision/p18_step1_served_path_eval.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[P18 Served-Path] DONE in {out['elapsed_seconds']}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
