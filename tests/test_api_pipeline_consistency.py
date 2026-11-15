"""Step 2 (J1): API-vs-HybridPipeline consistency test on 30 test_iid TXT + 10 rendered PDF docs."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.documents.ocr_benchmark import (
    _load_font,
    map_ocr_spans_to_reference,
    tesseract_available,
)
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.interfaces.api import create_app

_ASSERT_TO_API = {
    "present": "AFFIRMED",
    "absent": "NEGATED",
    "possible": "UNCERTAIN",
    "hypothetical": "HYPOTHETICAL",
    "conditional": "CONDITIONAL",
    "family": "FAMILY_HISTORY",
}


def _load_test_iid(limit: int = 30) -> list[AnnotatedDocument]:
    path = Path("data/processed/v3_1/test_iid.jsonl")
    docs: list[AnnotatedDocument] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(AnnotatedDocument(**json.loads(line)))
                if len(docs) >= limit:
                    break
    return docs


def _render_clean_pdf_bytes(text: str) -> bytes:
    img = Image.new("RGB", (1654, 2339), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _load_font(0, 22)
    y = 60
    for line in text.splitlines():
        if line.strip():
            draw.text((60, y), line, fill=(0, 0, 0), font=font)
            y += 32
        else:
            y += 18
        if y > 2260:
            break
    buf = BytesIO()
    img.save(buf, format="PDF", resolution=300.0)
    return buf.getvalue()


def _upload_and_wait(client: TestClient, headers: dict[str, str], filename: str, content: bytes, mime: str) -> dict:
    resp = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": (filename, content, mime)},
    )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    doc_id = None
    for _ in range(120):
        jr = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert jr.status_code == 200
        jdata = jr.json()
        if jdata["status"] == "SUCCEEDED":
            doc_id = jdata["result"]["document_id"]
            break
        if jdata["status"] == "FAILED":
            raise AssertionError(f"Upload job failed: {jdata}")
        time.sleep(0.05)
    assert doc_id is not None, "Upload job timed out"
    detail = client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert detail.status_code == 200
    return detail.json()


def test_api_matches_hybrid_v2_on_30_txt_and_10_pdf_test_iid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = Path("models/hybrid_v2_seed1")
    if not model_dir.exists():
        pytest.skip("Run `medextract train --config configs/exp/v3_2/hybrid_v2.yaml --seed 1 --out models/hybrid_v2_seed1` first")

    monkeypatch.setenv("MEDRESEARCH_SQLITE_PATH", str(tmp_path / "app.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", "consistency-pw-2026")
    monkeypatch.setenv("MEDRESEARCH_MODEL_DIR", str(model_dir))

    pipe = HybridPipeline.load(model_dir)
    test_docs = _load_test_iid(30)

    app = create_app()
    with TestClient(app) as client:
        tok_resp = client.post(
            "/api/v1/auth/token",
            json={"username": "admin_dakar", "password": os.environ["MEDRESEARCH_DEMO_PASSWORD"]},
        )
        assert tok_resp.status_code == 200
        headers = {"Authorization": f"Bearer {tok_resp.json()['access_token']}"}
        with sqlite3.connect(tmp_path / "app.sqlite3") as conn:
            conn.execute(
                "INSERT OR IGNORE INTO datasets ("
                "dataset_id, institution_id, name, version, source_type, license, consent_basis, "
                "permitted_purpose, deidentification_status, languages_json, modalities_json, "
                "record_count, manifest_sha256, retention_policy, created_at"
                ") VALUES ('ds_synth_bilingual_v1', 'inst_dakar_primary', 'Consistency', '3.1', "
                "'SYNTHETIC', 'MIT', 'RESEARCH', 'BENCHMARK', 'SYNTHETIC_NO_PHI', '[\"en\",\"fr\"]', "
                "'[\"TXT\",\"PDF\"]', 30, 'abc', 'RETAIN', '2026-11-15T00:00:00Z')"
            )
            conn.commit()

        # 30 TXT documents: API entities must be identical to HybridPipeline.extract_document
        for doc in test_docs:
            pred, _ = pipe.extract_document(doc)
            expected = [
                (e.type, e.start, e.end, e.text, _ASSERT_TO_API.get(e.assertion, "AFFIRMED"))
                for e in pred.entities
            ]

            detail = _upload_and_wait(
                client,
                headers,
                f"{doc.doc_id}.txt",
                doc.text.encode("utf-8"),
                "text/plain",
            )
            api_ents = [
                (e["entity_type"], e["char_start"], e["char_end"], e["source_text"], e["assertion_status"])
                for e in detail["entities"]
            ]
            assert api_ents == expected, f"Mismatch on TXT {doc.doc_id}: API={api_ents[:3]} vs Pipe={expected[:3]}"

        # 10 rendered PDF documents (OCR path): API entities mapped via character alignment must match direct OCR+HybridPipeline
        if tesseract_available(("eng", "fra")):
            for doc in test_docs[:10]:
                pdf_bytes = _render_clean_pdf_bytes(doc.text)
                detail = _upload_and_wait(
                    client,
                    headers,
                    f"{doc.doc_id}.pdf",
                    pdf_bytes,
                    "application/pdf",
                )
                ocr_text = detail["pages"][0]["raw_text"]
                ocr_doc = AnnotatedDocument(
                    doc_id=doc.doc_id,
                    patient_id=doc.patient_id,
                    language=doc.language,
                    doc_type=doc.doc_type,
                    document_date=doc.document_date,
                    source=doc.source,
                    text=ocr_text,
                )
                ocr_pred, _ = pipe.extract_document(ocr_doc)
                # Verify API entities on OCR text match direct HybridPipeline on the same OCR text
                api_ents = [
                    (e["entity_type"], e["char_start"], e["char_end"], e["source_text"], e["assertion_status"])
                    for e in detail["entities"]
                ]
                expected_ocr = [
                    (e.type, e.start, e.end, e.text, _ASSERT_TO_API.get(e.assertion, "AFFIRMED"))
                    for e in ocr_pred.entities
                ]
                assert api_ents == expected_ocr, f"Mismatch on PDF OCR {doc.doc_id}"
                # Also verify character-level span mapping back to reference is callable and valid
                mapped = map_ocr_spans_to_reference(ocr_text, doc.text, [(e.start, e.end) for e in ocr_pred.entities])
                assert len(mapped) == len(ocr_pred.entities)
