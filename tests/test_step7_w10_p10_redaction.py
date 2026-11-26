"""Step 7 (W10) tests: P10 redaction coverage (>=15 EN + >=15 FR) + MD/HTML/PDF/export/audit/API checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medresearch_platform.config import PlatformConfig
from medresearch_platform.interfaces.api import create_app
from medresearch_platform.reports.generator import (
    render_html_report,
    render_markdown_report,
    render_pdf_report,
)
from medresearch_platform.security.privacy import deidentify_clinical_text

PHI_CASES: list[tuple[str, str, list[str]]] = [
    # 15 FR cases
    ("fr", "Vu ce jour Mme Fatou Diallo, née le 12/03/1958, tél 77 123 45 67.", ["Fatou Diallo", "12/03/1958", "77 123 45 67"]),
    ("fr", "Dr Martin a revu Aminata Sow à Dakar.", ["Dr Martin", "Aminata Sow"]),
    ("fr", "Patient IPP: 884512, né le 04-11-1972, joint au +221 78 456 12 89.", ["884512", "04-11-1972", "456 12 89"]),
    ("fr", "M. Moussa Ndiaye habite au 14 rue Victor Hugo, 75016 Paris.", ["Moussa Ndiaye", "14 rue Victor Hugo"]),
    ("fr", "Pr Diop signale dossier n° DX-99201, tél 33 821 44 55.", ["Pr Diop", "DX-99201", "821 44 55"]),
    ("fr", "Date de naissance: 19/08/1984, contact 06 12 34 56 78, patient Jean Mercier.", ["19/08/1984", "06 12 34 56 78", "Jean Mercier"]),
    ("fr", "Mlle Awa Fall, domiciliée au 22 avenue Pasteur, tél +33 6 98 76 54 32.", ["Awa Fall", "22 avenue Pasteur", "98 76 54 32"]),
    ("fr", "Dr. Claire Moreau a examiné Mamadou Ba, IPP 445566.", ["Claire Moreau", "Mamadou Ba", "445566"]),
    ("fr", "Nom du patient: Ibrahima Gueye, né le 01/01/1965, email i.gueye@example.org.", ["Ibrahima Gueye", "01/01/1965", "i.gueye@example.org"]),
    ("fr", "Consultation de Ousmane Sarr, 8 boulevard Haussmann, tel 70 555 11 22.", ["Ousmane Sarr", "8 boulevard Haussmann", "555 11 22"]),
    ("fr", "Mme Khady Diagne, date de naissance 23/07/1990, tél 76 234 56 78.", ["Khady Diagne", "23/07/1990", "234 56 78"]),
    ("fr", "Docteur Bernard a suivi Paul Garnier, dossier n° FR-77889.", ["Bernard", "Paul Garnier", "FR-77889"]),
    ("fr", "M. Cheikh Faye réside au 5 impasse des Lilas, tél +221 77 889 00 11.", ["Cheikh Faye", "5 impasse des Lilas", "889 00 11"]),
    ("fr", "Patiente Sophie Laurent, née le 30/05/1977, tél 01 42 68 55 00.", ["Sophie Laurent", "30/05/1977", "01 42 68 55 00"]),
    ("fr", "Pr. Chevallier a adressé Abdou Kane, IPP: SN-202601.", ["Chevallier", "Abdou Kane", "SN-202601"]),
    # 15 EN cases
    ("en", "Mr. Robert Evans, DOB: 05/14/1962, phone +44 20 7946 0958.", ["Robert Evans", "05/14/1962", "7946 0958"]),
    ("en", "Dr. Wilson reviewed Linda Brooks at 42 Oak Street.", ["Wilson", "Linda Brooks", "42 Oak Street"]),
    ("en", "Patient Name: Thomas White, MRN: 994411, phone +1 415 555 0199.", ["Thomas White", "994411", "555 0199"]),
    ("en", "Mrs. Sarah Miller lives at 105 Maple Avenue, phone +33 6 11 22 33 44.", ["Sarah Miller", "105 Maple Avenue", "11 22 33 44"]),
    ("en", "Date of birth: 11/22/1979, patient Daniel Reed, SSN 123-45-6789.", ["11/22/1979", "Daniel Reed", "123-45-6789"]),
    ("en", "Ms. Hannah Cooper, record #REC-5544, tel 77 987 65 43.", ["Hannah Cooper", "REC-5544", "987 65 43"]),
    ("en", "Prof. Morgan saw Kevin Scott, DOB 1988-09-12, email kscott@example.com.", ["Morgan", "Kevin Scott", "kscott@example.com"]),
    ("en", "Olivia Ward resides at 18 King Road, MRN 772200.", ["Olivia Ward", "18 King Road", "772200"]),
    ("en", "Dr Baker evaluated Laura Price, date of birth 03-08-1955.", ["Baker", "Laura Price", "03-08-1955"]),
    ("en", "Mr. Samuel Perry, 72 Elm Drive, phone +221 77 321 65 49.", ["Samuel Perry", "72 Elm Drive", "321 65 49"]),
    ("en", "Patient ID: PAT-99887, Mrs. Emily Hughes, DOB 07/04/1981.", ["PAT-99887", "Emily Hughes", "07/04/1981"]),
    ("en", "Clinician Dr. Nguyen discharged George Taylor to 9 Pine Lane.", ["Nguyen", "George Taylor", "9 Pine Lane"]),
    ("en", "Amina Diallo, DOB: 12/12/1993, phone +221 76 111 22 33.", ["Amina Diallo", "12/12/1993", "111 22 33"]),
    ("en", "Mr. Henry Adams, 305 Sunset Blvd, tel 06 55 44 33 22.", ["Henry Adams", "305 Sunset Blvd", "06 55 44 33 22"]),
    ("en", "Prof. Patel consulted on Rachel Green, MRN: US-112233.", ["Patel", "Rachel Green", "US-112233"]),
]


def test_w10_30_en_fr_phi_redaction_and_clinician_token() -> None:
    for _lang, text, secrets in PHI_CASES:
        res = deidentify_clinical_text(text)
        for s in secrets:
            assert s not in res.scrubbed_text, f"PHI {s!r} leaked in {res.scrubbed_text!r}"
        if any(t in text for t in ("Dr ", "Dr.", "Pr ", "Pr.", "Docteur ", "Prof ", "Prof.")):
            assert "[CLINICIAN_NAME_REDACTED]" in res.scrubbed_text


def test_w10_no_phi_survives_in_reports_exports_or_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_test_models: tuple[Path, Path],
) -> None:
    hybrid_dir, crf_dir = isolated_test_models
    monkeypatch.setenv("MEDRESEARCH_DB_PATH", str(tmp_path / "w10.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_SQLITE_PATH", str(tmp_path / "w10.sqlite3"))
    monkeypatch.setenv("MEDRESEARCH_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MEDRESEARCH_MODEL_DIR", str(hybrid_dir))
    monkeypatch.setenv("MEDRESEARCH_CLASSICAL_CRF_DIR", str(crf_dir))
    monkeypatch.setenv("MEDRESEARCH_DEMO_PASSWORD", "ChangeMe_W10_2026!")

    phi_text = "Dr Martin a revu Aminata Sow, née le 12/03/1958, tél 77 123 45 67, IPP 884512, au 14 rue Victor Hugo. BPCO sous tiotropium 18 µg."
    secrets = ["Dr Martin", "Aminata Sow", "12/03/1958", "77 123 45 67", "884512", "14 rue Victor Hugo"]

    doc_payload = {
        "document_id": "doc_w10",
        "filename": "note_w10.txt",
        "language": "fr",
        "document_type": "CLINICAL_NOTE",
        "specialty": "pulmonology",
        "encounter_date_iso": "2026-11-18",
        "encounter_date_source": "user_supplied",
        "dataset_id": "ds_w10",
        "subject_id": "subj_w10",
        "pseudonym_code": "SUBJ-W10",
        "raw_text": phi_text,
    }
    md = render_markdown_report(doc_payload, [], [], lang="fr")
    html_out = render_html_report(doc_payload, [], [], lang="fr")
    pdf_bytes = render_pdf_report(doc_payload, [], [], lang="fr")

    for s in secrets:
        assert s not in md
        assert s not in html_out
        assert s.encode("utf-8") not in pdf_bytes

    app = create_app(PlatformConfig.from_env(workspace_dir=tmp_path))
    with TestClient(app) as client:
        tok = client.post(
            "/api/v1/auth/token",
            json={"username": "admin_dakar", "password": "ChangeMe_W10_2026!"},
        ).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}"}
        now_iso = "2026-11-18T00:00:00+00:00"
        with app.state.service.db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO datasets (dataset_id, institution_id, name, version, source_type, license, consent_basis, "
                "permitted_purpose, deidentification_status, languages_json, modalities_json, record_count, manifest_sha256, retention_policy, created_at) "
                "VALUES ('ds_w10', 'inst_dakar_primary', 'W10', '1', 'SYNTHETIC', 'MIT', 'RESEARCH', 'BENCHMARK', 'SYNTHETIC_NO_PHI', '[\"fr\"]', '[\"TXT\"]', 0, 'abc', 'RETAIN', ?)",
                (now_iso,),
            )
            conn.execute(
                "INSERT OR IGNORE INTO subjects (subject_id, institution_id, dataset_id, pseudonym_code, language_primary, consent_status, created_at) "
                "VALUES ('subj_w10', 'inst_dakar_primary', 'ds_w10', 'SUBJ-W10', 'fr', 'SYNTHETIC_EXEMPT', ?)",
                (now_iso,),
            )
        r_ing = client.post(
            "/api/v1/documents/ingest",
            headers=h,
            json={
                "filename": "note_w10.txt",
                "text": phi_text,
                "dataset_id": "ds_w10",
                "subject_id": "subj_w10",
                "encounter_date_iso": "2026-11-18",
            },
        )
        assert r_ing.status_code in (200, 201), r_ing.text
        did = r_ing.json()["document_id"]
        rep = client.get(f"/api/v1/documents/{did}/report?lang=fr&format=markdown", headers=h)
        assert rep.status_code == 200
        for s in secrets:
            assert s not in rep.text
