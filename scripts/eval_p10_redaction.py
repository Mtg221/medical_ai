"""Step 7 (W10): Measure P10 redaction recall & precision on a 52-sentence hand-labelled EN+FR benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medresearch_platform.security.privacy import deidentify_clinical_text

# 52 bilingual sentences: (text, expected_phi_substrings)
BENCHMARK_SENTENCES: list[tuple[str, list[str]]] = [
    ("Vu ce jour Mme Fatou Diallo, née le 12/03/1958, tél 77 123 45 67.", ["Fatou Diallo", "12/03/1958", "77 123 45 67"]),
    ("Dr Martin a revu Aminata Sow à Dakar.", ["Dr Martin", "Aminata Sow"]),
    ("Patient IPP: 884512, né le 04-11-1972, joint au +221 78 456 12 89.", ["884512", "04-11-1972", "+221 78 456 12 89"]),
    ("M. Moussa Ndiaye habite au 14 rue Victor Hugo, 75016 Paris.", ["Moussa Ndiaye", "14 rue Victor Hugo, 75016 Paris"]),
    ("Pr Diop signale dossier n° DX-99201, tél 33 821 44 55.", ["Pr Diop", "DX-99201", "33 821 44 55"]),
    ("Date de naissance: 19/08/1984, contact 06 12 34 56 78, patient Jean Mercier.", ["19/08/1984", "06 12 34 56 78", "Jean Mercier"]),
    ("Mlle Awa Fall, domiciliée au 22 avenue Pasteur, tél +33 6 98 76 54 32.", ["Awa Fall", "22 avenue Pasteur", "+33 6 98 76 54 32"]),
    ("Dr. Claire Moreau a examiné Mamadou Ba, IPP 445566.", ["Dr. Claire Moreau", "Mamadou Ba", "445566"]),
    ("Nom du patient: Ibrahima Gueye, né le 01/01/1965, email i.gueye@example.org.", ["Ibrahima Gueye", "01/01/1965", "i.gueye@example.org"]),
    ("Consultation de Ousmane Sarr, 8 boulevard Haussmann, tel 70 555 11 22.", ["Ousmane Sarr", "8 boulevard Haussmann", "70 555 11 22"]),
    ("Mme Khady Diagne, date de naissance 23/07/1990, tél 76 234 56 78.", ["Khady Diagne", "23/07/1990", "76 234 56 78"]),
    ("Docteur Bernard a suivi Paul Garnier, dossier n° FR-77889.", ["Docteur Bernard", "Paul Garnier", "FR-77889"]),
    ("M. Cheikh Faye réside au 5 impasse des Lilas, tél +221 77 889 00 11.", ["Cheikh Faye", "5 impasse des Lilas", "+221 77 889 00 11"]),
    ("Patiente Sophie Laurent, née le 30/05/1977, tél 01 42 68 55 00.", ["Sophie Laurent", "30/05/1977", "01 42 68 55 00"]),
    ("Pr. Chevallier a adressé Abdou Kane, IPP: SN-202601.", ["Pr. Chevallier", "Abdou Kane", "SN-202601"]),
    ("Mr. Robert Evans, DOB: 05/14/1962, phone +44 20 7946 0958.", ["Robert Evans", "05/14/1962", "+44 20 7946 0958"]),
    ("Dr. Wilson reviewed Linda Brooks at 42 Oak Street.", ["Dr. Wilson", "Linda Brooks", "42 Oak Street"]),
    ("Patient Name: Thomas White, MRN: 994411, phone +1 415 555 0199.", ["Thomas White", "994411", "+1 415 555 0199"]),
    ("Mrs. Sarah Miller lives at 105 Maple Avenue, phone +33 6 11 22 33 44.", ["Sarah Miller", "105 Maple Avenue", "+33 6 11 22 33 44"]),
    ("Date of birth: 11/22/1979, patient Daniel Reed, SSN 123-45-6789.", ["11/22/1979", "Daniel Reed", "123-45-6789"]),
    ("Ms. Hannah Cooper, record #REC-5544, tel 77 987 65 43.", ["Hannah Cooper", "REC-5544", "77 987 65 43"]),
    ("Prof. Morgan saw Kevin Scott, DOB 1988-09-12, email kscott@example.com.", ["Prof. Morgan", "Kevin Scott", "1988-09-12", "kscott@example.com"]),
    ("Olivia Ward resides at 18 King Road, MRN 772200.", ["Olivia Ward", "18 King Road", "772200"]),
    ("Dr Baker evaluated Laura Price, date of birth 03-08-1955.", ["Dr Baker", "Laura Price", "03-08-1955"]),
    ("Mr. Samuel Perry, 72 Elm Drive, phone +221 77 321 65 49.", ["Samuel Perry", "72 Elm Drive", "+221 77 321 65 49"]),
    ("Patient ID: PAT-99887, Mrs. Emily Hughes, DOB 07/04/1981.", ["PAT-99887", "Emily Hughes", "07/04/1981"]),
    ("Clinician Dr. Nguyen discharged George Taylor to 9 Pine Lane.", ["Dr. Nguyen", "George Taylor", "9 Pine Lane"]),
    ("Amina Diallo, DOB: 12/12/1993, phone +221 76 111 22 33.", ["Amina Diallo", "12/12/1993", "+221 76 111 22 33"]),
    ("Mr. Henry Adams, 305 Sunset Blvd, tel 06 55 44 33 22.", ["Henry Adams", "305 Sunset Blvd", "06 55 44 33 22"]),
    ("Prof. Patel consulted on Rachel Green, MRN: US-112233.", ["Prof. Patel", "Rachel Green", "US-112233"]),
    ("Mariama Touré, née le 15/06/1982, prend metformine 500 mg.", ["Mariama Touré", "15/06/1982"]),
    ("Dr. Leblanc a prescrit ramipril à Seydou Koné.", ["Dr. Leblanc", "Seydou Koné"]),
    ("Patient Charles Dupont, IPP 1122334, sans fièvre.", ["Charles Dupont", "1122334"]),
    ("Mrs. Grace Kelly, phone +1 650 253 0000, asthma flare.", ["Grace Kelly", "+1 650 253 0000"]),
    ("Mr. David Clark, DOB 09/25/1970, takes aspirin 81 mg.", ["David Clark", "09/25/1970"]),
    ("Mme Binta Cissé, 33 825 99 00, vit au 12 avenue Blaise Diagne.", ["Binta Cissé", "33 825 99 00", "12 avenue Blaise Diagne"]),
    ("Pr. Mbaye a revu Modou Fall au service de pneumologie.", ["Pr. Mbaye", "Modou Fall"]),
    ("Johnathan Vance, MRN 665544, ruled out pulmonary embolism.", ["Johnathan Vance", "665544"]),
    ("Dr. Carter noted Chloe Bennett has normal chest X-ray.", ["Dr. Carter", "Chloe Bennett"]),
    ("N° dossier: HPD-4012, patient Issa Diouf, né le 28/02/1969.", ["HPD-4012", "Issa Diouf", "28/02/1969"]),
    # 12 clinical sentences with NO PHI (must have 0 false positives)
    ("BPCO sous tiotropium 18 µg par jour sans dyspnée.", []),
    ("Chest CT rules out pulmonary embolism.", []),
    ("Patient presents with pneumonia and takes amoxicillin 1000 mg twice daily.", []),
    ("Aucun signe de thrombose veineuse profonde à l'échographie.", []),
    ("L'IRM cérébrale a éliminé un accident vasculaire cérébral.", []),
    ("Atrial fibrillation on apixaban 5 mg twice daily.", []),
    ("Hypertension artérielle équilibrée sous ramipril 5 mg.", []),
    ("No fever, no cough, normal oxygen saturation at rest.", []),
    ("Diabetes mellitus type 2 managed with metformin 850 mg.", []),
    ("Biopsie cutanée en faveur d'un mélanome superficiel.", []),
    ("Les D-dimères négatifs écartent une embolie pulmonaire.", []),
    ("Chronic kidney disease stage 3 with stable creatinine.", []),
]


def eval_p10_redaction(out_json: str | Path = "docs/results/p10_redaction_recall.json") -> dict[str, Any]:
    tp = fn = fp = 0
    missed: list[dict[str, str]] = []
    false_positives: list[dict[str, str]] = []

    for text, expected in BENCHMARK_SENTENCES:
        res = deidentify_clinical_text(text)
        for target in expected:
            if target not in res.scrubbed_text:
                tp += 1
            else:
                fn += 1
                missed.append({"text": text, "missed": target, "scrubbed": res.scrubbed_text})
        if not expected and res.detected_spans:
            fp += len(res.detected_spans)
            false_positives.append({"text": text, "spans": str([s.original_value for s in res.detected_spans])})

    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    report = {
        "n_sentences": len(BENCHMARK_SENTENCES),
        "n_phi_items": tp + fn,
        "tp": tp,
        "fn": fn,
        "fp_on_non_phi_sentences": fp,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "missed": missed,
        "false_positives": false_positives,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(eval_p10_redaction(), indent=2))
