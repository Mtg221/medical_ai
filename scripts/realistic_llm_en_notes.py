"""40 LLM-authored realistic English clinical notes (free-text, single-annotator, zero PHI) across 6 doc types."""

from __future__ import annotations

from typing import Any

# Each span entry is (exact_text, entity_type, assertion) or (exact_text, entity_type, assertion, occurrence_0_based)
EN_NOTES: list[dict[str, Any]] = [
    # --- DISCHARGE SUMMARIES (7) ---
    {
        "doc_id": "llm_en_01",
        "language": "en",
        "doc_type": "discharge_summary",
        "document_date": "2026-02-14",
        "text": (
            "DISCHARGE SUMMARY | ID: SIM-EN-001 | 2026-02-14\n"
            "68M h/o persistent atrial fibrillation and CKD stage 3 admitted early Feb w/ 4-day progressive orthopnea and bilateral leg swelling. "
            "NT-proBNP 4620 pg/mL; TTE showed LVEF 35% without pericardial effusion. "
            "Diuresed w/ furosemide 40 mg IV q12h x 3 days, transitioned to furosemide 40 mg PO daily plus spironolactone 25 mg PO once daily. "
            "Acute decompensated heart failure resolved; PE unlikely given normal D-dimer. Return immediately if chest pain or syncope develops."
        ),
        "spans": [
            ("SIM-EN-001", "PATIENT_ID", "present"),
            ("2026-02-14", "DATE", "present"),
            ("persistent atrial fibrillation", "DIAGNOSIS", "historical"),
            ("CKD stage 3", "DIAGNOSIS", "historical"),
            ("early Feb", "DATE", "present"),
            ("4-day", "DURATION", "present"),
            ("orthopnea", "SYMPTOM", "present"),
            ("bilateral leg swelling", "SYMPTOM", "present"),
            ("NT-proBNP", "LAB_TEST", "present"),
            ("4620", "LAB_VALUE", "present"),
            ("pg/mL", "LAB_UNIT", "present"),
            ("TTE", "PROCEDURE", "present"),
            ("LVEF", "LAB_TEST", "present"),
            ("35", "LAB_VALUE", "present"),
            ("%", "LAB_UNIT", "present"),
            ("pericardial effusion", "DIAGNOSIS", "absent"),
            ("furosemide", "MEDICATION", "present", 0),
            ("40 mg", "DOSAGE", "present", 0),
            ("IV", "ROUTE", "present"),
            ("q12h", "FREQUENCY", "present"),
            ("3 days", "DURATION", "present"),
            ("furosemide", "MEDICATION", "present", 1),
            ("40 mg", "DOSAGE", "present", 1),
            ("PO", "ROUTE", "present", 0),
            ("daily", "FREQUENCY", "present", 0),
            ("spironolactone", "MEDICATION", "present"),
            ("25 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("once daily", "FREQUENCY", "present"),
            ("Acute decompensated heart failure", "DIAGNOSIS", "present"),
            ("PE", "DIAGNOSIS", "absent"),
            ("D-dimer", "LAB_TEST", "present"),
            ("chest pain", "SYMPTOM", "hypothetical"),
            ("syncope", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_en_02",
        "language": "en",
        "doc_type": "discharge_summary",
        "document_date": "2026-01-22",
        "text": (
            "Hospital Course — Chart SIM-EN-002 (22-Jan-2026)\n"
            "Admitted from ED with severe right lower quadrant abdominal pain, low-grade fever, and anorexia since 2 days ago. "
            "WBC peaked at 15.8 x10^9/L and CRP 94 mg/L. Laparoscopic appendectomy performed without intraoperative complication; pathology confirmed acute suppurative appendicitis. "
            "Received ceftriaxone 1 g IV once daily and metronidazole 500 mg IV q8h for 48 hours. Tolerating regular diet, afebrile, discharged home; seek urgent care should wound erythema worsen."
        ),
        "spans": [
            ("SIM-EN-002", "PATIENT_ID", "present"),
            ("22-Jan-2026", "DATE", "present"),
            ("right lower quadrant", "ANATOMY", "present"),
            ("abdominal pain", "SYMPTOM", "present"),
            ("fever", "SYMPTOM", "present"),
            ("anorexia", "SYMPTOM", "present"),
            ("2 days ago", "DATE", "present"),
            ("WBC", "LAB_TEST", "present"),
            ("15.8", "LAB_VALUE", "present"),
            ("x10^9/L", "LAB_UNIT", "present"),
            ("CRP", "LAB_TEST", "present"),
            ("94", "LAB_VALUE", "present"),
            ("mg/L", "LAB_UNIT", "present"),
            ("Laparoscopic appendectomy", "PROCEDURE", "present"),
            ("acute suppurative appendicitis", "DIAGNOSIS", "present"),
            ("ceftriaxone", "MEDICATION", "present"),
            ("1 g", "DOSAGE", "present"),
            ("IV", "ROUTE", "present", 0),
            ("once daily", "FREQUENCY", "present"),
            ("metronidazole", "MEDICATION", "present"),
            ("500 mg", "DOSAGE", "present"),
            ("IV", "ROUTE", "present", 1),
            ("q8h", "FREQUENCY", "present"),
            ("for 48 hours", "DURATION", "present"),
            ("wound erythema", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_en_03",
        "language": "en",
        "doc_type": "discharge_summary",
        "document_date": "2025-11-30",
        "text": (
            "Inpatient Discharge Note [SIM-EN-003] Date: Nov 30, 2025\n"
            "54F w/ history of asthma and maternal breast cancer hospitalized 5 days for community-acquired pneumonia involving the left lower lobe. "
            "Initial sputum culture negative for MRSA; blood cultures sterile. Treated with amoxicillin-clavulanate 875 mg PO BID for 7 days and azithromycin 500 mg PO daily x 3d. "
            "Hypoxemia resolved (SpO2 97% on room air). Possible underlying bronchiectasis to be re-evaluated by high-resolution chest CT in 6 weeks."
        ),
        "spans": [
            ("SIM-EN-003", "PATIENT_ID", "present"),
            ("Nov 30, 2025", "DATE", "present"),
            ("asthma", "DIAGNOSIS", "historical"),
            ("breast cancer", "DIAGNOSIS", "family"),
            ("5 days", "DURATION", "present"),
            ("community-acquired pneumonia", "DIAGNOSIS", "present"),
            ("left lower lobe", "ANATOMY", "present"),
            ("sputum culture", "LAB_TEST", "present"),
            ("MRSA", "DIAGNOSIS", "absent"),
            ("amoxicillin-clavulanate", "MEDICATION", "present"),
            ("875 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("BID", "FREQUENCY", "present"),
            ("for 7 days", "DURATION", "present"),
            ("azithromycin", "MEDICATION", "present"),
            ("500 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("daily", "FREQUENCY", "present"),
            ("3d", "DURATION", "present"),
            ("Hypoxemia", "SYMPTOM", "present"),
            ("SpO2", "LAB_TEST", "present"),
            ("97", "LAB_VALUE", "present"),
            ("%", "LAB_UNIT", "present"),
            ("bronchiectasis", "DIAGNOSIS", "possible"),
            ("high-resolution chest CT", "PROCEDURE", "present"),
            ("6 weeks", "DURATION", "present"),
        ],
    },
    {
        "doc_id": "llm_en_04",
        "language": "en",
        "doc_type": "discharge_summary",
        "document_date": "2026-03-04",
        "text": (
            "ICU Stepdown Summary — Patient SIM-EN-004 — 2026-03-04\n"
            "71M s/p NSTEMI in 2021 brought in after sudden-onset slurred speech and left arm weakness starting 90 minutes prior to arrival. "
            "Non-contrast head CT excluded intracranial hemorrhage; CT angiography revealed right M1 occlusion. "
            "Mechanical thrombectomy achieved TICI 3 recanalization. Discharged on aspirin 81 mg PO daily, clopidogrel 75 mg PO daily for 21 days, and atorvastatin 80 mg PO nightly. "
            "No residual dysarthria; father had ischemic stroke at age 66."
        ),
        "spans": [
            ("SIM-EN-004", "PATIENT_ID", "present"),
            ("2026-03-04", "DATE", "present"),
            ("NSTEMI", "DIAGNOSIS", "historical"),
            ("2021", "DATE", "present"),
            ("slurred speech", "SYMPTOM", "present"),
            ("left arm", "ANATOMY", "present"),
            ("weakness", "SYMPTOM", "present"),
            ("90 minutes", "DURATION", "present"),
            ("Non-contrast head CT", "PROCEDURE", "present"),
            ("intracranial hemorrhage", "DIAGNOSIS", "absent"),
            ("CT angiography", "PROCEDURE", "present"),
            ("right M1 occlusion", "DIAGNOSIS", "present"),
            ("Mechanical thrombectomy", "PROCEDURE", "present"),
            ("aspirin", "MEDICATION", "present"),
            ("81 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("daily", "FREQUENCY", "present", 0),
            ("clopidogrel", "MEDICATION", "present"),
            ("75 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("daily", "FREQUENCY", "present", 1),
            ("for 21 days", "DURATION", "present"),
            ("atorvastatin", "MEDICATION", "present"),
            ("80 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 2),
            ("nightly", "FREQUENCY", "present"),
            ("dysarthria", "SYMPTOM", "absent"),
            ("ischemic stroke", "DIAGNOSIS", "family"),
        ],
    },
    {
        "doc_id": "llm_en_05",
        "language": "en",
        "doc_type": "discharge_summary",
        "document_date": "2026-04-11",
        "text": (
            "Discharge Brief | SIM-EN-005 | 11 Apr 2026\n"
            "29F T1DM admitted in diabetic ketoacidosis triggered by missed basal doses during viral gastroenteritis. "
            "Initial venous pH 7.14, beta-hydroxybutyrate 6.1 mmol/L, serum potassium 5.3 mmol/L. "
            "Managed on insulin regular IV infusion until anion gap closed, then resumed insulin glargine 22 units SC nightly and insulin lispro SC before meals. "
            "No evidence of urinary tract infection or sepsis. Educate on sick-day protocol in case recurrent vomiting occurs."
        ),
        "spans": [
            ("SIM-EN-005", "PATIENT_ID", "present"),
            ("11 Apr 2026", "DATE", "present"),
            ("T1DM", "DIAGNOSIS", "historical"),
            ("diabetic ketoacidosis", "DIAGNOSIS", "present"),
            ("viral gastroenteritis", "DIAGNOSIS", "present"),
            ("venous pH", "LAB_TEST", "present"),
            ("7.14", "LAB_VALUE", "present"),
            ("beta-hydroxybutyrate", "LAB_TEST", "present"),
            ("6.1", "LAB_VALUE", "present"),
            ("mmol/L", "LAB_UNIT", "present", 0),
            ("serum potassium", "LAB_TEST", "present"),
            ("5.3", "LAB_VALUE", "present"),
            ("mmol/L", "LAB_UNIT", "present", 1),
            ("insulin regular", "MEDICATION", "present"),
            ("IV", "ROUTE", "present"),
            ("insulin glargine", "MEDICATION", "present"),
            ("22 units", "DOSAGE", "present"),
            ("SC", "ROUTE", "present", 0),
            ("nightly", "FREQUENCY", "present"),
            ("insulin lispro", "MEDICATION", "present"),
            ("SC", "ROUTE", "present", 1),
            ("before meals", "FREQUENCY", "present"),
            ("urinary tract infection", "DIAGNOSIS", "absent"),
            ("sepsis", "DIAGNOSIS", "absent"),
            ("vomiting", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_en_06",
        "language": "en",
        "doc_type": "discharge_summary",
        "document_date": "2025-10-19",
        "text": (
            "Ward Discharge — Patient SIM-EN-006 (Oct 19, 2025)\n"
            "62M cirrhosis secondary to alcohol use admitted for two episodes of hematemesis. Hemoglobin dropped to 7.8 g/dL. "
            "Urgent upper endoscopy identified bleeding esophageal varices treated via endoscopic band ligation. "
            "Given octreotide IV drip x 72h and ceftriaxone 1 g IV daily for 5 days; started carvedilol 6.25 mg PO BID at discharge. "
            "Hepatic encephalopathy absent throughout stay; possible portal vein thrombosis vs slow flow to clarify on repeat Doppler ultrasound."
        ),
        "spans": [
            ("SIM-EN-006", "PATIENT_ID", "present"),
            ("Oct 19, 2025", "DATE", "present"),
            ("cirrhosis", "DIAGNOSIS", "historical"),
            ("hematemesis", "SYMPTOM", "present"),
            ("Hemoglobin", "LAB_TEST", "present"),
            ("7.8", "LAB_VALUE", "present"),
            ("g/dL", "LAB_UNIT", "present"),
            ("upper endoscopy", "PROCEDURE", "present"),
            ("esophageal varices", "DIAGNOSIS", "present"),
            ("endoscopic band ligation", "PROCEDURE", "present"),
            ("octreotide", "MEDICATION", "present"),
            ("IV", "ROUTE", "present", 0),
            ("72h", "DURATION", "present"),
            ("ceftriaxone", "MEDICATION", "present"),
            ("1 g", "DOSAGE", "present"),
            ("IV", "ROUTE", "present", 1),
            ("daily", "FREQUENCY", "present"),
            ("for 5 days", "DURATION", "present"),
            ("carvedilol", "MEDICATION", "present"),
            ("6.25 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present"),
            ("BID", "FREQUENCY", "present"),
            ("Hepatic encephalopathy", "DIAGNOSIS", "absent"),
            ("portal vein thrombosis", "DIAGNOSIS", "possible"),
            ("Doppler ultrasound", "PROCEDURE", "present"),
        ],
    },
    {
        "doc_id": "llm_en_07",
        "language": "en",
        "doc_type": "discharge_summary",
        "document_date": "2026-05-02",
        "text": (
            "Pulmonary Discharge Note | SIM-EN-007 | 2026-05-02\n"
            "74M ex-smoker with GOLD III COPD presented w/ 1 week of purulent sputum and worsening wheezing. "
            "ABG on arrival: PaCO2 58 mmHg, PaO2 54 mmHg. Improved with noninvasive ventilation, prednisone 40 mg PO daily for 5 days, doxycycline 100 mg PO q12h x 5 days, and tiotropium 18 mcg inhaled once daily. "
            "No pneumothorax on chest radiograph. Advise prompt review if resting dyspnea worsens."
        ),
        "spans": [
            ("SIM-EN-007", "PATIENT_ID", "present"),
            ("2026-05-02", "DATE", "present"),
            ("COPD", "DIAGNOSIS", "historical"),
            ("1 week", "DURATION", "present"),
            ("purulent sputum", "SYMPTOM", "present"),
            ("wheezing", "SYMPTOM", "present"),
            ("PaCO2", "LAB_TEST", "present"),
            ("58", "LAB_VALUE", "present"),
            ("mmHg", "LAB_UNIT", "present", 0),
            ("PaO2", "LAB_TEST", "present"),
            ("54", "LAB_VALUE", "present"),
            ("mmHg", "LAB_UNIT", "present", 1),
            ("noninvasive ventilation", "PROCEDURE", "present"),
            ("prednisone", "MEDICATION", "present"),
            ("40 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("daily", "FREQUENCY", "present", 0),
            ("for 5 days", "DURATION", "present"),
            ("doxycycline", "MEDICATION", "present"),
            ("100 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("q12h", "FREQUENCY", "present"),
            ("5 days", "DURATION", "present", 1),
            ("tiotropium", "MEDICATION", "present"),
            ("18 mcg", "DOSAGE", "present"),
            ("inhaled", "ROUTE", "present"),
            ("once daily", "FREQUENCY", "present"),
            ("pneumothorax", "DIAGNOSIS", "absent"),
            ("chest radiograph", "PROCEDURE", "present"),
            ("dyspnea", "SYMPTOM", "hypothetical"),
        ],
    },
    # --- CONSULTATIONS (7) ---
    {
        "doc_id": "llm_en_08",
        "language": "en",
        "doc_type": "consultation",
        "document_date": "2026-01-15",
        "text": (
            "Cardiology Outpatient Consult — SIM-EN-008 — mid-January 2026\n"
            "59F referred for intermittent palpitations and exertional lightheadedness since early Oct. "
            "Denies angina, syncope, or ankle edema. Family history notable for hypertrophic cardiomyopathy in brother. "
            "12-lead ECG shows sinus rhythm without ST depression. Holter monitor ordered for 14 days to evaluate possible paroxysmal supraventricular tachycardia; hold metoprolol 25 mg PO BID unless symptomatic tachycardia recurs."
        ),
        "spans": [
            ("SIM-EN-008", "PATIENT_ID", "present"),
            ("mid-January 2026", "DATE", "present"),
            ("palpitations", "SYMPTOM", "present"),
            ("lightheadedness", "SYMPTOM", "present"),
            ("early Oct", "DATE", "present"),
            ("angina", "SYMPTOM", "absent"),
            ("syncope", "SYMPTOM", "absent"),
            ("ankle edema", "SYMPTOM", "absent"),
            ("hypertrophic cardiomyopathy", "DIAGNOSIS", "family"),
            ("12-lead ECG", "PROCEDURE", "present"),
            ("Holter monitor", "PROCEDURE", "present"),
            ("for 14 days", "DURATION", "present"),
            ("paroxysmal supraventricular tachycardia", "DIAGNOSIS", "possible"),
            ("metoprolol", "MEDICATION", "hypothetical"),
            ("25 mg", "DOSAGE", "hypothetical"),
            ("PO", "ROUTE", "hypothetical"),
            ("BID", "FREQUENCY", "hypothetical"),
            ("tachycardia", "SYMPTOM", "hypothetical", 1),
        ],
    },
    {
        "doc_id": "llm_en_09",
        "language": "en",
        "doc_type": "consultation",
        "document_date": "2026-02-27",
        "text": (
            "Rheumatology Clinic Note | ID SIM-EN-009 | 2026-02-27\n"
            "43F c/o symmetric morning stiffness in both wrists and MCP joints lasting > 1 hour for 4 months, accompanied by fatigue. "
            "Anti-CCP 142 U/mL, ESR 48 mm/hr. No psoriasis or uveitis. "
            "Likely seropositive rheumatoid arthritis vs early lupus; initiated methotrexate 15 mg PO once weekly with folic acid 5 mg PO weekly. Screen for latent tuberculosis prior to any biologic escalation."
        ),
        "spans": [
            ("SIM-EN-009", "PATIENT_ID", "present"),
            ("2026-02-27", "DATE", "present"),
            ("morning stiffness", "SYMPTOM", "present"),
            ("wrists", "ANATOMY", "present"),
            ("MCP joints", "ANATOMY", "present"),
            ("1 hour", "DURATION", "present"),
            ("for 4 months", "DURATION", "present"),
            ("fatigue", "SYMPTOM", "present"),
            ("Anti-CCP", "LAB_TEST", "present"),
            ("142", "LAB_VALUE", "present"),
            ("U/mL", "LAB_UNIT", "present"),
            ("ESR", "LAB_TEST", "present"),
            ("48", "LAB_VALUE", "present"),
            ("mm/hr", "LAB_UNIT", "present"),
            ("psoriasis", "DIAGNOSIS", "absent"),
            ("uveitis", "DIAGNOSIS", "absent"),
            ("seropositive rheumatoid arthritis", "DIAGNOSIS", "possible"),
            ("lupus", "DIAGNOSIS", "possible"),
            ("methotrexate", "MEDICATION", "present"),
            ("15 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("once weekly", "FREQUENCY", "present"),
            ("folic acid", "MEDICATION", "present"),
            ("5 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("weekly", "FREQUENCY", "present", 1),
            ("latent tuberculosis", "DIAGNOSIS", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_en_10",
        "language": "en",
        "doc_type": "consultation",
        "document_date": "2026-03-18",
        "text": (
            "Neurology Consult — SIM-EN-010 (03/18/2026)\n"
            "36F h/o episodic migraine presents w/ throbbing unilateral headache, photophobia, and nausea 6 days per month. "
            "Neuro exam intact; no papilledema or focal weakness, making intracranial mass unlikely. "
            "Started sumatriptan 50 mg PO PRN for acute attacks and propranolol 40 mg PO BID as prophylaxis for 3 months. If aura or diplopia appears, obtain brain MRI."
        ),
        "spans": [
            ("SIM-EN-010", "PATIENT_ID", "present"),
            ("03/18/2026", "DATE", "present"),
            ("migraine", "DIAGNOSIS", "historical"),
            ("headache", "SYMPTOM", "present"),
            ("photophobia", "SYMPTOM", "present"),
            ("nausea", "SYMPTOM", "present"),
            ("papilledema", "SYMPTOM", "absent"),
            ("focal weakness", "SYMPTOM", "absent"),
            ("intracranial mass", "DIAGNOSIS", "absent"),
            ("sumatriptan", "MEDICATION", "present"),
            ("50 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("PRN", "FREQUENCY", "present"),
            ("propranolol", "MEDICATION", "present"),
            ("40 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("BID", "FREQUENCY", "present"),
            ("for 3 months", "DURATION", "present"),
            ("aura", "SYMPTOM", "hypothetical"),
            ("diplopia", "SYMPTOM", "hypothetical"),
            ("brain MRI", "PROCEDURE", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_en_11",
        "language": "en",
        "doc_type": "consultation",
        "document_date": "2025-12-09",
        "text": (
            "Endocrinology Follow-Up | SIM-EN-011 | Dec 9, 2025\n"
            "51M w/ type 2 diabetes mellitus and hypertension; mother had hypothyroidism. "
            "Reports occasional postprandial bloating on metformin 1000 mg PO BID, no hypoglycemia. "
            "HbA1c improved from 8.9% to 7.1%; TSH 2.4 mIU/L rules out thyroid dysfunction. Added empagliflozin 10 mg PO once daily; continue lisinopril 20 mg PO daily."
        ),
        "spans": [
            ("SIM-EN-011", "PATIENT_ID", "present"),
            ("Dec 9, 2025", "DATE", "present"),
            ("type 2 diabetes mellitus", "DIAGNOSIS", "present"),
            ("hypertension", "DIAGNOSIS", "present"),
            ("hypothyroidism", "DIAGNOSIS", "family"),
            ("bloating", "SYMPTOM", "present"),
            ("metformin", "MEDICATION", "present"),
            ("1000 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("BID", "FREQUENCY", "present"),
            ("hypoglycemia", "DIAGNOSIS", "absent"),
            ("HbA1c", "LAB_TEST", "present"),
            ("7.1", "LAB_VALUE", "present"),
            ("%", "LAB_UNIT", "present", 1),
            ("TSH", "LAB_TEST", "present"),
            ("2.4", "LAB_VALUE", "present"),
            ("mIU/L", "LAB_UNIT", "present"),
            ("thyroid dysfunction", "DIAGNOSIS", "absent"),
            ("empagliflozin", "MEDICATION", "present"),
            ("10 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("once daily", "FREQUENCY", "present"),
            ("lisinopril", "MEDICATION", "present"),
            ("20 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 2),
            ("daily", "FREQUENCY", "present", 1),
        ],
    },
    {
        "doc_id": "llm_en_12",
        "language": "en",
        "doc_type": "consultation",
        "document_date": "2026-04-03",
        "text": (
            "Gastroenterology Consult — Patient SIM-EN-012 — 2026-04-03\n"
            "47M c/o 6 weeks of epigastric burning worse at night and mild dysphagia to solids; denies melena or unintentional weight loss. "
            "History of Helicobacter pylori gastritis treated in 2019. "
            "Trial of omeprazole 40 mg PO once daily for 8 weeks; schedule diagnostic gastroscopy to rule out peptic ulcer disease or Barrett esophagus."
        ),
        "spans": [
            ("SIM-EN-012", "PATIENT_ID", "present"),
            ("2026-04-03", "DATE", "present"),
            ("6 weeks", "DURATION", "present"),
            ("epigastric burning", "SYMPTOM", "present"),
            ("dysphagia", "SYMPTOM", "present"),
            ("melena", "SYMPTOM", "absent"),
            ("weight loss", "SYMPTOM", "absent"),
            ("Helicobacter pylori gastritis", "DIAGNOSIS", "historical"),
            ("2019", "DATE", "present"),
            ("omeprazole", "MEDICATION", "present"),
            ("40 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present"),
            ("once daily", "FREQUENCY", "present"),
            ("for 8 weeks", "DURATION", "present"),
            ("gastroscopy", "PROCEDURE", "present"),
            ("peptic ulcer disease", "DIAGNOSIS", "possible"),
            ("Barrett esophagus", "DIAGNOSIS", "possible"),
        ],
    },
    {
        "doc_id": "llm_en_13",
        "language": "en",
        "doc_type": "consultation",
        "document_date": "2026-03-29",
        "text": (
            "Infectious Disease Eval | SIM-EN-013 | 29 Mar 2026\n"
            "31M returned 10 days ago from coastal West Africa w/ high fever, chills, and myalgias. "
            "Rapid malaria antigen test positive for Plasmodium falciparum; parasitemia 1.4%. No jaundice or altered mental status; dengue unlikely given negative NS1. "
            "Initiated artemether-lumefantrine PO BID for 3 days; admit immediately if persistent vomiting prevents oral intake."
        ),
        "spans": [
            ("SIM-EN-013", "PATIENT_ID", "present"),
            ("29 Mar 2026", "DATE", "present"),
            ("10 days ago", "DATE", "present"),
            ("fever", "SYMPTOM", "present"),
            ("chills", "SYMPTOM", "present"),
            ("myalgias", "SYMPTOM", "present"),
            ("Rapid malaria antigen test", "LAB_TEST", "present"),
            ("Plasmodium falciparum", "DIAGNOSIS", "present"),
            ("parasitemia", "LAB_TEST", "present"),
            ("1.4", "LAB_VALUE", "present"),
            ("%", "LAB_UNIT", "present"),
            ("jaundice", "SYMPTOM", "absent"),
            ("altered mental status", "SYMPTOM", "absent"),
            ("dengue", "DIAGNOSIS", "absent"),
            ("artemether-lumefantrine", "MEDICATION", "present"),
            ("PO", "ROUTE", "present"),
            ("BID", "FREQUENCY", "present"),
            ("for 3 days", "DURATION", "present"),
            ("vomiting", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_en_14",
        "language": "en",
        "doc_type": "consultation",
        "document_date": "2026-05-08",
        "text": (
            "Nephrology Clinic — SIM-EN-014 — May 8, 2026\n"
            "66F s/p right nephrectomy in 2018 seen for gradual rise in serum creatinine to 176 umol/L (eGFR 28 mL/min). "
            "Urine albumin-to-creatinine ratio 410 mg/g. Denies hematuria, flank pain, or NSAID use. "
            "Suspected diabetic nephropathy vs hypertensive nephrosclerosis; continue losartan 50 mg PO daily and check potassium in 2 weeks."
        ),
        "spans": [
            ("SIM-EN-014", "PATIENT_ID", "present"),
            ("May 8, 2026", "DATE", "present"),
            ("right nephrectomy", "PROCEDURE", "historical"),
            ("2018", "DATE", "present"),
            ("serum creatinine", "LAB_TEST", "present"),
            ("176", "LAB_VALUE", "present"),
            ("umol/L", "LAB_UNIT", "present"),
            ("eGFR", "LAB_TEST", "present"),
            ("28", "LAB_VALUE", "present"),
            ("mL/min", "LAB_UNIT", "present"),
            ("Urine albumin-to-creatinine ratio", "LAB_TEST", "present"),
            ("410", "LAB_VALUE", "present"),
            ("mg/g", "LAB_UNIT", "present"),
            ("hematuria", "SYMPTOM", "absent"),
            ("flank pain", "SYMPTOM", "absent"),
            ("diabetic nephropathy", "DIAGNOSIS", "possible"),
            ("hypertensive nephrosclerosis", "DIAGNOSIS", "possible"),
            ("losartan", "MEDICATION", "present"),
            ("50 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present"),
            ("daily", "FREQUENCY", "present"),
            ("potassium", "LAB_TEST", "present"),
            ("2 weeks", "DURATION", "present"),
        ],
    },
    # --- LAB REPORTS (7) ---
    {
        "doc_id": "llm_en_15",
        "language": "en",
        "doc_type": "lab_report",
        "document_date": "2026-01-19",
        "text": (
            "CLINICAL CHEMISTRY REPORT — ID: SIM-EN-015 — Collected: 2026-01-19\n"
            "Fasting glucose: 164 mg/dL (H)\n"
            "HbA1c: 8.2 %\n"
            "LDL cholesterol: 158 mg/dL\n"
            "Triglycerides: 245 mg/dL\n"
            "ALT: 31 U/L\n"
            "Comment: Findings consistent with poorly controlled diabetes mellitus and hyperlipidemia; no biochemical evidence of acute hepatitis."
        ),
        "spans": [
            ("SIM-EN-015", "PATIENT_ID", "present"),
            ("2026-01-19", "DATE", "present"),
            ("Fasting glucose", "LAB_TEST", "present"),
            ("164", "LAB_VALUE", "present"),
            ("mg/dL", "LAB_UNIT", "present", 0),
            ("HbA1c", "LAB_TEST", "present"),
            ("8.2", "LAB_VALUE", "present"),
            ("%", "LAB_UNIT", "present"),
            ("LDL cholesterol", "LAB_TEST", "present"),
            ("158", "LAB_VALUE", "present"),
            ("mg/dL", "LAB_UNIT", "present", 1),
            ("Triglycerides", "LAB_TEST", "present"),
            ("245", "LAB_VALUE", "present"),
            ("mg/dL", "LAB_UNIT", "present", 2),
            ("ALT", "LAB_TEST", "present"),
            ("31", "LAB_VALUE", "present"),
            ("U/L", "LAB_UNIT", "present"),
            ("diabetes mellitus", "DIAGNOSIS", "present"),
            ("hyperlipidemia", "DIAGNOSIS", "present"),
            ("acute hepatitis", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_en_16",
        "language": "en",
        "doc_type": "lab_report",
        "document_date": "2026-02-05",
        "text": (
            "HEMATOLOGY PANEL | Patient SIM-EN-016 | 05/02/2026\n"
            "Hemoglobin: 8.9 g/dL | MCV: 71 fL | Platelets: 435 x10^9/L\n"
            "Serum ferritin: 7 ng/mL | Transferrin saturation: 6 %\n"
            "Peripheral smear: microcytic hypochromic erythrocytes without blasts. Suggests iron deficiency anemia; acute leukemia excluded."
        ),
        "spans": [
            ("SIM-EN-016", "PATIENT_ID", "present"),
            ("05/02/2026", "DATE", "present"),
            ("Hemoglobin", "LAB_TEST", "present"),
            ("8.9", "LAB_VALUE", "present"),
            ("g/dL", "LAB_UNIT", "present"),
            ("MCV", "LAB_TEST", "present"),
            ("71", "LAB_VALUE", "present"),
            ("fL", "LAB_UNIT", "present"),
            ("Platelets", "LAB_TEST", "present"),
            ("435", "LAB_VALUE", "present"),
            ("x10^9/L", "LAB_UNIT", "present"),
            ("Serum ferritin", "LAB_TEST", "present"),
            ("7", "LAB_VALUE", "present"),
            ("ng/mL", "LAB_UNIT", "present"),
            ("Transferrin saturation", "LAB_TEST", "present"),
            ("6", "LAB_VALUE", "present"),
            ("%", "LAB_UNIT", "present"),
            ("iron deficiency anemia", "DIAGNOSIS", "possible"),
            ("acute leukemia", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_en_17",
        "language": "en",
        "doc_type": "lab_report",
        "document_date": "2026-03-11",
        "text": (
            "CARDIAC BIOMARKERS — SIM-EN-017 — 2026-03-11\n"
            "High-sensitivity troponin I (0h): 14 ng/L\n"
            "High-sensitivity troponin I (3h): 198 ng/L (CRITICAL)\n"
            "CK-MB: 26 ng/mL | Serum creatinine: 92 umol/L\n"
            "Interpretation: Rising troponin trajectory compatible with acute myocardial injury; correlate clinically for NSTEMI vs myocarditis."
        ),
        "spans": [
            ("SIM-EN-017", "PATIENT_ID", "present"),
            ("2026-03-11", "DATE", "present"),
            ("High-sensitivity troponin I", "LAB_TEST", "present", 0),
            ("14", "LAB_VALUE", "present"),
            ("ng/L", "LAB_UNIT", "present", 0),
            ("High-sensitivity troponin I", "LAB_TEST", "present", 1),
            ("198", "LAB_VALUE", "present"),
            ("ng/L", "LAB_UNIT", "present", 1),
            ("CK-MB", "LAB_TEST", "present"),
            ("26", "LAB_VALUE", "present"),
            ("ng/mL", "LAB_UNIT", "present"),
            ("Serum creatinine", "LAB_TEST", "present"),
            ("92", "LAB_VALUE", "present"),
            ("umol/L", "LAB_UNIT", "present"),
            ("acute myocardial injury", "DIAGNOSIS", "present"),
            ("NSTEMI", "DIAGNOSIS", "possible"),
            ("myocarditis", "DIAGNOSIS", "possible"),
        ],
    },
    {
        "doc_id": "llm_en_18",
        "language": "en",
        "doc_type": "lab_report",
        "document_date": "2026-04-01",
        "text": (
            "MICROBIOLOGY & SEPSIS LABS | SIM-EN-018 | Apr 1, 2026\n"
            "Procalcitonin: 8.7 ng/mL | Serum lactate: 3.6 mmol/L | CRP: 215 mg/L\n"
            "Urine culture: >100,000 CFU/mL Escherichia coli sensitive to meropenem and ceftriaxone.\n"
            "Blood culture (2/2 bottles): positive for E. coli, consistent with urosepsis; no fungal growth."
        ),
        "spans": [
            ("SIM-EN-018", "PATIENT_ID", "present"),
            ("Apr 1, 2026", "DATE", "present"),
            ("Procalcitonin", "LAB_TEST", "present"),
            ("8.7", "LAB_VALUE", "present"),
            ("ng/mL", "LAB_UNIT", "present"),
            ("Serum lactate", "LAB_TEST", "present"),
            ("3.6", "LAB_VALUE", "present"),
            ("mmol/L", "LAB_UNIT", "present"),
            ("CRP", "LAB_TEST", "present"),
            ("215", "LAB_VALUE", "present"),
            ("mg/L", "LAB_UNIT", "present"),
            ("Urine culture", "LAB_TEST", "present"),
            ("meropenem", "MEDICATION", "present"),
            ("ceftriaxone", "MEDICATION", "present"),
            ("Blood culture", "LAB_TEST", "present"),
            ("urosepsis", "DIAGNOSIS", "present"),
        ],
    },
    {
        "doc_id": "llm_en_19",
        "language": "en",
        "doc_type": "lab_report",
        "document_date": "2026-02-20",
        "text": (
            "THYROID & AUTOIMMUNE ASSAY — Chart SIM-EN-019 — 20-Feb-2026\n"
            "TSH: 0.02 mIU/L (L) | Free T4: 41.5 pmol/L (H) | Free T3: 14.2 pmol/L (H)\n"
            "TRAb: 9.8 IU/L (Positive).\n"
            "Biochemical overt hyperthyroidism strongly favoring Graves disease; subacute thyroiditis unlikely."
        ),
        "spans": [
            ("SIM-EN-019", "PATIENT_ID", "present"),
            ("20-Feb-2026", "DATE", "present"),
            ("TSH", "LAB_TEST", "present"),
            ("0.02", "LAB_VALUE", "present"),
            ("mIU/L", "LAB_UNIT", "present"),
            ("Free T4", "LAB_TEST", "present"),
            ("41.5", "LAB_VALUE", "present"),
            ("pmol/L", "LAB_UNIT", "present", 0),
            ("Free T3", "LAB_TEST", "present"),
            ("14.2", "LAB_VALUE", "present"),
            ("pmol/L", "LAB_UNIT", "present", 1),
            ("TRAb", "LAB_TEST", "present"),
            ("9.8", "LAB_VALUE", "present"),
            ("IU/L", "LAB_UNIT", "present"),
            ("hyperthyroidism", "DIAGNOSIS", "present"),
            ("Graves disease", "DIAGNOSIS", "possible"),
            ("subacute thyroiditis", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_en_20",
        "language": "en",
        "doc_type": "lab_report",
        "document_date": "2026-04-24",
        "text": (
            "HEPATIC & COAGULATION PROFILE | SIM-EN-020 | 2026-04-24\n"
            "AST: 312 U/L | ALT: 488 U/L | Alkaline phosphatase: 118 U/L\n"
            "Total bilirubin: 3.4 mg/dL | INR: 1.3 | Acetaminophen level: undetectable.\n"
            "Hepatocellular injury pattern; possible viral hepatitis or drug-induced liver injury. Acute liver failure absent."
        ),
        "spans": [
            ("SIM-EN-020", "PATIENT_ID", "present"),
            ("2026-04-24", "DATE", "present"),
            ("AST", "LAB_TEST", "present"),
            ("312", "LAB_VALUE", "present"),
            ("U/L", "LAB_UNIT", "present", 0),
            ("ALT", "LAB_TEST", "present"),
            ("488", "LAB_VALUE", "present"),
            ("U/L", "LAB_UNIT", "present", 1),
            ("Alkaline phosphatase", "LAB_TEST", "present"),
            ("118", "LAB_VALUE", "present"),
            ("U/L", "LAB_UNIT", "present", 2),
            ("Total bilirubin", "LAB_TEST", "present"),
            ("3.4", "LAB_VALUE", "present"),
            ("mg/dL", "LAB_UNIT", "present"),
            ("INR", "LAB_TEST", "present"),
            ("1.3", "LAB_VALUE", "present"),
            ("viral hepatitis", "DIAGNOSIS", "possible"),
            ("drug-induced liver injury", "DIAGNOSIS", "possible"),
            ("Acute liver failure", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_en_21",
        "language": "en",
        "doc_type": "lab_report",
        "document_date": "2026-05-12",
        "text": (
            "COAGULATION & D-DIMER LAB — SIM-EN-021 — May 12, 2026\n"
            "D-dimer: 2840 ng/mL (H) | Fibrinogen: 4.1 g/L | aPTT: 29 s | Platelets: 218 x10^9/L.\n"
            "Markedly elevated D-dimer warrants CT pulmonary angiography to rule out pulmonary embolism; DIC unlikely given normal fibrinogen and platelets."
        ),
        "spans": [
            ("SIM-EN-021", "PATIENT_ID", "present"),
            ("May 12, 2026", "DATE", "present"),
            ("D-dimer", "LAB_TEST", "present", 0),
            ("2840", "LAB_VALUE", "present"),
            ("ng/mL", "LAB_UNIT", "present"),
            ("Fibrinogen", "LAB_TEST", "present"),
            ("4.1", "LAB_VALUE", "present"),
            ("g/L", "LAB_UNIT", "present"),
            ("aPTT", "LAB_TEST", "present"),
            ("29", "LAB_VALUE", "present"),
            ("s", "LAB_UNIT", "present"),
            ("Platelets", "LAB_TEST", "present"),
            ("218", "LAB_VALUE", "present"),
            ("x10^9/L", "LAB_UNIT", "present"),
            ("CT pulmonary angiography", "PROCEDURE", "present"),
            ("pulmonary embolism", "DIAGNOSIS", "possible"),
            ("DIC", "DIAGNOSIS", "absent"),
        ],
    },
    # --- PRESCRIPTIONS (7) ---
    {
        "doc_id": "llm_en_22",
        "language": "en",
        "doc_type": "prescription",
        "document_date": "2026-01-28",
        "text": (
            "OUTPATIENT PRESCRIPTION | Patient: SIM-EN-022 | Date: 2026-01-28\n"
            "Diagnosis: acute bacterial sinusitis.\n"
            "1) Amoxicillin-clavulanate 875 mg PO q12h for 10 days.\n"
            "2) Fluticasone propionate 50 mcg intranasal twice daily for 14 days.\n"
            "3) Acetaminophen 500 mg PO q6h PRN for facial pain. Stop antibiotic immediately if rash develops."
        ),
        "spans": [
            ("SIM-EN-022", "PATIENT_ID", "present"),
            ("2026-01-28", "DATE", "present"),
            ("acute bacterial sinusitis", "DIAGNOSIS", "present"),
            ("Amoxicillin-clavulanate", "MEDICATION", "present"),
            ("875 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("q12h", "FREQUENCY", "present"),
            ("for 10 days", "DURATION", "present"),
            ("Fluticasone propionate", "MEDICATION", "present"),
            ("50 mcg", "DOSAGE", "present"),
            ("intranasal", "ROUTE", "present"),
            ("twice daily", "FREQUENCY", "present"),
            ("for 14 days", "DURATION", "present"),
            ("Acetaminophen", "MEDICATION", "present"),
            ("500 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("q6h", "FREQUENCY", "present"),
            ("PRN", "FREQUENCY", "present"),
            ("facial pain", "SYMPTOM", "present"),
            ("rash", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_en_23",
        "language": "en",
        "doc_type": "prescription",
        "document_date": "2026-02-11",
        "text": (
            "Rx Order — SIM-EN-023 — 11 Feb 2026\n"
            "Indication: persistent moderate asthma.\n"
            "- Budesonide-formoterol 160/4.5 mcg inhaled BID for 3 months.\n"
            "- Albuterol 90 mcg inhaled q4h PRN for wheezing or shortness of breath.\n"
            "- Montelukast 10 mg PO nightly for 90 days."
        ),
        "spans": [
            ("SIM-EN-023", "PATIENT_ID", "present"),
            ("11 Feb 2026", "DATE", "present"),
            ("asthma", "DIAGNOSIS", "present"),
            ("Budesonide-formoterol", "MEDICATION", "present"),
            ("160/4.5 mcg", "DOSAGE", "present"),
            ("inhaled", "ROUTE", "present", 0),
            ("BID", "FREQUENCY", "present"),
            ("for 3 months", "DURATION", "present"),
            ("Albuterol", "MEDICATION", "present"),
            ("90 mcg", "DOSAGE", "present"),
            ("inhaled", "ROUTE", "present", 1),
            ("q4h", "FREQUENCY", "present"),
            ("PRN", "FREQUENCY", "present"),
            ("wheezing", "SYMPTOM", "hypothetical"),
            ("shortness of breath", "SYMPTOM", "hypothetical"),
            ("Montelukast", "MEDICATION", "present"),
            ("10 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present"),
            ("nightly", "FREQUENCY", "present"),
            ("for 90 days", "DURATION", "present"),
        ],
    },
    {
        "doc_id": "llm_en_24",
        "language": "en",
        "doc_type": "prescription",
        "document_date": "2026-03-06",
        "text": (
            "Discharge Rx | ID SIM-EN-024 | 03/06/2026\n"
            "For deep vein thrombosis of the left femoral vein:\n"
            "Apixaban 10 mg PO twice daily for 7 days, then apixaban 5 mg PO twice daily for 6 months.\n"
            "Avoid ibuprofen; return to ED if hematuria or melena occurs."
        ),
        "spans": [
            ("SIM-EN-024", "PATIENT_ID", "present"),
            ("03/06/2026", "DATE", "present"),
            ("deep vein thrombosis", "DIAGNOSIS", "present"),
            ("left femoral vein", "ANATOMY", "present"),
            ("Apixaban", "MEDICATION", "present"),
            ("10 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("twice daily", "FREQUENCY", "present", 0),
            ("for 7 days", "DURATION", "present"),
            ("apixaban", "MEDICATION", "present"),
            ("5 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("twice daily", "FREQUENCY", "present", 1),
            ("for 6 months", "DURATION", "present"),
            ("ibuprofen", "MEDICATION", "absent"),
            ("hematuria", "SYMPTOM", "hypothetical"),
            ("melena", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_en_25",
        "language": "en",
        "doc_type": "prescription",
        "document_date": "2026-03-22",
        "text": (
            "Dermatology Prescription — SIM-EN-025 — Mar 22, 2026\n"
            "Condition: severe atopic dermatitis flare with secondary impetigo.\n"
            "1. Cephalexin 500 mg PO q6h x 7 days.\n"
            "2. Triamcinolone 0.1% topical twice daily for 14 days to flexural forearms.\n"
            "3. Hydroxyzine 25 mg PO nightly PRN for pruritus."
        ),
        "spans": [
            ("SIM-EN-025", "PATIENT_ID", "present"),
            ("Mar 22, 2026", "DATE", "present"),
            ("atopic dermatitis", "DIAGNOSIS", "present"),
            ("impetigo", "DIAGNOSIS", "present"),
            ("Cephalexin", "MEDICATION", "present"),
            ("500 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("q6h", "FREQUENCY", "present"),
            ("7 days", "DURATION", "present"),
            ("Triamcinolone", "MEDICATION", "present"),
            ("0.1%", "DOSAGE", "present"),
            ("topical", "ROUTE", "present"),
            ("twice daily", "FREQUENCY", "present"),
            ("for 14 days", "DURATION", "present"),
            ("forearms", "ANATOMY", "present"),
            ("Hydroxyzine", "MEDICATION", "present"),
            ("25 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("nightly", "FREQUENCY", "present"),
            ("PRN", "FREQUENCY", "present"),
            ("pruritus", "SYMPTOM", "present"),
        ],
    },
    {
        "doc_id": "llm_en_26",
        "language": "en",
        "doc_type": "prescription",
        "document_date": "2026-04-09",
        "text": (
            "Psychiatry Medication Order | SIM-EN-026 | 2026-04-09\n"
            "Indication: major depressive disorder and insomnia; no suicidal ideation.\n"
            "- Sertraline 50 mg PO once daily for 2 weeks, then increase to 100 mg PO daily for 6 months.\n"
            "- Trazodone 50 mg PO nightly PRN for insomnia."
        ),
        "spans": [
            ("SIM-EN-026", "PATIENT_ID", "present"),
            ("2026-04-09", "DATE", "present"),
            ("major depressive disorder", "DIAGNOSIS", "present"),
            ("insomnia", "SYMPTOM", "present", 0),
            ("suicidal ideation", "SYMPTOM", "absent"),
            ("Sertraline", "MEDICATION", "present"),
            ("50 mg", "DOSAGE", "present", 0),
            ("PO", "ROUTE", "present", 0),
            ("once daily", "FREQUENCY", "present"),
            ("for 2 weeks", "DURATION", "present"),
            ("100 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("daily", "FREQUENCY", "present", 1),
            ("for 6 months", "DURATION", "present"),
            ("Trazodone", "MEDICATION", "present"),
            ("50 mg", "DOSAGE", "present", 1),
            ("PO", "ROUTE", "present", 2),
            ("nightly", "FREQUENCY", "present"),
            ("PRN", "FREQUENCY", "present"),
            ("insomnia", "SYMPTOM", "hypothetical", 1),
        ],
    },
    {
        "doc_id": "llm_en_27",
        "language": "en",
        "doc_type": "prescription",
        "document_date": "2026-04-17",
        "text": (
            "Urology Rx — Chart SIM-EN-027 — 17-Apr-2026\n"
            "Acute uncomplicated cystitis:\n"
            "Nitrofurantoin 100 mg PO q12h for 5 days.\n"
            "Phenazopyridine 200 mg PO TID for 2 days for dysuria.\n"
            "Contact clinic if fever or flank pain develops."
        ),
        "spans": [
            ("SIM-EN-027", "PATIENT_ID", "present"),
            ("17-Apr-2026", "DATE", "present"),
            ("cystitis", "DIAGNOSIS", "present"),
            ("Nitrofurantoin", "MEDICATION", "present"),
            ("100 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("q12h", "FREQUENCY", "present"),
            ("for 5 days", "DURATION", "present"),
            ("Phenazopyridine", "MEDICATION", "present"),
            ("200 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("TID", "FREQUENCY", "present"),
            ("for 2 days", "DURATION", "present"),
            ("dysuria", "SYMPTOM", "present"),
            ("fever", "SYMPTOM", "hypothetical"),
            ("flank pain", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_en_28",
        "language": "en",
        "doc_type": "prescription",
        "document_date": "2026-05-05",
        "text": (
            "Cardiometabolic Refill | SIM-EN-028 | May 5, 2026\n"
            "Diagnoses: essential hypertension, gout.\n"
            "1. Amlodipine 5 mg PO once daily for 90 days.\n"
            "2. Allopurinol 300 mg PO daily for 3 months.\n"
            "3. Colchicine 0.6 mg PO once daily for 4 weeks for flare prophylaxis."
        ),
        "spans": [
            ("SIM-EN-028", "PATIENT_ID", "present"),
            ("May 5, 2026", "DATE", "present"),
            ("essential hypertension", "DIAGNOSIS", "present"),
            ("gout", "DIAGNOSIS", "present"),
            ("Amlodipine", "MEDICATION", "present"),
            ("5 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("once daily", "FREQUENCY", "present", 0),
            ("for 90 days", "DURATION", "present"),
            ("Allopurinol", "MEDICATION", "present"),
            ("300 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("daily", "FREQUENCY", "present", 1),
            ("for 3 months", "DURATION", "present"),
            ("Colchicine", "MEDICATION", "present"),
            ("0.6 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 2),
            ("once daily", "FREQUENCY", "present", 1),
            ("for 4 weeks", "DURATION", "present"),
        ],
    },
    # --- RADIOLOGY REPORTS (6) ---
    {
        "doc_id": "llm_en_29",
        "language": "en",
        "doc_type": "radiology_report",
        "document_date": "2026-01-12",
        "text": (
            "CT PULMONARY ANGIOGRAPHY | SIM-EN-029 | 2026-01-12\n"
            "Indication: sudden pleuritic chest pain and tachycardia.\n"
            "Findings: Intraluminal filling defect within the right main pulmonary artery and segmental branches of the right lower lobe. Small right pleural effusion.\n"
            "No aortic dissection or pneumothorax. Impression: Acute pulmonary embolism; mild right ventricular strain cannot be excluded."
        ),
        "spans": [
            ("SIM-EN-029", "PATIENT_ID", "present"),
            ("2026-01-12", "DATE", "present"),
            ("chest pain", "SYMPTOM", "present"),
            ("tachycardia", "SYMPTOM", "present"),
            ("right main pulmonary artery", "ANATOMY", "present"),
            ("right lower lobe", "ANATOMY", "present"),
            ("pleural effusion", "DIAGNOSIS", "present"),
            ("aortic dissection", "DIAGNOSIS", "absent"),
            ("pneumothorax", "DIAGNOSIS", "absent"),
            ("Acute pulmonary embolism", "DIAGNOSIS", "present"),
            ("right ventricular strain", "DIAGNOSIS", "possible"),
        ],
    },
    {
        "doc_id": "llm_en_30",
        "language": "en",
        "doc_type": "radiology_report",
        "document_date": "2026-02-09",
        "text": (
            "MRI BRAIN WITH CONTRAST — ID SIM-EN-030 — 09 Feb 2026\n"
            "Clinical info: 2 weeks of progressive diplopia and morning headache.\n"
            "Findings: Well-circumscribed 2.4 cm extra-axial enhancing mass along the right frontal convexity with mild perilesional edema; no acute infarct or midline shift.\n"
            "Impression: Findings favor meningioma; dural metastasis less likely."
        ),
        "spans": [
            ("SIM-EN-030", "PATIENT_ID", "present"),
            ("09 Feb 2026", "DATE", "present"),
            ("2 weeks", "DURATION", "present"),
            ("diplopia", "SYMPTOM", "present"),
            ("headache", "SYMPTOM", "present"),
            ("right frontal convexity", "ANATOMY", "present"),
            ("perilesional edema", "DIAGNOSIS", "present"),
            ("acute infarct", "DIAGNOSIS", "absent"),
            ("midline shift", "DIAGNOSIS", "absent"),
            ("meningioma", "DIAGNOSIS", "possible"),
            ("dural metastasis", "DIAGNOSIS", "possible"),
        ],
    },
    {
        "doc_id": "llm_en_31",
        "language": "en",
        "doc_type": "radiology_report",
        "document_date": "2026-03-01",
        "text": (
            "ABDOMINAL ULTRASOUND | SIM-EN-031 | Mar 1, 2026\n"
            "Indication: postprandial right upper quadrant pain and nausea.\n"
            "Findings: Multiple shadowing calculi within a distended gallbladder with wall thickening (5 mm) and sonographic Murphy sign. Common bile duct measures 4 mm without choledocholithiasis.\n"
            "Impression: Acute calculous cholecystitis; hepatic steatosis absent."
        ),
        "spans": [
            ("SIM-EN-031", "PATIENT_ID", "present"),
            ("Mar 1, 2026", "DATE", "present"),
            ("right upper quadrant", "ANATOMY", "present"),
            ("pain", "SYMPTOM", "present"),
            ("nausea", "SYMPTOM", "present"),
            ("gallbladder", "ANATOMY", "present"),
            ("Common bile duct", "ANATOMY", "present"),
            ("choledocholithiasis", "DIAGNOSIS", "absent"),
            ("Acute calculous cholecystitis", "DIAGNOSIS", "present"),
            ("hepatic steatosis", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_en_32",
        "language": "en",
        "doc_type": "radiology_report",
        "document_date": "2026-03-25",
        "text": (
            "LUMBAR SPINE MRI — Chart SIM-EN-032 — 2026-03-25\n"
            "History: 3 months of radiating left leg pain and numbness.\n"
            "Findings: Left paracentral disc extrusion at L4-L5 compressing the traversing left L5 nerve root. Mild facet arthropathy.\n"
            "No cauda equina compression, spondylodiscitis, or vertebral fracture. Impression: L4-L5 disc herniation."
        ),
        "spans": [
            ("SIM-EN-032", "PATIENT_ID", "present"),
            ("2026-03-25", "DATE", "present"),
            ("3 months", "DURATION", "present"),
            ("left leg", "ANATOMY", "present"),
            ("pain", "SYMPTOM", "present"),
            ("numbness", "SYMPTOM", "present"),
            ("L4-L5", "ANATOMY", "present", 0),
            ("left L5 nerve root", "ANATOMY", "present"),
            ("cauda equina compression", "DIAGNOSIS", "absent"),
            ("spondylodiscitis", "DIAGNOSIS", "absent"),
            ("vertebral fracture", "DIAGNOSIS", "absent"),
            ("disc herniation", "DIAGNOSIS", "present"),
        ],
    },
    {
        "doc_id": "llm_en_33",
        "language": "en",
        "doc_type": "radiology_report",
        "document_date": "2026-04-14",
        "text": (
            "CHEST RADIOGRAPH (PA/LAT) | SIM-EN-033 | 14 Apr 2026\n"
            "Indication: persistent dry cough for 5 weeks in a former smoker.\n"
            "Findings: Spiculated 1.9 cm nodule in the left upper lobe; hilar lymphadenopathy possible. No pleural effusion or cardiomegaly.\n"
            "Recommendation: Contrast chest CT to exclude primary lung carcinoma vs granuloma."
        ),
        "spans": [
            ("SIM-EN-033", "PATIENT_ID", "present"),
            ("14 Apr 2026", "DATE", "present"),
            ("dry cough", "SYMPTOM", "present"),
            ("for 5 weeks", "DURATION", "present"),
            ("left upper lobe", "ANATOMY", "present"),
            ("hilar lymphadenopathy", "DIAGNOSIS", "possible"),
            ("pleural effusion", "DIAGNOSIS", "absent"),
            ("cardiomegaly", "DIAGNOSIS", "absent"),
            ("Contrast chest CT", "PROCEDURE", "present"),
            ("lung carcinoma", "DIAGNOSIS", "possible"),
            ("granuloma", "DIAGNOSIS", "possible"),
        ],
    },
    {
        "doc_id": "llm_en_34",
        "language": "en",
        "doc_type": "radiology_report",
        "document_date": "2026-05-01",
        "text": (
            "KNEE MRI (RIGHT) — SIM-EN-034 — 05/01/2026\n"
            "History: twisting sports injury 6 days ago w/ locking and joint effusion.\n"
            "Findings: Complete tear of the anterior cruciate ligament and complex bucket-handle tear of the medial meniscus; bone contusion of the lateral femoral condyle.\n"
            "Posterior cruciate ligament and collateral ligaments intact; no occult tibial plateau fracture."
        ),
        "spans": [
            ("SIM-EN-034", "PATIENT_ID", "present"),
            ("05/01/2026", "DATE", "present"),
            ("6 days ago", "DATE", "present"),
            ("joint effusion", "SYMPTOM", "present"),
            ("anterior cruciate ligament", "ANATOMY", "present"),
            ("medial meniscus", "ANATOMY", "present"),
            ("bone contusion", "DIAGNOSIS", "present"),
            ("lateral femoral condyle", "ANATOMY", "present"),
            ("Posterior cruciate ligament", "ANATOMY", "present"),
            ("tibial plateau fracture", "DIAGNOSIS", "absent"),
        ],
    },
    # --- OPERATIVE NOTES (6) ---
    {
        "doc_id": "llm_en_35",
        "language": "en",
        "doc_type": "operative_note",
        "document_date": "2026-01-25",
        "text": (
            "OPERATIVE NOTE | SIM-EN-035 | 2026-01-25\n"
            "Preop/Postop Dx: Symptomatic cholelithiasis.\n"
            "Procedure: Laparoscopic cholecystectomy under general anesthesia.\n"
            "Prophylaxis: Cefazolin 2 g IV once pre-incision. Critical view of safety achieved; cystic duct and artery clipped and divided without bile duct injury or hemorrhage. Estimated blood loss 20 mL."
        ),
        "spans": [
            ("SIM-EN-035", "PATIENT_ID", "present"),
            ("2026-01-25", "DATE", "present"),
            ("cholelithiasis", "DIAGNOSIS", "present"),
            ("Laparoscopic cholecystectomy", "PROCEDURE", "present"),
            ("Cefazolin", "MEDICATION", "present"),
            ("2 g", "DOSAGE", "present"),
            ("IV", "ROUTE", "present"),
            ("cystic duct", "ANATOMY", "present"),
            ("bile duct injury", "DIAGNOSIS", "absent"),
            ("hemorrhage", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_en_36",
        "language": "en",
        "doc_type": "operative_note",
        "document_date": "2026-02-18",
        "text": (
            "ORTHOPEDIC SURGERY REPORT — SIM-EN-036 — Feb 18, 2026\n"
            "Indication: Displaced right femoral neck fracture after mechanical fall.\n"
            "Operation: Cemented right hip hemiarthroplasty via posterior approach. Tranexamic acid 1 g IV given at induction. "
            "Stable reduction confirmed; no periprosthetic fracture or sciatic nerve injury. Post-op enoxaparin 40 mg SC once daily for 28 days."
        ),
        "spans": [
            ("SIM-EN-036", "PATIENT_ID", "present"),
            ("Feb 18, 2026", "DATE", "present"),
            ("right femoral neck fracture", "DIAGNOSIS", "present"),
            ("right hip hemiarthroplasty", "PROCEDURE", "present"),
            ("Tranexamic acid", "MEDICATION", "present"),
            ("1 g", "DOSAGE", "present"),
            ("IV", "ROUTE", "present"),
            ("periprosthetic fracture", "DIAGNOSIS", "absent"),
            ("sciatic nerve injury", "DIAGNOSIS", "absent"),
            ("enoxaparin", "MEDICATION", "present"),
            ("40 mg", "DOSAGE", "present"),
            ("SC", "ROUTE", "present"),
            ("once daily", "FREQUENCY", "present"),
            ("for 28 days", "DURATION", "present"),
        ],
    },
    {
        "doc_id": "llm_en_37",
        "language": "en",
        "doc_type": "operative_note",
        "document_date": "2026-03-14",
        "text": (
            "CARDIAC CATHETERIZATION & PCI NOTE | SIM-EN-037 | 2026-03-14\n"
            "Indication: Anterior STEMI.\n"
            "Procedure: Coronary angiography via right radial artery revealed 99% thrombotic occlusion of the proximal left anterior descending artery. "
            "Primary percutaneous coronary intervention with drug-eluting stent placement restored TIMI 3 flow. Heparin IV administered; no coronary dissection or pericardial tamponade."
        ),
        "spans": [
            ("SIM-EN-037", "PATIENT_ID", "present"),
            ("2026-03-14", "DATE", "present"),
            ("Anterior STEMI", "DIAGNOSIS", "present"),
            ("Coronary angiography", "PROCEDURE", "present"),
            ("right radial artery", "ANATOMY", "present"),
            ("proximal left anterior descending artery", "ANATOMY", "present"),
            ("percutaneous coronary intervention", "PROCEDURE", "present"),
            ("Heparin", "MEDICATION", "present"),
            ("IV", "ROUTE", "present"),
            ("coronary dissection", "DIAGNOSIS", "absent"),
            ("pericardial tamponade", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_en_38",
        "language": "en",
        "doc_type": "operative_note",
        "document_date": "2026-04-06",
        "text": (
            "OBSTETRIC OPERATIVE RECORD — SIM-EN-038 — 06 Apr 2026\n"
            "Indication: Arrest of descent at 39 weeks gestation.\n"
            "Procedure: Primary low-transverse cesarean section under spinal anesthesia. Oxytocin 10 units IV infused after delivery of healthy infant; cefazolin 2 g IV administered. "
            "Uterus well contracted without postpartum hemorrhage or bladder injury."
        ),
        "spans": [
            ("SIM-EN-038", "PATIENT_ID", "present"),
            ("06 Apr 2026", "DATE", "present"),
            ("39 weeks", "DURATION", "present"),
            ("cesarean section", "PROCEDURE", "present"),
            ("spinal anesthesia", "PROCEDURE", "present"),
            ("Oxytocin", "MEDICATION", "present"),
            ("10 units", "DOSAGE", "present"),
            ("IV", "ROUTE", "present", 0),
            ("cefazolin", "MEDICATION", "present"),
            ("2 g", "DOSAGE", "present"),
            ("IV", "ROUTE", "present", 1),
            ("Uterus", "ANATOMY", "present"),
            ("postpartum hemorrhage", "DIAGNOSIS", "absent"),
            ("bladder injury", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_en_39",
        "language": "en",
        "doc_type": "operative_note",
        "document_date": "2026-04-21",
        "text": (
            "ENT OPERATIVE SUMMARY | SIM-EN-039 | 2026-04-21\n"
            "Dx: Multinodular goiter with tracheal compression.\n"
            "Operation: Total thyroidectomy with intraoperative recurrent laryngeal nerve monitoring. "
            "Parathyroid glands preserved in situ; hemostasis meticulous with no postoperative hematoma. Started levothyroxine 100 mcg PO daily and calcium carbonate 1000 mg PO BID."
        ),
        "spans": [
            ("SIM-EN-039", "PATIENT_ID", "present"),
            ("2026-04-21", "DATE", "present"),
            ("Multinodular goiter", "DIAGNOSIS", "present"),
            ("tracheal compression", "DIAGNOSIS", "present"),
            ("Total thyroidectomy", "PROCEDURE", "present"),
            ("recurrent laryngeal nerve", "ANATOMY", "present"),
            ("Parathyroid glands", "ANATOMY", "present"),
            ("hematoma", "DIAGNOSIS", "absent"),
            ("levothyroxine", "MEDICATION", "present"),
            ("100 mcg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("daily", "FREQUENCY", "present"),
            ("calcium carbonate", "MEDICATION", "present"),
            ("1000 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("BID", "FREQUENCY", "present"),
        ],
    },
    {
        "doc_id": "llm_en_40",
        "language": "en",
        "doc_type": "operative_note",
        "document_date": "2026-05-10",
        "text": (
            "VASCULAR SURGERY NOTE — SIM-EN-040 — May 10, 2026\n"
            "Dx: High-grade symptomatic right internal carotid artery stenosis (85%).\n"
            "Procedure: Right carotid endarterectomy with bovine pericardial patch angioplasty. Systemic heparin 5000 units IV prior to clamping; intraoperative shunt not required. "
            "Patient awoke neurologically intact without stroke or cranial nerve palsy. Continue aspirin 81 mg PO once daily."
        ),
        "spans": [
            ("SIM-EN-040", "PATIENT_ID", "present"),
            ("May 10, 2026", "DATE", "present"),
            ("right internal carotid artery", "ANATOMY", "present"),
            ("stenosis", "DIAGNOSIS", "present"),
            ("carotid endarterectomy", "PROCEDURE", "present"),
            ("heparin", "MEDICATION", "present"),
            ("5000 units", "DOSAGE", "present"),
            ("IV", "ROUTE", "present"),
            ("stroke", "DIAGNOSIS", "absent"),
            ("cranial nerve palsy", "DIAGNOSIS", "absent"),
            ("aspirin", "MEDICATION", "present"),
            ("81 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present"),
            ("once daily", "FREQUENCY", "present"),
        ],
    },
]
