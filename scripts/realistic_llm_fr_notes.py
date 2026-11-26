"""40 LLM-authored realistic French clinical notes (free-text, single-annotator, zero PHI) across 6 doc types."""

from __future__ import annotations

from typing import Any

# Each span entry is (exact_text, entity_type, assertion) or (exact_text, entity_type, assertion, occurrence_0_based)
FR_NOTES: list[dict[str, Any]] = [
    # --- DISCHARGE SUMMARIES (7) ---
    {
        "doc_id": "llm_fr_01",
        "language": "fr",
        "doc_type": "discharge_summary",
        "document_date": "2026-02-12",
        "text": (
            "COMPTE RENDU D'HOSPITALISATION | ID : SIM-FR-001 | 12/02/2026\n"
            "Patient 72 ans, ATCD de fibrillation auriculaire permanente et HTA, admis mi-février pour dyspnée stade III et œdèmes des membres inférieurs évoluant depuis 5 jours. "
            "NT-proBNP 5140 pg/mL ; ETT : FEVG 38 %, absence d'épanchement péricardique. "
            "PEC par furosémide 40 mg IV 2x/j pendant 3 jours puis relais furosémide 40 mg PO 1 fois/jour et bisoprolol 5 mg PO le matin. "
            "Décompensation cardiaque résolue ; embolie pulmonaire peu probable devant des D-dimères bas. Reconsulter si douleur thoracique."
        ),
        "spans": [
            ("SIM-FR-001", "PATIENT_ID", "present"),
            ("12/02/2026", "DATE", "present"),
            ("fibrillation auriculaire", "DIAGNOSIS", "historical"),
            ("HTA", "DIAGNOSIS", "historical"),
            ("mi-février", "DATE", "present"),
            ("dyspnée", "SYMPTOM", "present"),
            ("œdèmes des membres inférieurs", "SYMPTOM", "present"),
            ("depuis 5 jours", "DURATION", "present"),
            ("NT-proBNP", "LAB_TEST", "present"),
            ("5140", "LAB_VALUE", "present"),
            ("pg/mL", "LAB_UNIT", "present"),
            ("ETT", "PROCEDURE", "present"),
            ("FEVG", "LAB_TEST", "present"),
            ("38", "LAB_VALUE", "present"),
            ("%", "LAB_UNIT", "present"),
            ("épanchement péricardique", "DIAGNOSIS", "absent"),
            ("furosémide", "MEDICATION", "present", 0),
            ("40 mg", "DOSAGE", "present", 0),
            ("IV", "ROUTE", "present"),
            ("2x/j", "FREQUENCY", "present"),
            ("pendant 3 jours", "DURATION", "present"),
            ("furosémide", "MEDICATION", "present", 1),
            ("40 mg", "DOSAGE", "present", 1),
            ("PO", "ROUTE", "present", 0),
            ("1 fois/jour", "FREQUENCY", "present"),
            ("bisoprolol", "MEDICATION", "present"),
            ("5 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("le matin", "FREQUENCY", "present"),
            ("Décompensation cardiaque", "DIAGNOSIS", "present"),
            ("embolie pulmonaire", "DIAGNOSIS", "absent"),
            ("D-dimères", "LAB_TEST", "present"),
            ("douleur thoracique", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_fr_02",
        "language": "fr",
        "doc_type": "discharge_summary",
        "document_date": "2026-01-20",
        "text": (
            "Synthèse de sortie — Dossier SIM-FR-002 — 20 janvier 2026\n"
            "Femme 58 ans hospitalisée 4 jours pour pyélonéphrite aiguë droite avec fièvre à 39,2 °C et brûlures mictionnelles depuis il y a 3 jours. "
            "CRP initiale 168 mg/L, créatininémie 84 µmol/L. ECBU positif à Escherichia coli multisensible. "
            "TTT initial par ceftriaxone 1 g IV 1 fois/jour pendant 48 heures puis relais ciprofloxacine 500 mg PO 2x/j pendant 7 jours. "
            "Échographie rénale sans dilatation pyélocalicielle ni abcès rénal ; revenir aux urgences si reprise fébrile."
        ),
        "spans": [
            ("SIM-FR-002", "PATIENT_ID", "present"),
            ("20 janvier 2026", "DATE", "present"),
            ("4 jours", "DURATION", "present"),
            ("pyélonéphrite aiguë", "DIAGNOSIS", "present"),
            ("fièvre", "SYMPTOM", "present"),
            ("brûlures mictionnelles", "SYMPTOM", "present"),
            ("il y a 3 jours", "DATE", "present"),
            ("CRP", "LAB_TEST", "present"),
            ("168", "LAB_VALUE", "present"),
            ("mg/L", "LAB_UNIT", "present"),
            ("créatininémie", "LAB_TEST", "present"),
            ("84", "LAB_VALUE", "present"),
            ("µmol/L", "LAB_UNIT", "present"),
            ("ECBU", "LAB_TEST", "present"),
            ("ceftriaxone", "MEDICATION", "present"),
            ("1 g", "DOSAGE", "present"),
            ("IV", "ROUTE", "present"),
            ("1 fois/jour", "FREQUENCY", "present"),
            ("pendant 48 heures", "DURATION", "present"),
            ("ciprofloxacine", "MEDICATION", "present"),
            ("500 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present"),
            ("2x/j", "FREQUENCY", "present"),
            ("pendant 7 jours", "DURATION", "present"),
            ("Échographie rénale", "PROCEDURE", "present"),
            ("abcès rénal", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_fr_03",
        "language": "fr",
        "doc_type": "discharge_summary",
        "document_date": "2025-11-28",
        "text": (
            "CRH Pneumologie | SIM-FR-003 | 2025-11-28\n"
            "Homme 64 ans, ATCD de BPCO stade II et cancer bronchique chez le père, admis pour pneumopathie aiguë communautaire du lobe inférieur droit avec toux productive et hypoxémie. "
            "Antigénurie légionelle négative ; absence d'argument pour une tuberculose pulmonaire. "
            "Antibiothérapie par amoxicilline 1000 mg PO 3x/j pendant 7 jours. Possible syndrome d'apnées du sommeil à réévaluer par polysomnographie."
        ),
        "spans": [
            ("SIM-FR-003", "PATIENT_ID", "present"),
            ("2025-11-28", "DATE", "present"),
            ("BPCO", "DIAGNOSIS", "historical"),
            ("cancer bronchique", "DIAGNOSIS", "family"),
            ("pneumopathie aiguë communautaire", "DIAGNOSIS", "present"),
            ("lobe inférieur droit", "ANATOMY", "present"),
            ("toux productive", "SYMPTOM", "present"),
            ("hypoxémie", "SYMPTOM", "present"),
            ("tuberculose pulmonaire", "DIAGNOSIS", "absent"),
            ("amoxicilline", "MEDICATION", "present"),
            ("1000 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present"),
            ("3x/j", "FREQUENCY", "present"),
            ("pendant 7 jours", "DURATION", "present"),
            ("syndrome d'apnées du sommeil", "DIAGNOSIS", "possible"),
            ("polysomnographie", "PROCEDURE", "present"),
        ],
    },
    {
        "doc_id": "llm_fr_04",
        "language": "fr",
        "doc_type": "discharge_summary",
        "document_date": "2026-03-07",
        "text": (
            "Lettre de sortie UNV — SIM-FR-004 — 07/03/2026\n"
            "Patient 69 ans admis en urgence pour hémiparésie gauche et aphasie brutales. IRM cérébrale confirmant un AVC ischémique sylvien droit sans transformation hémorragique. "
            "Thrombolyse intraveineuse par altéplase IV réalisée à H2 avec récupération motrice quasi complète. "
            "TTT de sortie : aspirine 75 mg PO 1 fois/jour, atorvastatine 40 mg PO le soir et ramipril 5 mg PO le matin. Pas de trouble de la déglutition résiduel."
        ),
        "spans": [
            ("SIM-FR-004", "PATIENT_ID", "present"),
            ("07/03/2026", "DATE", "present"),
            ("hémiparésie gauche", "SYMPTOM", "present"),
            ("aphasie", "SYMPTOM", "present"),
            ("IRM cérébrale", "PROCEDURE", "present"),
            ("AVC ischémique", "DIAGNOSIS", "present"),
            ("transformation hémorragique", "DIAGNOSIS", "absent"),
            ("Thrombolyse intraveineuse", "PROCEDURE", "present"),
            ("altéplase", "MEDICATION", "present"),
            ("IV", "ROUTE", "present"),
            ("aspirine", "MEDICATION", "present"),
            ("75 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("1 fois/jour", "FREQUENCY", "present"),
            ("atorvastatine", "MEDICATION", "present"),
            ("40 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("le soir", "FREQUENCY", "present"),
            ("ramipril", "MEDICATION", "present"),
            ("5 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 2),
            ("le matin", "FREQUENCY", "present"),
            ("trouble de la déglutition", "SYMPTOM", "absent"),
        ],
    },
    {
        "doc_id": "llm_fr_05",
        "language": "fr",
        "doc_type": "discharge_summary",
        "document_date": "2026-04-15",
        "text": (
            "CRH Diabétologie | Patient SIM-FR-005 | 15 avril 2026\n"
            "Jeune homme 24 ans, diabète de type 1 connu depuis 2016, hospitalisé pour acidocétose diabétique inaugurée par vomissements et douleurs abdominales. "
            "Glycémie capillaire 4,8 g/L, pH artériel 7,16, cétonémie 5,4 mmol/L. "
            "Réhydratation et insuline rapide IV puis reprise insuline glargine 20 UI SC le soir. Aucun foyer infectieux retrouvé. Consulter sans délai en cas de nausées persistantes."
        ),
        "spans": [
            ("SIM-FR-005", "PATIENT_ID", "present"),
            ("15 avril 2026", "DATE", "present"),
            ("diabète de type 1", "DIAGNOSIS", "historical"),
            ("2016", "DATE", "present"),
            ("acidocétose diabétique", "DIAGNOSIS", "present"),
            ("vomissements", "SYMPTOM", "present"),
            ("douleurs abdominales", "SYMPTOM", "present"),
            ("Glycémie capillaire", "LAB_TEST", "present"),
            ("4,8", "LAB_VALUE", "present"),
            ("g/L", "LAB_UNIT", "present"),
            ("pH artériel", "LAB_TEST", "present"),
            ("7,16", "LAB_VALUE", "present"),
            ("cétonémie", "LAB_TEST", "present"),
            ("5,4", "LAB_VALUE", "present"),
            ("mmol/L", "LAB_UNIT", "present"),
            ("insuline rapide", "MEDICATION", "present"),
            ("IV", "ROUTE", "present"),
            ("insuline glargine", "MEDICATION", "present"),
            ("20 UI", "DOSAGE", "present"),
            ("SC", "ROUTE", "present"),
            ("le soir", "FREQUENCY", "present"),
            ("nausées", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_fr_06",
        "language": "fr",
        "doc_type": "discharge_summary",
        "document_date": "2025-10-25",
        "text": (
            "Compte rendu de séjour — SIM-FR-006 — 25/10/2025\n"
            "Patiente 49 ans admise pour pancréatite aiguë biliaire avec épigastralgies transfixiantes depuis 24 heures. Lipasémie à 1420 U/L. "
            "Scanner abdominal sans nécrose pancréatique ni coulée de nécrose ; angiocholite écartée. "
            "Jeûne court, hydratation IV, paracétamol 1000 mg IV 3x/j puis réalimentation orale bien tolérée. Cholécystectomie programmée dans 3 semaines."
        ),
        "spans": [
            ("SIM-FR-006", "PATIENT_ID", "present"),
            ("25/10/2025", "DATE", "present"),
            ("pancréatite aiguë biliaire", "DIAGNOSIS", "present"),
            ("épigastralgies", "SYMPTOM", "present"),
            ("depuis 24 heures", "DURATION", "present"),
            ("Lipasémie", "LAB_TEST", "present"),
            ("1420", "LAB_VALUE", "present"),
            ("U/L", "LAB_UNIT", "present"),
            ("Scanner abdominal", "PROCEDURE", "present"),
            ("nécrose pancréatique", "DIAGNOSIS", "absent"),
            ("angiocholite", "DIAGNOSIS", "absent"),
            ("IV", "ROUTE", "present", 0),
            ("paracétamol", "MEDICATION", "present"),
            ("1000 mg", "DOSAGE", "present"),
            ("IV", "ROUTE", "present", 1),
            ("3x/j", "FREQUENCY", "present"),
            ("Cholécystectomie", "PROCEDURE", "present"),
            ("3 semaines", "DURATION", "present"),
        ],
    },
    {
        "doc_id": "llm_fr_07",
        "language": "fr",
        "doc_type": "discharge_summary",
        "document_date": "2026-05-06",
        "text": (
            "CRH Hématologie | ID SIM-FR-007 | 2026-05-06\n"
            "Homme 28 ans drépanocytaire homozygote SS admis pour crise vaso-occlusive hyperalgique touchant les fémurs et le rachis lombaire. "
            "Hémoglobine 8,1 g/dL, réticulocytes 240 G/L. Radiographie thoracique normale éliminant un syndrome thoracique aigu. "
            "PEC par morphine IV puis relais paracétamol 1000 mg PO 3x/j et poursuite hydroxyurée 1000 mg PO 1 fois/jour. Revenir si fièvre ou dyspnée."
        ),
        "spans": [
            ("SIM-FR-007", "PATIENT_ID", "present"),
            ("2026-05-06", "DATE", "present"),
            ("crise vaso-occlusive", "DIAGNOSIS", "present"),
            ("fémurs", "ANATOMY", "present"),
            ("rachis lombaire", "ANATOMY", "present"),
            ("Hémoglobine", "LAB_TEST", "present"),
            ("8,1", "LAB_VALUE", "present"),
            ("g/dL", "LAB_UNIT", "present"),
            ("réticulocytes", "LAB_TEST", "present"),
            ("240", "LAB_VALUE", "present"),
            ("G/L", "LAB_UNIT", "present"),
            ("Radiographie thoracique", "PROCEDURE", "present"),
            ("syndrome thoracique aigu", "DIAGNOSIS", "absent"),
            ("morphine", "MEDICATION", "present"),
            ("IV", "ROUTE", "present"),
            ("paracétamol", "MEDICATION", "present"),
            ("1000 mg", "DOSAGE", "present", 0),
            ("PO", "ROUTE", "present", 0),
            ("3x/j", "FREQUENCY", "present"),
            ("hydroxyurée", "MEDICATION", "present"),
            ("1000 mg", "DOSAGE", "present", 1),
            ("PO", "ROUTE", "present", 1),
            ("1 fois/jour", "FREQUENCY", "present"),
            ("fièvre", "SYMPTOM", "hypothetical"),
            ("dyspnée", "SYMPTOM", "hypothetical"),
        ],
    },
    # --- CONSULTATIONS (7) ---
    {
        "doc_id": "llm_fr_08",
        "language": "fr",
        "doc_type": "consultation",
        "document_date": "2026-01-16",
        "text": (
            "Consultation de Cardiologie — SIM-FR-008 — mi-janvier 2026\n"
            "Patiente 61 ans vue pour palpitations paroxystiques et lipothymies début novembre. "
            "Pas d'angor ni de syncope. ATCD familial d'infarctus du myocarde chez la mère. "
            "ECG en rythme sinusal normal. Pose d'un Holter ECG sur 72 heures pour rechercher une fibrillation auriculaire paroxystique ; débuter métoprolol 50 mg PO 1 fois/jour si récidive."
        ),
        "spans": [
            ("SIM-FR-008", "PATIENT_ID", "present"),
            ("mi-janvier 2026", "DATE", "present"),
            ("palpitations", "SYMPTOM", "present"),
            ("lipothymies", "SYMPTOM", "present"),
            ("début novembre", "DATE", "present"),
            ("angor", "SYMPTOM", "absent"),
            ("syncope", "SYMPTOM", "absent"),
            ("infarctus du myocarde", "DIAGNOSIS", "family"),
            ("ECG", "PROCEDURE", "present", 0),
            ("Holter ECG", "PROCEDURE", "present"),
            ("72 heures", "DURATION", "present"),
            ("fibrillation auriculaire", "DIAGNOSIS", "possible"),
            ("métoprolol", "MEDICATION", "hypothetical"),
            ("50 mg", "DOSAGE", "hypothetical"),
            ("PO", "ROUTE", "hypothetical"),
            ("1 fois/jour", "FREQUENCY", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_fr_09",
        "language": "fr",
        "doc_type": "consultation",
        "document_date": "2026-02-23",
        "text": (
            "Avis Rhumatologie | SIM-FR-009 | 23/02/2026\n"
            "Femme 46 ans consultant pour arthralgies inflammatoires des poignets et chevilles avec dérouillage matinal de 90 minutes depuis 3 mois. "
            "Facteur rhumatoïde 98 UI/mL, CRP 34 mg/L. Pas de psoriasis cutané. "
            "Suspecte polyarthrite rhumatoïde débutante vs rhumatisme psoriasique ; instauration de méthotrexate 15 mg PO 1 fois par semaine et acide folique 5 mg PO hebdomadaire."
        ),
        "spans": [
            ("SIM-FR-009", "PATIENT_ID", "present"),
            ("23/02/2026", "DATE", "present"),
            ("arthralgies", "SYMPTOM", "present"),
            ("poignets", "ANATOMY", "present"),
            ("chevilles", "ANATOMY", "present"),
            ("90 minutes", "DURATION", "present"),
            ("depuis 3 mois", "DURATION", "present"),
            ("Facteur rhumatoïde", "LAB_TEST", "present"),
            ("98", "LAB_VALUE", "present"),
            ("UI/mL", "LAB_UNIT", "present"),
            ("CRP", "LAB_TEST", "present"),
            ("34", "LAB_VALUE", "present"),
            ("mg/L", "LAB_UNIT", "present"),
            ("psoriasis", "DIAGNOSIS", "absent"),
            ("polyarthrite rhumatoïde", "DIAGNOSIS", "possible"),
            ("rhumatisme psoriasique", "DIAGNOSIS", "possible"),
            ("méthotrexate", "MEDICATION", "present"),
            ("15 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("1 fois par semaine", "FREQUENCY", "present"),
            ("acide folique", "MEDICATION", "present"),
            ("5 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("hebdomadaire", "FREQUENCY", "present"),
        ],
    },
    {
        "doc_id": "llm_fr_10",
        "language": "fr",
        "doc_type": "consultation",
        "document_date": "2026-03-19",
        "text": (
            "Consultation Neurologie — SIM-FR-010 — 19 mars 2026\n"
            "Homme 39 ans, ATCD de migraines sans aura, présente des céphalées hémicrâniennes pulsatiles avec photophobie et nausées. "
            "Examen neurologique normal : absence de syndrome méningé ou de déficit moteur. "
            "Prescription de sumatriptan 50 mg PO si crise et amitriptyline 25 mg PO le soir pendant 3 mois. IRM cérébrale à prévoir si aggravation brutale."
        ),
        "spans": [
            ("SIM-FR-010", "PATIENT_ID", "present"),
            ("19 mars 2026", "DATE", "present"),
            ("migraines", "DIAGNOSIS", "historical"),
            ("céphalées", "SYMPTOM", "present"),
            ("photophobie", "SYMPTOM", "present"),
            ("nausées", "SYMPTOM", "present"),
            ("syndrome méningé", "DIAGNOSIS", "absent"),
            ("déficit moteur", "SYMPTOM", "absent"),
            ("sumatriptan", "MEDICATION", "present"),
            ("50 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("amitriptyline", "MEDICATION", "present"),
            ("25 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("le soir", "FREQUENCY", "present"),
            ("pendant 3 mois", "DURATION", "present"),
            ("IRM cérébrale", "PROCEDURE", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_fr_11",
        "language": "fr",
        "doc_type": "consultation",
        "document_date": "2025-12-11",
        "text": (
            "Suivi Endocrinologie | SIM-FR-011 | 2025-12-11\n"
            "Patient 55 ans suivi pour diabète de type 2 et dyslipidémie ; sœur atteinte d'hypothyroïdie. "
            "Bonne tolérance de la metformine 850 mg PO matin et soir ; aucun épisode d'hypoglycémie. "
            "HbA1c à 7,3 %, TSH 1,9 mUI/L écartant une dysthyroïdie. Ajout de dapagliflozine 10 mg PO 1 fois/jour."
        ),
        "spans": [
            ("SIM-FR-011", "PATIENT_ID", "present"),
            ("2025-12-11", "DATE", "present"),
            ("diabète de type 2", "DIAGNOSIS", "present"),
            ("dyslipidémie", "DIAGNOSIS", "present"),
            ("hypothyroïdie", "DIAGNOSIS", "family"),
            ("metformine", "MEDICATION", "present"),
            ("850 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("matin et soir", "FREQUENCY", "present"),
            ("hypoglycémie", "DIAGNOSIS", "absent"),
            ("HbA1c", "LAB_TEST", "present"),
            ("7,3", "LAB_VALUE", "present"),
            ("%", "LAB_UNIT", "present"),
            ("TSH", "LAB_TEST", "present"),
            ("1,9", "LAB_VALUE", "present"),
            ("mUI/L", "LAB_UNIT", "present"),
            ("dysthyroïdie", "DIAGNOSIS", "absent"),
            ("dapagliflozine", "MEDICATION", "present"),
            ("10 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("1 fois/jour", "FREQUENCY", "present"),
        ],
    },
    {
        "doc_id": "llm_fr_12",
        "language": "fr",
        "doc_type": "consultation",
        "document_date": "2026-04-04",
        "text": (
            "Consultation Gastro-entérologie — ID SIM-FR-012 — 04/04/2026\n"
            "Patiente 52 ans, pyrosis nocturne et régurgitations acides depuis 2 mois sans dysphagie ni amaigrissement. "
            "ATCD d'ulcère duodénal guéri en 2020. "
            "Test thérapeutique par ésoméprazole 40 mg PO 1 fois/jour pendant 6 semaines ; gastroscopie programmée pour éliminer une œsophagite peptique ou une gastrite chronique."
        ),
        "spans": [
            ("SIM-FR-012", "PATIENT_ID", "present"),
            ("04/04/2026", "DATE", "present"),
            ("pyrosis", "SYMPTOM", "present"),
            ("régurgitations acides", "SYMPTOM", "present"),
            ("depuis 2 mois", "DURATION", "present"),
            ("dysphagie", "SYMPTOM", "absent"),
            ("amaigrissement", "SYMPTOM", "absent"),
            ("ulcère duodénal", "DIAGNOSIS", "historical"),
            ("2020", "DATE", "present"),
            ("ésoméprazole", "MEDICATION", "present"),
            ("40 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present"),
            ("1 fois/jour", "FREQUENCY", "present"),
            ("pendant 6 semaines", "DURATION", "present"),
            ("gastroscopie", "PROCEDURE", "present"),
            ("œsophagite peptique", "DIAGNOSIS", "possible"),
            ("gastrite chronique", "DIAGNOSIS", "possible"),
        ],
    },
    {
        "doc_id": "llm_fr_13",
        "language": "fr",
        "doc_type": "consultation",
        "document_date": "2026-03-30",
        "text": (
            "Consultation Maladies Infectieuses | SIM-FR-013 | 30 mars 2026\n"
            "Patient 34 ans vu pour fièvre à 39,5 °C, frissons et céphalées apparus il y a 4 jours après un séjour en zone d'endémie. "
            "Goutte épaisse positive à Plasmodium falciparum, parasitémie 0,9 %. Absence d'ictère ou de confusion ; fièvre typhoïde peu probable. "
            "TTT ambulatoire par artéméther-luméfantrine PO 2x/j pendant 3 jours ; hospitalisation immédiate si vomissements."
        ),
        "spans": [
            ("SIM-FR-013", "PATIENT_ID", "present"),
            ("30 mars 2026", "DATE", "present"),
            ("fièvre", "SYMPTOM", "present", 0),
            ("frissons", "SYMPTOM", "present"),
            ("céphalées", "SYMPTOM", "present"),
            ("il y a 4 jours", "DATE", "present"),
            ("Goutte épaisse", "LAB_TEST", "present"),
            ("Plasmodium falciparum", "DIAGNOSIS", "present"),
            ("parasitémie", "LAB_TEST", "present"),
            ("0,9", "LAB_VALUE", "present"),
            ("%", "LAB_UNIT", "present"),
            ("ictère", "SYMPTOM", "absent"),
            ("confusion", "SYMPTOM", "absent"),
            ("fièvre typhoïde", "DIAGNOSIS", "absent"),
            ("artéméther-luméfantrine", "MEDICATION", "present"),
            ("PO", "ROUTE", "present"),
            ("2x/j", "FREQUENCY", "present"),
            ("pendant 3 jours", "DURATION", "present"),
            ("vomissements", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_fr_14",
        "language": "fr",
        "doc_type": "consultation",
        "document_date": "2026-05-09",
        "text": (
            "Consultation Néphrologie — SIM-FR-014 — 09/05/2026\n"
            "Patient 67 ans adressé pour progression d'une maladie rénale chronique : créatininémie 192 µmol/L, DFG 31 mL/min. "
            "Protéinurie à 1,4 g/24h. Pas d'hématurie macroscopique ni de douleurs lombaires. "
            "Néphropathie diabétique probable vs néphroangiosclérose ; maintien de l'irbésartan 150 mg PO 1 fois/jour et contrôle ionogramme dans 15 jours."
        ),
        "spans": [
            ("SIM-FR-014", "PATIENT_ID", "present"),
            ("09/05/2026", "DATE", "present"),
            ("maladie rénale chronique", "DIAGNOSIS", "present"),
            ("créatininémie", "LAB_TEST", "present"),
            ("192", "LAB_VALUE", "present"),
            ("µmol/L", "LAB_UNIT", "present"),
            ("DFG", "LAB_TEST", "present"),
            ("31", "LAB_VALUE", "present"),
            ("mL/min", "LAB_UNIT", "present"),
            ("Protéinurie", "LAB_TEST", "present"),
            ("1,4", "LAB_VALUE", "present"),
            ("g/24h", "LAB_UNIT", "present"),
            ("hématurie", "SYMPTOM", "absent"),
            ("douleurs lombaires", "SYMPTOM", "absent"),
            ("Néphropathie diabétique", "DIAGNOSIS", "possible"),
            ("néphroangiosclérose", "DIAGNOSIS", "possible"),
            ("irbésartan", "MEDICATION", "present"),
            ("150 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present"),
            ("1 fois/jour", "FREQUENCY", "present"),
            ("15 jours", "DURATION", "present"),
        ],
    },
    # --- LAB REPORTS (7) ---
    {
        "doc_id": "llm_fr_15",
        "language": "fr",
        "doc_type": "lab_report",
        "document_date": "2026-01-18",
        "text": (
            "BILAN BIOCHIMIQUE — Patient SIM-FR-015 — Prélèvement du 18/01/2026\n"
            "Glycémie à jeun : 1,72 g/L | HbA1c : 8,4 %\n"
            "LDL-cholestérol : 1,68 g/L | Triglycérides : 2,30 g/L | ALAT : 29 UI/L\n"
            "Conclusion : Déséquilibre glycémique et hypercholestérolémie ; absence de cytolyse hépatique."
        ),
        "spans": [
            ("SIM-FR-015", "PATIENT_ID", "present"),
            ("18/01/2026", "DATE", "present"),
            ("Glycémie à jeun", "LAB_TEST", "present"),
            ("1,72", "LAB_VALUE", "present"),
            ("g/L", "LAB_UNIT", "present", 0),
            ("HbA1c", "LAB_TEST", "present"),
            ("8,4", "LAB_VALUE", "present"),
            ("%", "LAB_UNIT", "present"),
            ("LDL-cholestérol", "LAB_TEST", "present"),
            ("1,68", "LAB_VALUE", "present"),
            ("g/L", "LAB_UNIT", "present", 1),
            ("Triglycérides", "LAB_TEST", "present"),
            ("2,30", "LAB_VALUE", "present"),
            ("g/L", "LAB_UNIT", "present", 2),
            ("ALAT", "LAB_TEST", "present"),
            ("29", "LAB_VALUE", "present"),
            ("UI/L", "LAB_UNIT", "present"),
            ("hypercholestérolémie", "DIAGNOSIS", "present"),
            ("cytolyse hépatique", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_fr_16",
        "language": "fr",
        "doc_type": "lab_report",
        "document_date": "2026-02-07",
        "text": (
            "NUMÉRATION FORMULE SANGUINE | ID SIM-FR-016 | 07/02/2026\n"
            "Hémoglobine : 9,1 g/dL | VGM : 73 fL | Plaquettes : 412 G/L\n"
            "Ferritinémie : 9 ng/mL | CRP : 3 mg/L\n"
            "Frottis sanguin : microcytose sans blaste circulant. Évoque une anémie ferriprive ; leucémie aiguë exclue."
        ),
        "spans": [
            ("SIM-FR-016", "PATIENT_ID", "present"),
            ("07/02/2026", "DATE", "present"),
            ("Hémoglobine", "LAB_TEST", "present"),
            ("9,1", "LAB_VALUE", "present"),
            ("g/dL", "LAB_UNIT", "present"),
            ("VGM", "LAB_TEST", "present"),
            ("73", "LAB_VALUE", "present"),
            ("fL", "LAB_UNIT", "present"),
            ("Plaquettes", "LAB_TEST", "present"),
            ("412", "LAB_VALUE", "present"),
            ("G/L", "LAB_UNIT", "present"),
            ("Ferritinémie", "LAB_TEST", "present"),
            ("9", "LAB_VALUE", "present"),
            ("ng/mL", "LAB_UNIT", "present"),
            ("CRP", "LAB_TEST", "present"),
            ("3", "LAB_VALUE", "present"),
            ("mg/L", "LAB_UNIT", "present"),
            ("anémie ferriprive", "DIAGNOSIS", "possible"),
            ("leucémie aiguë", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_fr_17",
        "language": "fr",
        "doc_type": "lab_report",
        "document_date": "2026-03-10",
        "text": (
            "BIOMARQUEURS CARDIAQUES — SIM-FR-017 — 2026-03-10\n"
            "Troponine Ic ultrasensible H0 : 18 ng/L\n"
            "Troponine Ic ultrasensible H3 : 245 ng/L\n"
            "BNP : 610 pg/mL | Kaliémie : 4,2 mmol/L\n"
            "Interprétation : Cinétique ascendante en faveur d'un syndrome coronarien aigu ; myocardite aiguë à discuter."
        ),
        "spans": [
            ("SIM-FR-017", "PATIENT_ID", "present"),
            ("2026-03-10", "DATE", "present"),
            ("Troponine Ic ultrasensible", "LAB_TEST", "present", 0),
            ("18", "LAB_VALUE", "present"),
            ("ng/L", "LAB_UNIT", "present", 0),
            ("Troponine Ic ultrasensible", "LAB_TEST", "present", 1),
            ("245", "LAB_VALUE", "present"),
            ("ng/L", "LAB_UNIT", "present", 1),
            ("BNP", "LAB_TEST", "present"),
            ("610", "LAB_VALUE", "present"),
            ("pg/mL", "LAB_UNIT", "present"),
            ("Kaliémie", "LAB_TEST", "present"),
            ("4,2", "LAB_VALUE", "present"),
            ("mmol/L", "LAB_UNIT", "present"),
            ("syndrome coronarien aigu", "DIAGNOSIS", "possible"),
            ("myocardite aiguë", "DIAGNOSIS", "possible"),
        ],
    },
    {
        "doc_id": "llm_fr_18",
        "language": "fr",
        "doc_type": "lab_report",
        "document_date": "2026-04-02",
        "text": (
            "BACTÉRIOLOGIE & BILAN INFECTIEUX | SIM-FR-018 | 02 avril 2026\n"
            "Procalcitonine : 7,4 ng/mL | Lactatémie : 3,1 mmol/L | Leucocytes : 18,4 G/L\n"
            "Hémocultures : positives à Klebsiella pneumoniae sensible à la pipéracilline-tazobactam et à l'amikacine.\n"
            "Tableau biologique de choc septique d'origine urinaire ; absence de candidémie."
        ),
        "spans": [
            ("SIM-FR-018", "PATIENT_ID", "present"),
            ("02 avril 2026", "DATE", "present"),
            ("Procalcitonine", "LAB_TEST", "present"),
            ("7,4", "LAB_VALUE", "present"),
            ("ng/mL", "LAB_UNIT", "present"),
            ("Lactatémie", "LAB_TEST", "present"),
            ("3,1", "LAB_VALUE", "present"),
            ("mmol/L", "LAB_UNIT", "present"),
            ("Leucocytes", "LAB_TEST", "present"),
            ("18,4", "LAB_VALUE", "present"),
            ("G/L", "LAB_UNIT", "present"),
            ("Hémocultures", "LAB_TEST", "present"),
            ("pipéracilline-tazobactam", "MEDICATION", "present"),
            ("amikacine", "MEDICATION", "present"),
            ("choc septique", "DIAGNOSIS", "present"),
            ("candidémie", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_fr_19",
        "language": "fr",
        "doc_type": "lab_report",
        "document_date": "2026-02-21",
        "text": (
            "BILAN THYROÏDIEN — Dossier SIM-FR-019 — 21/02/2026\n"
            "TSH : 0,01 mUI/L | T4 libre : 38,6 pmol/L | T3 libre : 13,1 pmol/L\n"
            "Anticorps anti-récepteurs de la TSH : 8,4 UI/L.\n"
            "Hyperthyroïdie périphérique franche très évocatrice d'une maladie de Basedow ; thyroïdite subaiguë peu probable."
        ),
        "spans": [
            ("SIM-FR-019", "PATIENT_ID", "present"),
            ("21/02/2026", "DATE", "present"),
            ("TSH", "LAB_TEST", "present", 0),
            ("0,01", "LAB_VALUE", "present"),
            ("mUI/L", "LAB_UNIT", "present"),
            ("T4 libre", "LAB_TEST", "present"),
            ("38,6", "LAB_VALUE", "present"),
            ("pmol/L", "LAB_UNIT", "present", 0),
            ("T3 libre", "LAB_TEST", "present"),
            ("13,1", "LAB_VALUE", "present"),
            ("pmol/L", "LAB_UNIT", "present", 1),
            ("Anticorps anti-récepteurs de la TSH", "LAB_TEST", "present"),
            ("8,4", "LAB_VALUE", "present"),
            ("UI/L", "LAB_UNIT", "present"),
            ("Hyperthyroïdie", "DIAGNOSIS", "present"),
            ("maladie de Basedow", "DIAGNOSIS", "possible"),
            ("thyroïdite subaiguë", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_fr_20",
        "language": "fr",
        "doc_type": "lab_report",
        "document_date": "2026-04-25",
        "text": (
            "BILAN HÉPATIQUE COMPLET | SIM-FR-020 | 2026-04-25\n"
            "ASAT : 290 UI/L | ALAT : 445 UI/L | GGT : 184 UI/L | Bilirubine totale : 42 µmol/L | TP : 78 %\n"
            "Cytolyse hépatique marquée sans insuffisance hépatocellulaire ; hépatite virale aiguë ou hépatite médicamenteuse à explorer."
        ),
        "spans": [
            ("SIM-FR-020", "PATIENT_ID", "present"),
            ("2026-04-25", "DATE", "present"),
            ("ASAT", "LAB_TEST", "present"),
            ("290", "LAB_VALUE", "present"),
            ("UI/L", "LAB_UNIT", "present", 0),
            ("ALAT", "LAB_TEST", "present"),
            ("445", "LAB_VALUE", "present"),
            ("UI/L", "LAB_UNIT", "present", 1),
            ("GGT", "LAB_TEST", "present"),
            ("184", "LAB_VALUE", "present"),
            ("UI/L", "LAB_UNIT", "present", 2),
            ("Bilirubine totale", "LAB_TEST", "present"),
            ("42", "LAB_VALUE", "present"),
            ("µmol/L", "LAB_UNIT", "present"),
            ("TP", "LAB_TEST", "present"),
            ("78", "LAB_VALUE", "present"),
            ("%", "LAB_UNIT", "present"),
            ("Cytolyse hépatique", "DIAGNOSIS", "present"),
            ("insuffisance hépatocellulaire", "DIAGNOSIS", "absent"),
            ("hépatite virale aiguë", "DIAGNOSIS", "possible"),
            ("hépatite médicamenteuse", "DIAGNOSIS", "possible"),
        ],
    },
    {
        "doc_id": "llm_fr_21",
        "language": "fr",
        "doc_type": "lab_report",
        "document_date": "2026-05-11",
        "text": (
            "HÉMOSTASE & D-DIMÈRES — SIM-FR-021 — 11 mai 2026\n"
            "D-dimères : 2390 ng/mL | Fibrinogène : 3,9 g/L | TCA : 31 s | Plaquettes : 245 G/L\n"
            "Élévation franche des D-dimères justifiant un angioscanner thoracique pour éliminer une embolie pulmonaire ; CIVD écartée."
        ),
        "spans": [
            ("SIM-FR-021", "PATIENT_ID", "present"),
            ("11 mai 2026", "DATE", "present"),
            ("D-dimères", "LAB_TEST", "present", 0),
            ("2390", "LAB_VALUE", "present"),
            ("ng/mL", "LAB_UNIT", "present"),
            ("Fibrinogène", "LAB_TEST", "present"),
            ("3,9", "LAB_VALUE", "present"),
            ("g/L", "LAB_UNIT", "present"),
            ("TCA", "LAB_TEST", "present"),
            ("31", "LAB_VALUE", "present"),
            ("s", "LAB_UNIT", "present"),
            ("Plaquettes", "LAB_TEST", "present"),
            ("245", "LAB_VALUE", "present"),
            ("G/L", "LAB_UNIT", "present"),
            ("angioscanner thoracique", "PROCEDURE", "present"),
            ("embolie pulmonaire", "DIAGNOSIS", "possible"),
            ("CIVD", "DIAGNOSIS", "absent"),
        ],
    },
    # --- PRESCRIPTIONS (7) ---
    {
        "doc_id": "llm_fr_22",
        "language": "fr",
        "doc_type": "prescription",
        "document_date": "2026-01-29",
        "text": (
            "ORDONNANCE MÉDICALE | Patient : SIM-FR-022 | Date : 29/01/2026\n"
            "Motif : sinusite maxillaire aiguë bactérienne.\n"
            "1) Amoxicilline-acide clavulanique 1000 mg PO 3x/j pendant 7 jours.\n"
            "2) Mométasone 50 µg par voie nasale 2x/j pendant 14 jours.\n"
            "3) Paracétamol 1000 mg PO si douleur faciale. Arrêter l'antibiotique en cas d'urticaire."
        ),
        "spans": [
            ("SIM-FR-022", "PATIENT_ID", "present"),
            ("29/01/2026", "DATE", "present"),
            ("sinusite maxillaire aiguë", "DIAGNOSIS", "present"),
            ("Amoxicilline-acide clavulanique", "MEDICATION", "present"),
            ("1000 mg", "DOSAGE", "present", 0),
            ("PO", "ROUTE", "present", 0),
            ("3x/j", "FREQUENCY", "present"),
            ("pendant 7 jours", "DURATION", "present"),
            ("Mométasone", "MEDICATION", "present"),
            ("50 µg", "DOSAGE", "present"),
            ("par voie nasale", "ROUTE", "present"),
            ("2x/j", "FREQUENCY", "present"),
            ("pendant 14 jours", "DURATION", "present"),
            ("Paracétamol", "MEDICATION", "present"),
            ("1000 mg", "DOSAGE", "present", 1),
            ("PO", "ROUTE", "present", 1),
            ("douleur faciale", "SYMPTOM", "hypothetical"),
            ("urticaire", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_fr_23",
        "language": "fr",
        "doc_type": "prescription",
        "document_date": "2026-02-10",
        "text": (
            "Ordonnance Pneumologie — SIM-FR-023 — 10 février 2026\n"
            "Indication : asthme persistant modéré.\n"
            "- Budésonide-formotérol 200/6 µg par voie inhalée matin et soir pendant 3 mois.\n"
            "- Salbutamol 100 µg par voie inhalée si sifflements ou dyspnée.\n"
            "- Montélukast 10 mg PO le soir pendant 90 jours."
        ),
        "spans": [
            ("SIM-FR-023", "PATIENT_ID", "present"),
            ("10 février 2026", "DATE", "present"),
            ("asthme", "DIAGNOSIS", "present"),
            ("Budésonide-formotérol", "MEDICATION", "present"),
            ("200/6 µg", "DOSAGE", "present"),
            ("par voie inhalée", "ROUTE", "present", 0),
            ("matin et soir", "FREQUENCY", "present"),
            ("pendant 3 mois", "DURATION", "present"),
            ("Salbutamol", "MEDICATION", "present"),
            ("100 µg", "DOSAGE", "present"),
            ("par voie inhalée", "ROUTE", "present", 1),
            ("sifflements", "SYMPTOM", "hypothetical"),
            ("dyspnée", "SYMPTOM", "hypothetical"),
            ("Montélukast", "MEDICATION", "present"),
            ("10 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present"),
            ("le soir", "FREQUENCY", "present"),
            ("pendant 90 jours", "DURATION", "present"),
        ],
    },
    {
        "doc_id": "llm_fr_24",
        "language": "fr",
        "doc_type": "prescription",
        "document_date": "2026-03-05",
        "text": (
            "Ordonnance de sortie | ID SIM-FR-024 | 05/03/2026\n"
            "Pour thrombose veineuse profonde de la veine poplitée droite :\n"
            "Rivaroxaban 15 mg PO 2x/j pendant 21 jours puis rivaroxaban 20 mg PO 1 fois/jour pendant 6 mois.\n"
            "Contre-indication à l'ibuprofène ; consulter en urgence si hématurie ou méléna."
        ),
        "spans": [
            ("SIM-FR-024", "PATIENT_ID", "present"),
            ("05/03/2026", "DATE", "present"),
            ("thrombose veineuse profonde", "DIAGNOSIS", "present"),
            ("veine poplitée droite", "ANATOMY", "present"),
            ("Rivaroxaban", "MEDICATION", "present"),
            ("15 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("2x/j", "FREQUENCY", "present"),
            ("pendant 21 jours", "DURATION", "present"),
            ("rivaroxaban", "MEDICATION", "present"),
            ("20 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("1 fois/jour", "FREQUENCY", "present"),
            ("pendant 6 mois", "DURATION", "present"),
            ("ibuprofène", "MEDICATION", "absent"),
            ("hématurie", "SYMPTOM", "hypothetical"),
            ("méléna", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_fr_25",
        "language": "fr",
        "doc_type": "prescription",
        "document_date": "2026-03-23",
        "text": (
            "Ordonnance Dermatologie — SIM-FR-025 — 23/03/2026\n"
            "Diagnostic : poussée de dermatite atopique surinfectée.\n"
            "1. Pristinamycine 1000 mg PO 2x/j pendant 7 jours.\n"
            "2. Bétaméthasone 0,05% par voie cutanée 1 fois/jour pendant 14 jours sur les avant-bras.\n"
            "3. Cétirizine 10 mg PO le soir si prurit."
        ),
        "spans": [
            ("SIM-FR-025", "PATIENT_ID", "present"),
            ("23/03/2026", "DATE", "present"),
            ("dermatite atopique", "DIAGNOSIS", "present"),
            ("Pristinamycine", "MEDICATION", "present"),
            ("1000 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("2x/j", "FREQUENCY", "present"),
            ("pendant 7 jours", "DURATION", "present"),
            ("Bétaméthasone", "MEDICATION", "present"),
            ("0,05%", "DOSAGE", "present"),
            ("par voie cutanée", "ROUTE", "present"),
            ("1 fois/jour", "FREQUENCY", "present"),
            ("pendant 14 jours", "DURATION", "present"),
            ("avant-bras", "ANATOMY", "present"),
            ("Cétirizine", "MEDICATION", "present"),
            ("10 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("le soir", "FREQUENCY", "present"),
            ("prurit", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_fr_26",
        "language": "fr",
        "doc_type": "prescription",
        "document_date": "2026-04-08",
        "text": (
            "Prescription Psychiatrie | SIM-FR-026 | 2026-04-08\n"
            "Indication : épisode dépressif caractérisé et insomnie ; absence d'idées suicidaires.\n"
            "- Escitalopram 10 mg PO le matin pendant 6 mois.\n"
            "- Hydroxyzine 25 mg PO le soir si anxiété aiguë pendant 14 j."
        ),
        "spans": [
            ("SIM-FR-026", "PATIENT_ID", "present"),
            ("2026-04-08", "DATE", "present"),
            ("épisode dépressif caractérisé", "DIAGNOSIS", "present"),
            ("insomnie", "SYMPTOM", "present"),
            ("idées suicidaires", "SYMPTOM", "absent"),
            ("Escitalopram", "MEDICATION", "present"),
            ("10 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("le matin", "FREQUENCY", "present"),
            ("pendant 6 mois", "DURATION", "present"),
            ("Hydroxyzine", "MEDICATION", "present"),
            ("25 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("le soir", "FREQUENCY", "present"),
            ("anxiété", "SYMPTOM", "hypothetical"),
            ("pendant 14 j", "DURATION", "present"),
        ],
    },
    {
        "doc_id": "llm_fr_27",
        "language": "fr",
        "doc_type": "prescription",
        "document_date": "2026-04-18",
        "text": (
            "Ordonnance Urologie — SIM-FR-027 — 18 avril 2026\n"
            "Cystite aiguë simple :\n"
            "Fosfomycine-trométamol 3 g PO en prise unique.\n"
            "Phloroglucinol 80 mg PO 3x/j pendant 3 jours pour dysurie.\n"
            "Reconsulter rapidement si apparition de fièvre ou de lombalgie."
        ),
        "spans": [
            ("SIM-FR-027", "PATIENT_ID", "present"),
            ("18 avril 2026", "DATE", "present"),
            ("Cystite aiguë", "DIAGNOSIS", "present"),
            ("Fosfomycine-trométamol", "MEDICATION", "present"),
            ("3 g", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("Phloroglucinol", "MEDICATION", "present"),
            ("80 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("3x/j", "FREQUENCY", "present"),
            ("pendant 3 jours", "DURATION", "present"),
            ("dysurie", "SYMPTOM", "present"),
            ("fièvre", "SYMPTOM", "hypothetical"),
            ("lombalgie", "SYMPTOM", "hypothetical"),
        ],
    },
    {
        "doc_id": "llm_fr_28",
        "language": "fr",
        "doc_type": "prescription",
        "document_date": "2026-05-04",
        "text": (
            "Renouvellement Médecine Interne | SIM-FR-028 | 04/05/2026\n"
            "Pathologies : hypertension artérielle essentielle et goutte.\n"
            "1. Amlodipine 10 mg PO 1 fois/jour pendant 3 mois.\n"
            "2. Allopurinol 200 mg PO 1 fois/jour pendant 90 jours.\n"
            "3. Colchicine 0,5 mg PO 1 fois/jour pendant 4 semaines."
        ),
        "spans": [
            ("SIM-FR-028", "PATIENT_ID", "present"),
            ("04/05/2026", "DATE", "present"),
            ("hypertension artérielle essentielle", "DIAGNOSIS", "present"),
            ("goutte", "DIAGNOSIS", "present"),
            ("Amlodipine", "MEDICATION", "present"),
            ("10 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("1 fois/jour", "FREQUENCY", "present", 0),
            ("pendant 3 mois", "DURATION", "present"),
            ("Allopurinol", "MEDICATION", "present"),
            ("200 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("1 fois/jour", "FREQUENCY", "present", 1),
            ("pendant 90 jours", "DURATION", "present"),
            ("Colchicine", "MEDICATION", "present"),
            ("0,5 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 2),
            ("1 fois/jour", "FREQUENCY", "present", 2),
            ("pendant 4 semaines", "DURATION", "present"),
        ],
    },
    # --- RADIOLOGY REPORTS (6) ---
    {
        "doc_id": "llm_fr_29",
        "language": "fr",
        "doc_type": "radiology_report",
        "document_date": "2026-01-13",
        "text": (
            "ANGIOSCANNER THORACIQUE | SIM-FR-029 | 13/01/2026\n"
            "Indication : douleur basithoracique droite et tachycardie.\n"
            "Résultats : Défaut de remplissage endoluminal de l'artère pulmonaire gauche et des branches lobaires du lobe inférieur gauche. Lame d'épanchement pleural gauche.\n"
            "Pas de dissection aortique ni de pneumothorax. Conclusion : Embolie pulmonaire aiguë ; cœur pulmonaire aigu non exclu."
        ),
        "spans": [
            ("SIM-FR-029", "PATIENT_ID", "present"),
            ("13/01/2026", "DATE", "present"),
            ("douleur basithoracique", "SYMPTOM", "present"),
            ("tachycardie", "SYMPTOM", "present"),
            ("artère pulmonaire gauche", "ANATOMY", "present"),
            ("lobe inférieur gauche", "ANATOMY", "present"),
            ("épanchement pleural", "DIAGNOSIS", "present"),
            ("dissection aortique", "DIAGNOSIS", "absent"),
            ("pneumothorax", "DIAGNOSIS", "absent"),
            ("Embolie pulmonaire aiguë", "DIAGNOSIS", "present"),
            ("cœur pulmonaire aigu", "DIAGNOSIS", "possible"),
        ],
    },
    {
        "doc_id": "llm_fr_30",
        "language": "fr",
        "doc_type": "radiology_report",
        "document_date": "2026-02-08",
        "text": (
            "IRM CÉRÉBRALE AVEC INJECTION — SIM-FR-030 — 08 février 2026\n"
            "Motif : diplopie et céphalées matinales depuis 2 semaines.\n"
            "Résultats : Lésion extra-axiale rehaussée de 2,1 cm de la convexité pariétale gauche avec discret œdème périlésionnel ; absence d'infarctus cérébral ou d'engagement.\n"
            "Conclusion : Aspect en faveur d'un méningiome ; métastase durale moins probable."
        ),
        "spans": [
            ("SIM-FR-030", "PATIENT_ID", "present"),
            ("08 février 2026", "DATE", "present"),
            ("diplopie", "SYMPTOM", "present"),
            ("céphalées", "SYMPTOM", "present"),
            ("depuis 2 semaines", "DURATION", "present"),
            ("convexité pariétale gauche", "ANATOMY", "present"),
            ("œdème périlésionnel", "DIAGNOSIS", "present"),
            ("infarctus cérébral", "DIAGNOSIS", "absent"),
            ("méningiome", "DIAGNOSIS", "possible"),
            ("métastase durale", "DIAGNOSIS", "possible"),
        ],
    },
    {
        "doc_id": "llm_fr_31",
        "language": "fr",
        "doc_type": "radiology_report",
        "document_date": "2026-03-02",
        "text": (
            "ÉCHOGRAPHIE ABDOMINALE | SIM-FR-031 | 02/03/2026\n"
            "Indication : colique hépatique et vomissements.\n"
            "Résultats : Vésicule biliaire distendue à paroi épaissie (6 mm) contenant plusieurs calculs avec signe de Murphy échographique. Voie biliaire principale fine sans lithiase choledocienne.\n"
            "Conclusion : Cholécystite aiguë lithiasique ; pas de stéatose hépatique."
        ),
        "spans": [
            ("SIM-FR-031", "PATIENT_ID", "present"),
            ("02/03/2026", "DATE", "present"),
            ("colique hépatique", "SYMPTOM", "present"),
            ("vomissements", "SYMPTOM", "present"),
            ("Vésicule biliaire", "ANATOMY", "present"),
            ("Voie biliaire principale", "ANATOMY", "present"),
            ("lithiase choledocienne", "DIAGNOSIS", "absent"),
            ("Cholécystite aiguë lithiasique", "DIAGNOSIS", "present"),
            ("stéatose hépatique", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_fr_32",
        "language": "fr",
        "doc_type": "radiology_report",
        "document_date": "2026-03-26",
        "text": (
            "IRM DU RACHIS LOMBAIRE — SIM-FR-032 — 2026-03-26\n"
            "Clinique : lombosciatique droite hyperalgique depuis 6 semaines avec paresthésies.\n"
            "Résultats : Hernie discale paramédiane droite en L5-S1 comprimant la racine S1 droite. Absence de syndrome de la queue de cheval, de spondylodiscite ou de tassement vertébral."
        ),
        "spans": [
            ("SIM-FR-032", "PATIENT_ID", "present"),
            ("2026-03-26", "DATE", "present"),
            ("lombosciatique", "SYMPTOM", "present"),
            ("depuis 6 semaines", "DURATION", "present"),
            ("paresthésies", "SYMPTOM", "present"),
            ("Hernie discale", "DIAGNOSIS", "present"),
            ("L5-S1", "ANATOMY", "present"),
            ("racine S1 droite", "ANATOMY", "present"),
            ("syndrome de la queue de cheval", "DIAGNOSIS", "absent"),
            ("spondylodiscite", "DIAGNOSIS", "absent"),
            ("tassement vertébral", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_fr_33",
        "language": "fr",
        "doc_type": "radiology_report",
        "document_date": "2026-04-13",
        "text": (
            "RADIOGRAPHIE THORACIQUE | SIM-FR-033 | 13 avril 2026\n"
            "Indication : toux sèche persistante depuis 4 semaines chez un tabagique sevré.\n"
            "Résultats : Opacité nodulaire spiculée de 2 cm du lobe supérieur droit ; adénopathie hilaire possible. Pas d'épanchement pleural ni de cardiomégalie.\n"
            "Conduite : Scanner thoracique injecté pour éliminer un carcinome bronchique ou une sarcoïdose."
        ),
        "spans": [
            ("SIM-FR-033", "PATIENT_ID", "present"),
            ("13 avril 2026", "DATE", "present"),
            ("toux sèche", "SYMPTOM", "present"),
            ("depuis 4 semaines", "DURATION", "present"),
            ("lobe supérieur droit", "ANATOMY", "present"),
            ("adénopathie hilaire", "DIAGNOSIS", "possible"),
            ("épanchement pleural", "DIAGNOSIS", "absent"),
            ("cardiomégalie", "DIAGNOSIS", "absent"),
            ("Scanner thoracique injecté", "PROCEDURE", "present"),
            ("carcinome bronchique", "DIAGNOSIS", "possible"),
            ("sarcoïdose", "DIAGNOSIS", "possible"),
        ],
    },
    {
        "doc_id": "llm_fr_34",
        "language": "fr",
        "doc_type": "radiology_report",
        "document_date": "2026-05-03",
        "text": (
            "IRM DU GENOU GAUCHE — SIM-FR-034 — 03/05/2026\n"
            "Contexte : entorse sportive il y a 5 jours avec hémarthrose.\n"
            "Résultats : Rupture complète du ligament croisé antérieur et lésion en anse de seau du ménisque interne ; contusion osseuse du condyle fémoral externe.\n"
            "Ligament croisé postérieur intact ; pas de fracture du plateau tibial."
        ),
        "spans": [
            ("SIM-FR-034", "PATIENT_ID", "present"),
            ("03/05/2026", "DATE", "present"),
            ("il y a 5 jours", "DATE", "present"),
            ("hémarthrose", "SYMPTOM", "present"),
            ("ligament croisé antérieur", "ANATOMY", "present"),
            ("ménisque interne", "ANATOMY", "present"),
            ("contusion osseuse", "DIAGNOSIS", "present"),
            ("condyle fémoral externe", "ANATOMY", "present"),
            ("Ligament croisé postérieur", "ANATOMY", "present"),
            ("fracture du plateau tibial", "DIAGNOSIS", "absent"),
        ],
    },
    # --- OPERATIVE NOTES (6) ---
    {
        "doc_id": "llm_fr_35",
        "language": "fr",
        "doc_type": "operative_note",
        "document_date": "2026-01-24",
        "text": (
            "COMPTE RENDU OPÉRATOIRE | SIM-FR-035 | 24/01/2026\n"
            "Diagnostic : lithiase vésiculaire symptomatique.\n"
            "Intervention : Cholécystectomie cœlioscopique sous anesthésie générale. Antibioprophylaxie par céfazoline 2 g IV.\n"
            "Dissection du canal cystique et de l'artère cystique sans plaie des voies biliaires ni hémorragie."
        ),
        "spans": [
            ("SIM-FR-035", "PATIENT_ID", "present"),
            ("24/01/2026", "DATE", "present"),
            ("lithiase vésiculaire", "DIAGNOSIS", "present"),
            ("Cholécystectomie cœlioscopique", "PROCEDURE", "present"),
            ("céfazoline", "MEDICATION", "present"),
            ("2 g", "DOSAGE", "present"),
            ("IV", "ROUTE", "present"),
            ("canal cystique", "ANATOMY", "present"),
            ("plaie des voies biliaires", "DIAGNOSIS", "absent"),
            ("hémorragie", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_fr_36",
        "language": "fr",
        "doc_type": "operative_note",
        "document_date": "2026-02-19",
        "text": (
            "CRO CHIRURGIE ORTHOPÉDIQUE — SIM-FR-036 — 19 février 2026\n"
            "Indication : fracture déplacée du col fémoral gauche.\n"
            "Geste : Prothèse intermédiaire de hanche gauche cimentée. Acide tranexamique 1 g IV à l'induction.\n"
            "Stabilité vérifiée ; pas de fracture périprothétique ni de lésion du nerf sciatique. Post-op : énoxaparine 40 mg SC 1 fois/jour pendant 30 jours."
        ),
        "spans": [
            ("SIM-FR-036", "PATIENT_ID", "present"),
            ("19 février 2026", "DATE", "present"),
            ("fracture déplacée du col fémoral gauche", "DIAGNOSIS", "present"),
            ("Prothèse intermédiaire de hanche gauche", "PROCEDURE", "present"),
            ("Acide tranexamique", "MEDICATION", "present"),
            ("1 g", "DOSAGE", "present"),
            ("IV", "ROUTE", "present"),
            ("fracture périprothétique", "DIAGNOSIS", "absent"),
            ("nerf sciatique", "ANATOMY", "present"),
            ("énoxaparine", "MEDICATION", "present"),
            ("40 mg", "DOSAGE", "present"),
            ("SC", "ROUTE", "present"),
            ("1 fois/jour", "FREQUENCY", "present"),
            ("pendant 30 jours", "DURATION", "present"),
        ],
    },
    {
        "doc_id": "llm_fr_37",
        "language": "fr",
        "doc_type": "operative_note",
        "document_date": "2026-03-15",
        "text": (
            "CORONAROGRAPHIE ET ANGIOPLASTIE | SIM-FR-037 | 2026-03-15\n"
            "Indication : SCA ST+ antérieur.\n"
            "Procédure : Coronarographie par voie radiale droite montrant une occlusion aiguë de l'artère interventriculaire antérieure proximale. "
            "Angioplastie primaire avec pose d'un stent actif, flux TIMI 3 obtenu. Héparine IV administrée ; absence de dissection coronaire ou de tamponnade."
        ),
        "spans": [
            ("SIM-FR-037", "PATIENT_ID", "present"),
            ("2026-03-15", "DATE", "present"),
            ("SCA ST+", "DIAGNOSIS", "present"),
            ("Coronarographie", "PROCEDURE", "present"),
            ("artère interventriculaire antérieure proximale", "ANATOMY", "present"),
            ("Angioplastie primaire", "PROCEDURE", "present"),
            ("Héparine", "MEDICATION", "present"),
            ("IV", "ROUTE", "present"),
            ("dissection coronaire", "DIAGNOSIS", "absent"),
            ("tamponnade", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_fr_38",
        "language": "fr",
        "doc_type": "operative_note",
        "document_date": "2026-04-07",
        "text": (
            "CRO OBSTÉTRIQUE — SIM-FR-038 — 07 avril 2026\n"
            "Motif : stagnation de la dilatation à 40 semaines d'aménorrhée.\n"
            "Intervention : Césarienne segmentaire transverse sous rachianesthésie. Oxytocine 10 UI IV après extraction et céfazoline 2 g IV.\n"
            "Utérus bien rétracté ; pas d'hémorragie du post-partum ni de plaie vésicale."
        ),
        "spans": [
            ("SIM-FR-038", "PATIENT_ID", "present"),
            ("07 avril 2026", "DATE", "present"),
            ("40 semaines", "DURATION", "present"),
            ("Césarienne", "PROCEDURE", "present"),
            ("rachianesthésie", "PROCEDURE", "present"),
            ("Oxytocine", "MEDICATION", "present"),
            ("10 UI", "DOSAGE", "present"),
            ("IV", "ROUTE", "present", 0),
            ("céfazoline", "MEDICATION", "present"),
            ("2 g", "DOSAGE", "present"),
            ("IV", "ROUTE", "present", 1),
            ("Utérus", "ANATOMY", "present"),
            ("hémorragie du post-partum", "DIAGNOSIS", "absent"),
            ("plaie vésicale", "DIAGNOSIS", "absent"),
        ],
    },
    {
        "doc_id": "llm_fr_39",
        "language": "fr",
        "doc_type": "operative_note",
        "document_date": "2026-04-22",
        "text": (
            "CRO CHIRURGIE ORL | SIM-FR-039 | 22/04/2026\n"
            "Diagnostic : goitre multinodulaire compressif.\n"
            "Intervention : Thyroïdectomie totale avec neuromonitoring des nerfs récurrents. Glandes parathyroïdes préservées ; absence d'hématome cervical post-opératoire. "
            "Relais par lévothyroxine 100 µg PO le matin et carbonate de calcium 1000 mg PO 2x/j."
        ),
        "spans": [
            ("SIM-FR-039", "PATIENT_ID", "present"),
            ("22/04/2026", "DATE", "present"),
            ("goitre multinodulaire", "DIAGNOSIS", "present"),
            ("Thyroïdectomie totale", "PROCEDURE", "present"),
            ("nerfs récurrents", "ANATOMY", "present"),
            ("Glandes parathyroïdes", "ANATOMY", "present"),
            ("hématome cervical", "DIAGNOSIS", "absent"),
            ("lévothyroxine", "MEDICATION", "present"),
            ("100 µg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 0),
            ("le matin", "FREQUENCY", "present"),
            ("carbonate de calcium", "MEDICATION", "present"),
            ("1000 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present", 1),
            ("2x/j", "FREQUENCY", "present"),
        ],
    },
    {
        "doc_id": "llm_fr_40",
        "language": "fr",
        "doc_type": "operative_note",
        "document_date": "2026-05-11",
        "text": (
            "CRO CHIRURGIE VASCULAIRE — SIM-FR-040 — 11 mai 2026\n"
            "Diagnostic : sténose serrée symptomatique de l'artère carotide interne gauche (80 %).\n"
            "Intervention : Endartériectomie carotidienne gauche avec patch péricardique. Héparine 5000 UI IV avant clampage.\n"
            "Réveil sans déficit neurologique, sans AVC péri-opératoire ni paralysie faciale. Poursuite aspirine 75 mg PO 1 fois/jour."
        ),
        "spans": [
            ("SIM-FR-040", "PATIENT_ID", "present"),
            ("11 mai 2026", "DATE", "present"),
            ("sténose", "DIAGNOSIS", "present"),
            ("artère carotide interne gauche", "ANATOMY", "present"),
            ("Endartériectomie carotidienne", "PROCEDURE", "present"),
            ("Héparine", "MEDICATION", "present"),
            ("5000 UI", "DOSAGE", "present"),
            ("IV", "ROUTE", "present"),
            ("déficit neurologique", "SYMPTOM", "absent"),
            ("AVC", "DIAGNOSIS", "absent"),
            ("paralysie faciale", "DIAGNOSIS", "absent"),
            ("aspirine", "MEDICATION", "present"),
            ("75 mg", "DOSAGE", "present"),
            ("PO", "ROUTE", "present"),
            ("1 fois/jour", "FREQUENCY", "present"),
        ],
    },
]
