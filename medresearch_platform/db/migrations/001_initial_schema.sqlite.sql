-- Synchronized SQLite Transactional Schema for Local Sandbox Execution (ADR-0002)

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL,
    checksum_sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS institutions (
    institution_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    country_code TEXT NOT NULL,
    governance_status TEXT NOT NULL CHECK (governance_status IN ('ACTIVE', 'PENDING_APPROVAL', 'SUSPENDED', 'WITHDRAWN')),
    data_use_agreement_ref TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    institution_id TEXT NOT NULL REFERENCES institutions(institution_id),
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('RESEARCHER', 'ANNOTATOR', 'ADJUDICATOR', 'ML_ENGINEER', 'INSTITUTION_ADMIN', 'AUDITOR')),
    permissions_json TEXT NOT NULL,
    qualification_level TEXT NOT NULL DEFAULT 'STANDARD',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    institution_id TEXT NOT NULL REFERENCES institutions(institution_id),
    title TEXT NOT NULL,
    protocol_version TEXT NOT NULL,
    ethics_approval_ref TEXT,
    permitted_purpose TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('DRAFT', 'APPROVED', 'ACTIVE', 'ARCHIVED', 'SUSPENDED')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    institution_id TEXT NOT NULL REFERENCES institutions(institution_id),
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('SYNTHETIC', 'PUBLIC', 'RESTRICTED', 'INSTITUTIONAL', 'REFERENCE')),
    license TEXT NOT NULL,
    consent_basis TEXT NOT NULL,
    permitted_purpose TEXT NOT NULL,
    deidentification_status TEXT NOT NULL CHECK (deidentification_status IN ('SYNTHETIC_NO_PHI', 'PSEUDONYMIZED', 'DEIDENTIFIED_VERIFIED', 'RAW_RESTRICTED')),
    languages_json TEXT NOT NULL,
    modalities_json TEXT NOT NULL,
    record_count INTEGER NOT NULL DEFAULT 0,
    manifest_sha256 TEXT NOT NULL,
    retention_policy TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS subjects (
    subject_id TEXT PRIMARY KEY,
    institution_id TEXT NOT NULL REFERENCES institutions(institution_id),
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    pseudonym_code TEXT NOT NULL UNIQUE,
    date_shift_days INTEGER NOT NULL DEFAULT 0,
    language_primary TEXT NOT NULL,
    age_group TEXT,
    consent_status TEXT NOT NULL CHECK (consent_status IN ('SYNTHETIC_EXEMPT', 'ACTIVE_CONSENT', 'WITHDRAWN')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS encounters (
    encounter_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES subjects(subject_id),
    institution_id TEXT NOT NULL REFERENCES institutions(institution_id),
    encounter_type TEXT NOT NULL,
    specialty TEXT NOT NULL,
    encounter_date_iso TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_families (
    family_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES subjects(subject_id),
    encounter_id TEXT REFERENCES encounters(encounter_id),
    institution_id TEXT NOT NULL REFERENCES institutions(institution_id),
    split_assignment TEXT NOT NULL CHECK (split_assignment IN ('train', 'val', 'locked_test', 'unassigned')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES document_families(family_id),
    subject_id TEXT NOT NULL REFERENCES subjects(subject_id),
    encounter_id TEXT REFERENCES encounters(encounter_id),
    institution_id TEXT NOT NULL REFERENCES institutions(institution_id),
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    original_filename TEXT NOT NULL,
    file_format TEXT NOT NULL CHECK (file_format IN ('TXT', 'PDF', 'DOCX', 'PNG', 'JPEG', 'TIFF', 'WEBP', 'DICOM')),
    content_sha256 TEXT NOT NULL,
    storage_object_path TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 1,
    detected_language TEXT NOT NULL CHECK (detected_language IN ('en', 'fr', 'mixed_en_fr', 'unknown')),
    document_type TEXT NOT NULL,
    specialty TEXT NOT NULL,
    extraction_mode TEXT NOT NULL CHECK (extraction_mode IN ('NATIVE_TEXT', 'TESSERACT_OCR', 'CUSTOM_OCR', 'HYBRID')),
    untrusted_flags_json TEXT NOT NULL DEFAULT '[]',
    deidentified_preview TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_pages (
    page_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    width_px INTEGER NOT NULL DEFAULT 600,
    height_px INTEGER NOT NULL DEFAULT 800,
    raw_text TEXT NOT NULL,
    extraction_source TEXT NOT NULL CHECK (extraction_source IN ('NATIVE_PARSER', 'TESSERACT_OCR', 'CUSTOM_OCR_MODEL')),
    mean_confidence REAL NOT NULL DEFAULT 1.0,
    mean_uncertainty REAL NOT NULL DEFAULT 0.0,
    UNIQUE (document_id, page_number)
);

CREATE TABLE IF NOT EXISTS document_sections (
    section_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    section_type TEXT NOT NULL,
    heading_text TEXT NOT NULL,
    body_text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    bbox_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ocr_blocks (
    ocr_block_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    block_index INTEGER NOT NULL,
    recognized_text TEXT NOT NULL,
    ground_truth_text TEXT,
    bbox_json TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    source_stage TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL,
    aleatoric_uncertainty REAL NOT NULL,
    epistemic_uncertainty REAL NOT NULL,
    review_status TEXT NOT NULL CHECK (review_status IN ('AI_PREDICTED', 'HUMAN_CORRECTED', 'APPROVED_GROUND_TRUTH', 'AMBIGUOUS'))
);

CREATE TABLE IF NOT EXISTS terminology_concepts (
    concept_id TEXT PRIMARY KEY,
    vocabulary TEXT NOT NULL CHECK (vocabulary IN ('ICD-10', 'SNOMED-CT', 'LOINC', 'RXNORM', 'ATC', 'LOCAL-MED')),
    code TEXT NOT NULL,
    preferred_term_en TEXT NOT NULL,
    preferred_term_fr TEXT NOT NULL,
    synonyms_en_json TEXT NOT NULL,
    synonyms_fr_json TEXT NOT NULL,
    concept_category TEXT NOT NULL,
    version TEXT NOT NULL,
    UNIQUE (vocabulary, code, version)
);

CREATE TABLE IF NOT EXISTS extracted_entities (
    entity_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    section_type TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK (entity_type IN (
        'SYMPTOM', 'DIAGNOSIS', 'MEDICATION', 'DOSAGE', 'FREQUENCY', 'ROUTE', 'DURATION',
        'LAB_TEST', 'LAB_VALUE', 'LAB_UNIT', 'PROCEDURE', 'ANATOMY', 'TEMPORAL_EXPR',
        'PATIENT_ID', 'DATE'
    )),
    source_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    terminology_concept_id TEXT REFERENCES terminology_concepts(concept_id),
    mapping_match_type TEXT NOT NULL CHECK (mapping_match_type IN ('EXACT', 'SYNONYM', 'APPROXIMATE', 'AMBIGUOUS', 'UNMAPPED')),
    assertion_status TEXT NOT NULL CHECK (assertion_status IN (
        'AFFIRMED', 'NEGATED', 'UNCERTAIN', 'HYPOTHETICAL', 'CONDITIONAL', 'FAMILY_HISTORY'
    )),
    temporal_status TEXT NOT NULL CHECK (temporal_status IN ('CURRENT', 'HISTORICAL', 'PLANNED', 'RESOLVED', 'UNKNOWN')),
    medication_status TEXT NOT NULL CHECK (medication_status IN (
        'NOT_APPLICABLE', 'PRESCRIBED', 'DISPENSED', 'TAKEN', 'ADMINISTERED', 'DISCONTINUED', 'UNKNOWN'
    )),
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    bbox_json TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    source_stage TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL,
    uncertainty REAL NOT NULL,
    abstained INTEGER NOT NULL DEFAULT 0,
    review_status TEXT NOT NULL CHECK (review_status IN ('AI_PREDICTED', 'needs_review', 'HUMAN_CORRECTED', 'APPROVED_GROUND_TRUTH', 'REJECTED', 'UNRESOLVED_DISAGREEMENT')),
    review_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extracted_relations (
    relation_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    source_entity_id TEXT NOT NULL REFERENCES extracted_entities(entity_id) ON DELETE CASCADE,
    target_entity_id TEXT NOT NULL REFERENCES extracted_entities(entity_id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL CHECK (relation_type IN (
        'TREATS', 'HAS_DOSAGE', 'HAS_FREQUENCY', 'HAS_ROUTE', 'HAS_VALUE', 'HAS_UNIT',
        'REVEALS', 'CAUSES', 'LOCATED_IN',
        'HAS_LAB_VALUE', 'HAS_LAB_UNIT', 'LOCATED_IN_ANATOMY', 'TEMPORALLY_ANCHORED_TO', 'TREATS_CONDITION'
    )),
    derivation_type TEXT NOT NULL CHECK (derivation_type IN ('EXTRACTED', 'INFERRED')),
    evidence_text TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    source_stage TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL,
    uncertainty REAL NOT NULL,
    review_status TEXT NOT NULL CHECK (review_status IN ('AI_PREDICTED', 'HUMAN_CORRECTED', 'APPROVED_GROUND_TRUTH', 'REJECTED'))
);

CREATE TABLE IF NOT EXISTS human_reviews (
    review_id TEXT PRIMARY KEY,
    target_type TEXT NOT NULL CHECK (target_type IN ('ENTITY', 'OCR_BLOCK', 'RELATION')),
    target_id TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    reviewer_user_id TEXT NOT NULL REFERENCES users(user_id),
    previous_value_json TEXT NOT NULL,
    corrected_value_json TEXT NOT NULL,
    review_decision TEXT NOT NULL CHECK (review_decision IN ('APPROVE_AS_GROUND_TRUTH', 'CORRECT_AND_APPROVE', 'FLAG_AMBIGUOUS', 'REJECT')),
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    approach_category TEXT NOT NULL CHECK (approach_category IN (
        'DOCUMENT_OCR', 'MEDICAL_NLP', 'MEDICAL_CV', 'MULTIMODAL', 'LONGITUDINAL_KG', 'RETRIEVAL'
    )),
    track TEXT NOT NULL CHECK (track IN ('PRIMARY_RANDOM_INIT', 'REFERENCE_COMPARISON')),
    architecture_name TEXT NOT NULL,
    parameter_count INTEGER NOT NULL,
    init_method TEXT NOT NULL,
    init_seed INTEGER NOT NULL,
    initial_weights_sha256 TEXT NOT NULL,
    trained_weights_sha256 TEXT NOT NULL,
    pretrained_weights_used INTEGER NOT NULL CHECK (pretrained_weights_used IN (0, 1)),
    checkpoint_path TEXT NOT NULL,
    lifecycle_status TEXT NOT NULL CHECK (lifecycle_status IN (
        'EXPERIMENTAL', 'RESEARCH_ONLY', 'INTERNALLY_APPROVED', 'EXTERNALLY_EVALUATED', 'SUSPENDED', 'RETIRED', 'CLINICALLY_VALIDATED'
    )),
    intended_use TEXT NOT NULL,
    prohibited_use TEXT NOT NULL,
    limitations TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    approach_category TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    hardware_snapshot_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    calibration_json TEXT NOT NULL,
    fairness_json TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED')),
    runtime_seconds REAL NOT NULL,
    peak_rss_mb REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS longitudinal_events (
    event_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES subjects(subject_id),
    encounter_id TEXT REFERENCES encounters(encounter_id),
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    entity_id TEXT REFERENCES extracted_entities(entity_id) ON DELETE SET NULL,
    event_category TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    assertion_status TEXT NOT NULL,
    temporal_status TEXT NOT NULL,
    event_date_iso TEXT NOT NULL,
    derivation_type TEXT NOT NULL CHECK (derivation_type IN ('EXTRACTED', 'INFERRED')),
    confidence REAL NOT NULL,
    uncertainty REAL NOT NULL,
    contradiction_flag INTEGER NOT NULL DEFAULT 0,
    contradiction_note TEXT
);

CREATE TABLE IF NOT EXISTS federated_rounds (
    round_id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(experiment_id),
    round_number INTEGER NOT NULL,
    aggregation_algorithm TEXT NOT NULL CHECK (aggregation_algorithm IN ('FEDAVG', 'FEDPROX')),
    participating_institutions_json TEXT NOT NULL,
    dp_epsilon_spent REAL NOT NULL DEFAULT 0.0,
    global_weights_sha256 TEXT NOT NULL,
    metrics_by_site_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    audit_id TEXT PRIMARY KEY,
    sequence_num INTEGER NOT NULL UNIQUE,
    actor_user_id TEXT NOT NULL,
    institution_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    metadata_safe_json TEXT NOT NULL,
    prev_hash_sha256 TEXT NOT NULL,
    entry_hash_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    institution_id TEXT NOT NULL,
    actor_user_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    original_filename TEXT,
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED')),
    progress_pct INTEGER NOT NULL DEFAULT 0,
    current_stage TEXT NOT NULL DEFAULT 'QUEUED',
    result_json TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_institution ON documents(institution_id);
CREATE INDEX IF NOT EXISTS idx_documents_subject ON documents(subject_id);
CREATE INDEX IF NOT EXISTS idx_entities_document ON extracted_entities(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON extracted_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_events_subject_date ON longitudinal_events(subject_id, event_date_iso);
CREATE INDEX IF NOT EXISTS idx_audit_seq ON audit_logs(sequence_num);
CREATE INDEX IF NOT EXISTS idx_jobs_institution ON jobs(institution_id, created_at);
