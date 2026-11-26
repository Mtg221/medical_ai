import type { components, paths } from "./openapi";

export type OpenApiSchemas = components["schemas"];
export type OpenApiPaths = paths;
export type ReviewSubmissionRequest = OpenApiSchemas["ReviewSubmissionRequest"] & {
  corrected_certainty?: "confirmed" | "possible" | "absent" | null;
  corrected_temporality?: "current" | "historical" | null;
  corrected_experiencer?: "patient" | "family" | null;
};

export interface HumanReviewRecord {
  review_id: string;
  target_type: string;
  target_id: string;
  document_id: string;
  reviewer_user_id: string;
  previous_value: Record<string, unknown>;
  corrected_value: Record<string, unknown>;
  review_decision: string;
  rationale: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  institution_id: string;
  role: string;
  permissions: string[];
}

export interface DocumentSummary {
  document_id: string;
  family_id: string;
  subject_id: string;
  institution_id: string;
  dataset_id: string;
  original_filename: string;
  file_format: string;
  content_sha256: string;
  byte_size: number;
  page_count: number;
  detected_language: string;
  document_type: string;
  specialty: string;
  extraction_mode: string;
  untrusted_flags_json: string;
  deidentified_preview: string;
  created_at: string;
}

export interface DocumentPage {
  page_id: string;
  document_id: string;
  page_number: number;
  width_px: number;
  height_px: number;
  raw_text: string;
  extraction_source: string;
  mean_confidence: number;
  mean_uncertainty: number;
}

export interface ExtractedEntity {
  entity_id: string;
  document_id: string;
  page_number: number;
  section_type?: string;
  entity_type: string;
  source_text?: string;
  surface_text?: string;
  normalized_text: string;
  char_start: number;
  char_end: number;
  assertion_status: string;
  certainty?: "confirmed" | "possible" | "absent";
  temporality?: "current" | "historical";
  experiencer?: "patient" | "family";
  temporal_status: string;
  confidence: number;
  uncertainty: number;
  review_status: string;
  review_reason?: string;
  vocabulary?: string;
  terminology_code?: string;
}

export interface ExtractedRelation {
  relation_id: string;
  document_id: string;
  source_entity_id: string;
  target_entity_id: string;
  relation_type: string;
  confidence: number;
}

export interface DocumentDetailResponse {
  document: DocumentSummary;
  pages: DocumentPage[];
  sections: Array<Record<string, unknown>>;
  ocr_blocks: Array<Record<string, unknown>>;
  entities: ExtractedEntity[];
  relations: ExtractedRelation[];
  clinical_disclaimer: string;
}

export interface JobStatus {
  job_id: string;
  institution_id: string;
  actor_user_id: string;
  job_type: string;
  status: "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED";
  progress_pct: number;
  current_stage: string;
  result: Record<string, unknown> | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface TimelineEvent {
  event_id: string;
  subject_id: string;
  event_date_iso: string;
  event_type: string;
  summary_text: string;
  contradiction_flag: number;
  contradiction_note?: string | null;
}

export interface AuditVerifyResult {
  valid: boolean;
  entries_checked: number;
  broken_at_sequence?: number | null;
}

export interface DiagnosisCandidate {
  rank: number;
  diagnosis: string;
  probability: number;
}

export interface ImageQualityReport {
  quality_ok: boolean;
  quality_score: number;
  width: number;
  height: number;
  laplacian_variance: number;
  mean_brightness: number;
  contrast_std: number;
  skin_area_ratio: number;
  reasons_code: string[];
  reasons_en: string[];
  reasons_fr: string[];
}

export interface SameLesionLink {
  link_id: string;
  relation_type: "SAME_LESION";
  patient_id: string | null;
  document_id: string;
  image_id: string;
  text_entity_id: string;
  text_span: string;
  char_start: number;
  char_end: number;
  matched_body_site: string;
  date_delta_days: number | null;
  confidence: number;
  review_status: string;
  rationale_en: string;
  rationale_fr: string;
}

export interface ImageFindingRecord {
  image_id: string;
  document_id: string | null;
  patient_id: string | null;
  original_filename?: string;
  picture_type: string;
  quality_result: ImageQualityReport;
  malignancy_risk_probability: number | null;
  malignancy_risk_band: "low" | "medium" | "high" | "abstained";
  top3_diagnoses: DiagnosisCandidate[];
  corrected_diagnosis?: string | null;
  heatmap_uri: string | null;
  thumbnail_uri?: string | null;
  thumbnail_base64?: string | null;
  abstained: boolean;
  abstain_reason_en: string | null;
  abstain_reason_fr: string | null;
  body_site: string | null;
  captured_date_iso: string | null;
  exif_stripped?: boolean;
  model_name: string;
  model_version: string;
  review_status: string;
  safety_notice_en: string;
  safety_notice_fr: string;
  same_lesion_links?: SameLesionLink[];
  created_at?: string;
}

export interface PatientUnifiedRecord {
  patient_id: string;
  pseudonym_code: string;
  institution_id: string;
  documents: DocumentSummary[];
  text_entities: ExtractedEntity[];
  image_findings: ImageFindingRecord[];
  same_lesion_links: SameLesionLink[];
  timeline_events: Array<{
    event_id: string;
    event_category: string;
    summary_text: string;
    event_date_iso: string;
    review_status?: string;
    image_id?: string;
    document_id?: string;
  }>;
  safety_notice_en: string;
  safety_notice_fr: string;
}
