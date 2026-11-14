# Security & Privacy Architecture Review

**Specification References:** Sections 2, 24, 25, 29, 50

---

## 1. Zero-Trust Access Control (RBAC / ABAC)
- **Password Storage:** Argon2id (`argon2-cffi`, `time_cost=2`, `memory_cost=19456`, `parallelism=2`). Plaintext passwords are never persisted or logged.
- **Scoped Access Tokens:** HMAC-SHA256 signed tokens (`medresearch_platform.security.auth`) carrying `user_id`, `institution_id`, `role`, `permissions`, and expiration timestamp (`ttl_minutes=120`).
- **Separation of Duties (Section 35):**
  - `RESEARCHER`: `medical_data:read`, `dataset:read`, `experiment:read`, `terminology:read`, `search:query`
  - `ANNOTATOR` / `ADJUDICATOR`: `medical_data:read`, `annotation:review`, `annotation:adjudicate`
  - `ML_ENGINEER`: `dataset:read`, `training:execute`, `experiment:read`, `model:register` (no direct `medical_data:read` by default)
  - `AUDITOR`: `audit:read`, `dataset:read`, `experiment:read`
  - `INSTITUTION_ADMIN`: Full institutional governance within their own `institution_id` only.
- **Cross-Institution Isolation:** Every medical document, subject, timeline, and search query enforces `principal.institution_id == target_institution_id` (and PostgreSQL Row-Level Security `CREATE POLICY institution_isolation_documents` in `001_initial_schema.postgres.sql`).

## 2. Untrusted Document & Prompt-Injection Defenses
- **Binary Validation (`inspect_untrusted_bytes`):** Enforces `max_upload_bytes` (default 25 MB), non-empty payloads, and strict magic-byte verification for `PDF` (`%PDF-`), `PNG`, `JPEG`, `TIFF`, and `DOCX` (`PK\x03\x04`).
- **Archive-Bomb Protection:** Inspects `DOCX` zip entry uncompressed sizes against compressed payload size (`max_zip_expansion_ratio = 40.0`) before XML parsing.
- **Prompt-Injection & Script Detection (`scan_text_for_untrusted_instructions`):** Flags English/French instruction-override strings and `<script>` tags in `untrusted_flags_json`. All document text is treated strictly as inert data.

## 3. Bilingual Privacy & De-Identification
- **PII Scrubber (`deidentify_clinical_text`):** Detects and redacts English & French patient names, MRN/IPP numbers, French NIR, US SSN, email addresses, and phone numbers.
- **Interval-Preserving Date Shifting:** Shifts ISO dates by a deterministic, subject-specific negative offset (`-15` to `-180` days) derived from HMAC-SHA256 so longitudinal intervals remain intact.
- **Isolated Pseudonym Vault (`IsolatedPseudonymVault`):** Separates identity reversal from research tables and requires `INSTITUTION_ADMIN` role.
- **Zero-PHI Audit Logging:** `sanitize_audit_metadata` automatically redacts forbidden keys (`password`, `secret`, `token`, `raw_text`, `patient_name`, `nir`, `ssn`, etc.) and truncates long strings (>256 chars) before computing the SHA-256 hash chain.

## 4. Tamper-Evident Audit Chain & Residual Risks
- **Hash Chain (`verify_audit_chain`):** Each `audit_logs` entry computes `SHA-256(sequence_num | actor_user_id | institution_id | action_type | resource_type | resource_id | metadata_safe_json | prev_hash_sha256 | created_at)`. Any modification or sequence gap is detected deterministically.
- **Known Residual Risks:**
  - Rule-based PII detection cannot guarantee 100% recall on unstructured free-text nicknames or rare institutional identifiers; human review is required before external sharing.
  - In `development`/`test` profiles without `MEDRESEARCH_TOKEN_SECRET`, ephemeral secrets are generated per process. Production deployments **must** supply explicit environment secrets (`PlatformConfig.from_env` enforces this).
