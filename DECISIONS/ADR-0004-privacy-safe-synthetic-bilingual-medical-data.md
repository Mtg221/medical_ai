# ADR-0004: Privacy-Safe Synthetic Bilingual (English/French) Development Data Policy

- **Status:** Accepted
- **Date:** 2026-11-12
- **Specification References:** Sections 2, 5, 15, 17, 25, 46 (Phase 0 & Phase 1)

## Context

Section 2 Rule 9 and Section 15 strictly forbid using unauthorized patient data or exposing Protected Health Information (PHI) in development, logs, test fixtures, or public artifacts. At the same time, Phase 1 requires realistic end-to-end testing of native PDF/DOCX/TXT parsing, scanned image OCR, bilingual (English and French) medical entity extraction, negation/uncertainty/temporality detection, terminology normalization, and leak-free patient/document-family dataset splitting.

## Decision

1. **100% Synthetic Bilingual Clinical Corpus (`SYNTH-MED-EN-FR-v1`):** All development, testing, and Phase 1 baseline experiments use deterministically generated synthetic clinical documents in English, French, and mixed EN/FR across multiple simulated patients and encounters.
2. **Explicit Governance Metadata:** Even synthetic datasets must be registered in the Dataset Registry with explicit `source_type = 'SYNTHETIC'`, license (`CC0-1.0-SYNTHETIC-RESEARCH-ONLY`), de-identification status, SHA-256 manifest checksum, and patient/document-family group identifiers to enforce zero patient leakage between train, validation, and locked test splits.
3. **Zero-PHI Logging Guardrails:** Audit logs and application telemetry pass through an automated sanitizer (`medresearch_platform.security.audit`) that rejects or redacts raw text, patient identifiers, tokens, and secrets.

## Consequences

- **Positive:** Zero privacy risk during software development and baseline validation; reproducible ground-truth spans for OCR (character coordinates) and NLP (entities, assertions, temporality, relations).
- **Negative / Limitations:** Performance metrics measured on synthetic documents reflect software and algorithmic verification (`Experimentally validated` on synthetic benchmark), not real-world clinical generalization (`Clinically validated` remains strictly unsupported without IRB-approved clinical studies).
