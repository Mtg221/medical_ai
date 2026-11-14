# ADR-0001: Modular Monolith and CPU-First Engineering Architecture

- **Status:** Accepted
- **Date:** 2026-11-12
- **Specification References:** Sections 2, 5, 6, 8, 36, 39

## Context

The Master Specification mandates building a research-grade medical information intelligence and federated scientific research platform supporting English and French documents. Hardware inspection of the target execution environment confirms:
- **CPU:** Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (6 physical cores / 12 logical threads, `x86_64`)
- **Memory:** 32 GB RAM (34,359,738,368 bytes)
- **GPU:** None (CPU-only execution)
- **OS:** macOS Darwin 25.6.0 (`x86_64`)
- **Available Disk:** ~235 GB free on `/System/Volumes/Data`

Additionally, Section 2 Rule 5 requires starting with a simple, maintainable modular monolith with strict domain boundaries, explicit contracts, and minimal dependencies before introducing microservices or distributed infrastructure.

## Decision

1. **Modular Monolith Structure:** We organize the platform inside a single Python package (`medresearch_platform`) divided into strict, acyclic domain layers:
   - `config` & `diagnostics`: Hardware/runtime inspection and environment profiles.
   - `db`: Transactional relational persistence, migrations, and typed repositories.
   - `security`: Authentication (Argon2id + HMAC tokens), RBAC/ABAC authorization, tamper-evident hash-chained audit logs, untrusted-input guards, and bilingual PHI de-identification.
   - `documents`: Content-addressed document ingestion, native PDF/DOCX/TXT/Image parsing, layout/section segmentation, and synthetic bilingual corpus generation.
   - `ai`: Independently versioned, randomly initialized custom models (OCR ConvNet, Medical NLP Sequence/Assertion/Relation Extractor, Bilingual Terminology Mapper, Longitudinal & Knowledge Graph Engine, and Hybrid Retrieval).
   - `workflows`: Dependency-aware document processing pipeline and human annotation/adjudication lifecycle.
   - `interfaces`: Versioned REST API + accessible Web UI (Starlette/ASGI) and CLI.
2. **CPU-First Execution:** All data pipelines and neural training loops are engineered for bounded RAM and rapid CPU execution using vectorized NumPy/SciPy routines, mini-batch streaming, explicit random seed control, and `psutil` memory/latency telemetry.

## Consequences

- **Positive:** Zero container/microservice overhead; sub-second startup; deterministic testing; full end-to-end execution on the 32 GB Intel Mac without GPU dependencies.
- **Negative / Limitations:** Large-scale corpus pretraining or high-concurrency multi-tenant throughput is constrained by single-node CPU compute; heavy workloads must be queued and bounded.
