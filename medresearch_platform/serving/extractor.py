"""Production serving service backed by saved evaluated models (Hybrid v2 / Classical CRF)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from medresearch_platform.approaches.classical_ml import ClassicalCRFModel
from medresearch_platform.data.schema import AnnotatedDocument
from medresearch_platform.eval_harness.metrics import DocumentPrediction
from medresearch_platform.extraction.contract import (
    AssertionStatus,
    ExtractedRecord,
    ExtractedRelationship,
    ExtractionResult,
    NegationStatus,
    RegionCoordinates,
    ReviewStatus,
)
from medresearch_platform.hybrid.pipeline import HybridPipeline
from medresearch_platform.hybrid.stage4_temporal import resolve_document_date

_ASSERTION_TO_CONTRACT: dict[str, AssertionStatus] = {
    "present": "present",
    "absent": "absent",
    "possible": "uncertain",
    "hypothetical": "hypothetical",
    "conditional": "uncertain",
    "family": "family_history",
}

_ASSERTION_TO_NEGATION: dict[str, NegationStatus] = {
    "present": "affirmed",
    "absent": "negated",
    "possible": "uncertain",
    "hypothetical": "uncertain",
    "conditional": "uncertain",
    "family": "affirmed",
}


def _resolve_page_and_bbox(
    char_start: int,
    char_end: int,
    page_map: list[dict[str, Any]] | None,
) -> tuple[int, RegionCoordinates | None]:
    if not page_map:
        return 1, None
    for page in page_map:
        p_num = int(page.get("page_number", 1))
        p_start = int(page.get("char_start", 0))
        p_end = int(page.get("char_end", 0))
        if p_start <= char_start < max(p_end, p_start + 1):
            word_boxes = page.get("word_boxes") or []
            overlapping = []
            for wb in word_boxes:
                if isinstance(wb, dict):
                    wb_start = int(wb.get("char_start", -1))
                    wb_end = int(wb.get("char_end", -1))
                    x0 = float(wb["x0"]) if wb.get("x0") is not None else float(wb.get("x") or 0.0)
                    y0 = float(wb["y0"]) if wb.get("y0") is not None else float(wb.get("y") or 0.0)
                    x1 = float(wb["x1"]) if wb.get("x1") is not None else x0 + float(wb.get("width") or 0.0)
                    y1 = float(wb["y1"]) if wb.get("y1") is not None else y0 + float(wb.get("height") or 0.0)
                else:
                    wb_start = int(getattr(wb, "char_start", -1))
                    wb_end = int(getattr(wb, "char_end", -1))
                    x0 = float(getattr(wb, "x", 0.0))
                    y0 = float(getattr(wb, "y", 0.0))
                    x1 = x0 + float(getattr(wb, "width", 0.0))
                    y1 = y0 + float(getattr(wb, "height", 0.0))
                if max(wb_start, char_start) < min(wb_end, char_end):
                    overlapping.append((x0, y0, x1, y1))
            if overlapping:
                return p_num, RegionCoordinates(
                    x0=round(min(b[0] for b in overlapping), 4),
                    y0=round(min(b[1] for b in overlapping), 4),
                    x1=round(max(b[2] for b in overlapping), 4),
                    y1=round(max(b[3] for b in overlapping), 4),
                )
            return p_num, None
    return int(page_map[0].get("page_number", 1)), None


class ExtractionService:
    """Loads saved evaluated models from disk and serves ExtractionResult records."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        classical_crf_dir: Path | str | None = None,
    ) -> None:
        default_dir = Path(os.environ.get("MEDRESEARCH_MODEL_DIR", "models/hybrid_v2_seed1"))
        self.model_dir = Path(model_dir) if model_dir is not None else default_dir
        self.classical_crf_dir = (
            Path(classical_crf_dir)
            if classical_crf_dir is not None
            else Path(os.environ.get("MEDRESEARCH_CLASSICAL_CRF_DIR", "models/classical_crf_seed1"))
        )
        self.torch_ner_dir = Path(
            os.environ.get("MEDRESEARCH_TORCH_NER_DIR", "models/hospital_char_cnn_bilstm_crf_seed1")
        )
        self._hybrid: HybridPipeline | None = None
        self._classical_crf: ClassicalCRFModel | None = None
        self._torch_ner_models: list[Any] | None = None

    def _discover_torch_ner_dirs(self) -> list[Path]:
        raw_env = os.environ.get("MEDRESEARCH_TORCH_NER_DIR", "").strip()
        if "," in raw_env:
            return [Path(p.strip()) for p in raw_env.split(",") if p.strip() and Path(p.strip()).exists()]
        if raw_env:
            p = Path(raw_env)
            return [p] if p.exists() else []
        # Auto-discover multi-seed hospital checkpoints (seed1, seed2, seed3, ...)
        base_parent = self.torch_ner_dir.parent
        discovered = sorted(
            d for d in base_parent.glob("hospital_char_cnn_bilstm_crf_seed*")
            if d.is_dir() and (d / "manifest.json").exists()
        )
        if discovered:
            return discovered
        return [self.torch_ner_dir] if self.torch_ner_dir.exists() else []

    def _get_torch_ner_models(self) -> list[Any]:
        if self._torch_ner_models is None:
            dirs = self._discover_torch_ner_dirs()
            if not dirs:
                self._torch_ner_models = []
            else:
                from medresearch_platform.approaches.torch_ner import (
                    PyTorchCharCNNBiLSTMCRF,
                    is_torch_available,
                )

                if is_torch_available():
                    self._torch_ner_models = [PyTorchCharCNNBiLSTMCRF.load(d, device="cpu") for d in dirs]
                else:
                    self._torch_ner_models = []
        return self._torch_ner_models

    def _get_torch_ner(self) -> Any:
        models = self._get_torch_ner_models()
        return models[0] if models else None

    def _get_hybrid(self) -> HybridPipeline:
        if self._hybrid is None:
            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"Saved model not found at '{self.model_dir}'. "
                    f"Train and save it first with: medextract train --config configs/exp/v3_2/hybrid_v2.yaml "
                    f"--seed 1 --out {self.model_dir}"
                )
            self._hybrid = HybridPipeline.load(self.model_dir)
        return self._hybrid

    def _get_classical_crf(self) -> ClassicalCRFModel:
        if self._classical_crf is None:
            if not self.classical_crf_dir.exists():
                raise FileNotFoundError(
                    f"Saved classical_crf model not found at '{self.classical_crf_dir}'. "
                    f"Train and save it first with: medextract train --config configs/exp/v3_1/classical_crf.yaml "
                    f"--seed 1 --out {self.classical_crf_dir}"
                )
            self._classical_crf = ClassicalCRFModel.load(self.classical_crf_dir)
        return self._classical_crf

    @staticmethod
    def get_review_threshold(cfg: dict[str, Any] | None = None) -> float:
        env_val = os.environ.get("MEDRESEARCH_REVIEW_THRESHOLD")
        if env_val is not None and env_val.strip() != "":
            return float(env_val)
        if cfg and cfg.get("review_threshold") is not None:
            return float(cfg["review_threshold"])
        return 0.60

    def list_loaded_models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.model_dir.exists():
            h = self._get_hybrid()
            cfg = h.manifest.get("cfg", h.cfg)
            out.append({
                "model_id": "hybrid_v2",
                "name": "hybrid_v2",
                "model_type": h.manifest.get("model_type", "hybrid"),
                "seed": h.manifest.get("seed", h.seed),
                "cfg": cfg,
                "review_threshold": self.get_review_threshold(cfg if isinstance(cfg, dict) else None),
                "corpus_id": h.manifest.get("corpus_id", h.corpus_id),
                "corpus_hash": h.manifest.get("corpus_hash", h.corpus_hash),
                "git_commit": h.manifest.get("git_commit", "unknown"),
                "saved_at_utc": h.manifest.get("saved_at_utc", ""),
                "files_sha256": h.manifest.get("files", {}),
                "model_dir": str(self.model_dir),
            })
        if self.classical_crf_dir.exists():
            c = self._get_classical_crf()
            cfg = c.manifest.get("cfg", c.cfg)
            out.append({
                "model_id": "classical_crf",
                "name": "classical_crf",
                "model_type": c.manifest.get("model_type", "classical_crf"),
                "seed": c.manifest.get("seed", c.seed),
                "cfg": cfg,
                "review_threshold": self.get_review_threshold(cfg if isinstance(cfg, dict) else None),
                "corpus_id": c.manifest.get("corpus_id", c.corpus_id),
                "corpus_hash": c.manifest.get("corpus_hash", c.corpus_hash),
                "git_commit": c.manifest.get("git_commit", "unknown"),
                "saved_at_utc": c.manifest.get("saved_at_utc", ""),
                "files_sha256": c.manifest.get("files", {}),
                "model_dir": str(self.classical_crf_dir),
            })
        t_models = self._get_torch_ner_models()
        if t_models:
            t_ner = t_models[0]
            out.append({
                "model_id": "hospital_char_cnn_bilstm_crf",
                "name": "hospital_char_cnn_bilstm_crf",
                "model_type": "pytorch_char_cnn_bilstm_crf",
                "seed": t_ner.seed,
                "num_seeds": len(t_models),
                "cfg": {
                    "pretrained_weights_used": False,
                    "partial_label_marginal_crf": True,
                    "ensemble_seeds": [m.seed for m in t_models],
                },
                "review_threshold": self.get_review_threshold(None),
                "corpus_id": "SYNTH-MED-V3.1+open_clinical",
                "corpus_hash": t_ner.trained_weights_sha256[:16],
                "git_commit": "local",
                "saved_at_utc": "",
                "files_sha256": {"char_cnn_bilstm_crf.pt": t_ner.trained_weights_sha256},
                "model_dir": str(self.torch_ner_dir),
            })
        return out

    def extract(
        self,
        text: str,
        language: str = "en",
        document_date: str | None = None,
        page_map: list[dict[str, Any]] | None = None,
        *,
        document_id: str = "doc_serving",
        source_record_id: str = "rec_serving",
        approach: str = "hybrid_v2",
    ) -> ExtractionResult:
        doc_date, _doc_date_source = resolve_document_date(text, fallback_date=document_date)
        doc = AnnotatedDocument(
            doc_id=document_id,
            patient_id="PAT-SERVING",
            language=language if language in ("en", "fr") else "en",
            doc_type="discharge_summary",
            document_date=doc_date,
            source=document_id,
            text=text,
        )

        source_stage_by_key: dict[tuple[str, int, int], str] = {}
        eff_approach = os.environ.get("MEDRESEARCH_DEFAULT_APPROACH", approach) if approach == "hybrid_v2" else approach
        t_models = self._get_torch_ner_models() if eff_approach == "hospital_ensemble" else []
        if eff_approach == "hospital_ensemble" and t_models:
            from medresearch_platform.approaches.context_negex import apply_context_and_relations
            from medresearch_platform.approaches.torch_ner import ensemble_torch_ner_predictions
            from medresearch_platform.eval_harness.metrics import PredictedEntity

            model = self._get_hybrid()
            pred_h, payload = model.extract_document(doc)
            torch_ents = ensemble_torch_ner_predictions(t_models, doc.text)
            merged: dict[tuple[str, int, int], PredictedEntity] = {}
            for e in torch_ents:
                merged[(e.type, e.start, e.end)] = e
                source_stage_by_key[(e.type, e.start, e.end)] = "torch_char_cnn_bilstm_crf"
            for he in pred_h.entities:
                key = (he.type, he.start, he.end)
                if key in merged:
                    prev = merged[key]
                    merged[key] = PredictedEntity(
                        type=prev.type,
                        start=prev.start,
                        end=prev.end,
                        text=prev.text,
                        assertion=he.assertion,
                        confidence=round(min(0.99, max(prev.confidence, he.confidence) + 0.05), 4),
                    )
                    source_stage_by_key[key] = "hospital_ensemble(torch+hybrid_v2)"
                else:
                    overlaps = [k for k, m in merged.items() if max(he.start, m.start) < min(he.end, m.end)]
                    if not overlaps and he.confidence >= 0.72:
                        merged[key] = he
                        source_stage_by_key[key] = "hybrid_v2"
                    elif overlaps and all(he.confidence > merged[k].confidence + 0.12 for k in overlaps):
                        for k in overlaps:
                            del merged[k]
                        merged[key] = he
                        source_stage_by_key[key] = "hybrid_v2"
            ens_ents = sorted(merged.values(), key=lambda x: x.start)
            enriched_ents, ens_rels, _ = apply_context_and_relations(doc, ens_ents)
            pred = DocumentPrediction(
                doc_id=doc.doc_id,
                language=doc.language,
                doc_type=doc.doc_type,
                predicted_doc_type=pred_h.predicted_doc_type,
                entities=enriched_ents,
                relations=pred_h.relations or ens_rels,
                timex=pred_h.timex,
            )
            manifest = model.manifest
            model_name = "hospital_ensemble"
            model_version = f"seed{model.seed}+torch_partial_crf"
            warnings = tuple(payload.get("validation", {}).get("warnings", []))
        elif eff_approach == "hybrid_v2" or approach == "hybrid_v2":
            model = self._get_hybrid()
            pred, payload = model.extract_document(doc)
            for item in payload.get("records", []):
                prov = item.get("provenance", {})
                key = (str(item.get("entity_type")), int(item.get("char_start", 0)), int(item.get("char_end", 0)))
                source_stage_by_key[key] = str(prov.get("source_stage", "hybrid_v2"))
            manifest = model.manifest
            model_name = "hybrid_v2"
            model_version = f"seed{model.seed}@{manifest.get('git_commit', 'unknown')}"
            warnings = tuple(payload.get("validation", {}).get("warnings", []))
        elif approach == "classical_crf":
            crf_model = self._get_classical_crf()
            pred = crf_model.predict_document(doc)
            manifest = crf_model.manifest
            model_name = "classical_crf"
            model_version = f"seed{crf_model.seed}@{manifest.get('git_commit', 'unknown')}"
            warnings = ()
        else:
            raise ValueError(f"Unsupported serving approach: {approach!r}")

        return self._prediction_to_result(
            pred=pred,
            text=text,
            page_map=page_map,
            document_id=document_id,
            source_record_id=source_record_id,
            model_name=model_name,
            model_version=model_version,
            manifest=manifest,
            source_stage_by_key=source_stage_by_key,
            warnings=warnings,
        )

    def _prediction_to_result(
        self,
        *,
        pred: DocumentPrediction,
        text: str,
        page_map: list[dict[str, Any]] | None,
        document_id: str,
        source_record_id: str,
        model_name: str,
        model_version: str,
        manifest: dict[str, Any],
        source_stage_by_key: dict[tuple[str, int, int], str],
        warnings: tuple[str, ...],
    ) -> ExtractionResult:
        ent_by_span = {(e.start, e.end): e for e in pred.entities}
        rels_by_head: dict[tuple[int, int], list[ExtractedRelationship]] = {}
        for rel in pred.relations:
            tail = ent_by_span.get((rel.tail_start, rel.tail_end))
            tail_text = tail.text if tail is not None else text[rel.tail_start : rel.tail_end]
            tail_type = tail.type if tail is not None else "UNKNOWN"
            rels_by_head.setdefault((rel.head_start, rel.head_end), []).append(
                ExtractedRelationship(
                    relation_type=rel.type,
                    target_value=tail_text,
                    target_entity_type=tail_type,
                    confidence=float(rel.confidence),
                    evidence_text=text[min(rel.head_start, rel.tail_start) : max(rel.head_end, rel.tail_end)][:160],
                )
            )

        cfg_dict = manifest.get("cfg") if isinstance(manifest.get("cfg"), dict) else None
        review_threshold = self.get_review_threshold(cfg_dict)

        records: list[ExtractedRecord] = []
        for idx, ent in enumerate(pred.entities):
            page_num, bbox = _resolve_page_and_bbox(ent.start, ent.end, page_map)
            matched_timex = next(
                (t.normalized for t in pred.timex if max(t.start, ent.start) < min(t.end, ent.end)),
                None,
            )
            stage_name = source_stage_by_key.get((ent.type, ent.start, ent.end), model_name)
            assertion_contract = _ASSERTION_TO_CONTRACT.get(ent.assertion, "present")
            negation_contract = _ASSERTION_TO_NEGATION.get(ent.assertion, "affirmed")
            ent_conf = float(ent.confidence)
            is_low_conf = ent_conf < review_threshold
            record_review_status: ReviewStatus = "needs_review" if is_low_conf else "unreviewed"
            review_reason = "low_confidence" if is_low_conf else ""
            records.append(
                ExtractedRecord(
                    source_record_id=f"{source_record_id}_{idx}",
                    document_id=document_id,
                    original_text=ent.text,
                    extracted_value=ent.text,
                    entity_type=ent.type,
                    normalized_concept=matched_timex,
                    char_start=ent.start,
                    char_end=ent.end,
                    page_number=page_num,
                    region_coordinates=bbox,
                    language=pred.language,
                    negation_status=negation_contract,
                    assertion_status=assertion_contract,
                    uncertainty=round(max(0.0, 1.0 - ent_conf), 4),
                    temporal_information=matched_timex,
                    relationships=tuple(rels_by_head.get((ent.start, ent.end), [])),
                    confidence_score=round(ent_conf, 4),
                    model_name=model_name,
                    model_version=model_version,
                    review_status=record_review_status,
                    provenance={
                        "source_stage": stage_name,
                        "raw_assertion": ent.assertion,
                        "predicted_doc_type": pred.predicted_doc_type,
                        "review_reason": review_reason,
                        "review_threshold": review_threshold,
                        "corpus_id": manifest.get("corpus_id", "unknown"),
                        "corpus_hash": manifest.get("corpus_hash", ""),
                        "git_commit": manifest.get("git_commit", "unknown"),
                        "files_sha256": manifest.get("files", {}),
                        "relations": [
                            {
                                "type": r.type,
                                "head_start": r.head_start,
                                "head_end": r.head_end,
                                "tail_start": r.tail_start,
                                "tail_end": r.tail_end,
                                "confidence": r.confidence,
                            }
                            for r in pred.relations
                            if r.head_start == ent.start and r.head_end == ent.end
                        ],
                        "timex": [
                            {"type": t.type, "start": t.start, "end": t.end, "text": t.text, "normalized": t.normalized}
                            for t in pred.timex
                        ],
                    },
                )
            )

        return ExtractionResult(
            source_record_id=source_record_id,
            document_id=document_id,
            language=pred.language,
            model_name=model_name,
            model_version=model_version,
            records=tuple(records),
            modality_status={"text": "evaluated_model", "ocr": "tesseract_layout"},
            warnings=warnings,
        )
