"""Approach 6: Seven-Stage Hybrid Multimodal Medical Extraction Pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from PIL import Image

from medresearch_platform.ai.vision_multimodal import CustomMedicalVisionNet
from medresearch_platform.extraction.bilstm_crf import BiLSTMCRFExtractor
from medresearch_platform.extraction.classical_ml import ClassicalMLExtractor
from medresearch_platform.extraction.contract import (
    ExtractedRecord,
    ExtractionResult,
    build_extracted_record,
)
from medresearch_platform.extraction.rule_based import RuleBasedMedicalExtractor


@dataclass
class HybridMultimodalExtractor:
    """
    Approach 6: Multi-stage hybrid pipeline combining rule-based, classical ML,
    neural sequence tagging (BiLSTM-CRF), optional OCR, optional image vision features,
    and structured clinical metadata without fabricating image-text pairs.
    """

    rule_extractor: RuleBasedMedicalExtractor
    classical_extractor: ClassicalMLExtractor | None = None
    neural_extractor: BiLSTMCRFExtractor | None = None
    vision_model: CustomMedicalVisionNet | None = None
    model_name: str = "HybridMultimodalPipeline"
    model_version: str = "1.0.0"

    def extract(
        self,
        *,
        text: str | None = None,
        ocr_text: str | None = None,
        image_rgb: np.ndarray | None = None,
        image_record_id: str | None = None,
        explicit_image_pair_verified: bool = False,
        metadata: dict[str, Any] | None = None,
        source_record_id: str = "record-0",
        document_id: str = "doc-0",
        language: str = "en",
        page_number: int = 1,
    ) -> ExtractionResult:
        # Stage 1: Ingestion & explicit modality availability tracking
        has_text = bool(text and text.strip())
        has_ocr = bool(ocr_text and ocr_text.strip())
        has_image = image_rgb is not None and image_rgb.size > 0
        has_metadata = bool(metadata)

        modality_status = {
            "text": "available" if has_text else "absent",
            "ocr": "available" if has_ocr else "absent",
            "image": "available" if has_image else "absent",
            "metadata": "available" if has_metadata else "absent",
        }
        warnings: list[str] = []

        # Stage 2: Quality validation
        active_text = text if has_text else (ocr_text if has_ocr else "")
        text_source = "native_text" if has_text else ("ocr_text" if has_ocr else "none")
        if has_ocr and not has_text:
            warnings.append("using_ocr_fallback_without_native_text")

        image_summary: dict[str, Any] | None = None
        image_aligned = False
        if has_image:
            if explicit_image_pair_verified and image_record_id == source_record_id:
                image_aligned = True
            else:
                warnings.append("unverified_image_pair_not_linked")
                modality_status["image"] = "unavailable"

        # Stage 3: Independent modality processing
        stage_results: list[tuple[str, float, ExtractionResult]] = []
        if active_text:
            stage_results.append(
                (
                    "rule_based",
                    0.45,
                    self.rule_extractor.extract(
                        active_text,
                        source_record_id=source_record_id,
                        document_id=document_id,
                        language=language,
                        page_number=page_number,
                    ),
                )
            )
            if self.classical_extractor is not None and self.classical_extractor.is_fitted:
                stage_results.append(
                    (
                        "classical_ml",
                        0.25,
                        self.classical_extractor.extract(
                            active_text,
                            source_record_id=source_record_id,
                            document_id=document_id,
                            language=language,
                            page_number=page_number,
                        ),
                    )
                )
            if self.neural_extractor is not None and self.neural_extractor.encoder is not None:
                stage_results.append(
                    (
                        "bilstm_crf",
                        0.30,
                        self.neural_extractor.extract(
                            active_text,
                            source_record_id=source_record_id,
                            document_id=document_id,
                            language=language,
                            page_number=page_number,
                        ),
                    )
                )

        if image_aligned and image_rgb is not None:
            vmodel = self.vision_model or CustomMedicalVisionNet(seed=42)
            v_pred = vmodel.analyze_image(Image.fromarray(image_rgb))
            image_summary = {
                "vision_model": "CustomMedicalVisionNet-v1",
                "predicted_modality": v_pred.predicted_modality,
                "quality_score": v_pred.quality_score,
                "confidence": round(float(v_pred.confidence), 4),
                "pretrained_weights_used": False,
            }

        # Stage 4 & 5: Evidence alignment & multimodal fusion across spans
        span_buckets: dict[tuple[int | None, int | None, str], list[tuple[str, float, ExtractedRecord]]] = {}
        for branch_name, weight, res in stage_results:
            for rec in res.records:
                key = (rec.char_start, rec.char_end, rec.entity_type)
                span_buckets.setdefault(key, []).append((branch_name, weight, rec))

        fused_records: list[ExtractedRecord] = []
        for (_cs, _ce, _etype), votes in sorted(
            span_buckets.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or 0)
        ):
            total_w = sum(w for _, w, _ in votes)
            weighted_conf = sum(w * r.confidence_score for _, w, r in votes) / max(1e-6, total_w)
            # Small agreement bonus capped at 0.99 when multiple independent models concur
            if len(votes) >= 2:
                weighted_conf = min(0.99, weighted_conf + 0.04 * (len(votes) - 1))

            primary = votes[0][2]
            agreeing_branches = [b for b, _, _ in votes]
            # Stage 7: Human validation readiness flagging
            review_status = "flagged" if (len(votes) == 1 and weighted_conf < 0.75) else "unreviewed"

            fused_records.append(
                build_extracted_record(
                    source_record_id=source_record_id,
                    document_id=document_id,
                    original_text=primary.original_text,
                    extracted_value=primary.extracted_value,
                    entity_type=primary.entity_type,
                    normalized_concept=primary.normalized_concept,
                    char_start=primary.char_start,
                    char_end=primary.char_end,
                    language=language,
                    negation_status=primary.negation_status,
                    assertion_status=primary.assertion_status,
                    temporal_information=primary.temporal_information,
                    confidence_score=weighted_conf,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    provenance={
                        "approach": "hybrid_multimodal_7stage",
                        "stages_executed": [
                            "ingestion",
                            "quality_validation",
                            "independent_modalities",
                            "evidence_alignment",
                            "multimodal_fusion",
                            "structured_output",
                            "human_review_ready",
                        ],
                        "text_source": text_source,
                        "agreeing_branches": agreeing_branches,
                        "metadata_present": has_metadata,
                        "metadata_keys": sorted(metadata.keys()) if metadata else [],
                        "image_aligned": image_aligned,
                        "image_summary": image_summary,
                        "pretrained_weights_used": False,
                    },
                    page_number=page_number,
                    relationships=primary.relationships,
                    review_status=review_status,
                )
            )

        if not fused_records and not active_text and not has_metadata and not image_aligned:
            warnings.append("no_usable_modality_available")

        # Stage 6: Structured medical output generation
        return ExtractionResult(
            source_record_id=source_record_id,
            document_id=document_id,
            language=language,
            model_name=self.model_name,
            model_version=self.model_version,
            records=tuple(fused_records),
            modality_status=modality_status,
            warnings=tuple(warnings),
        )
