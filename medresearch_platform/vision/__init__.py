"""Dermatology vision package (`medresearch_platform.vision`)."""

from __future__ import annotations

from medresearch_platform.vision.card import (
    generate_derm_corpus_card,
    inspect_local_thumbnail_cache,
)
from medresearch_platform.vision.catalog import (
    MALIGNANCY_CLASSES,
    MODALITIES,
    SKIN_TONES,
    DermCatalogRecord,
    generate_synthetic_derm_samples,
    iter_derm_catalog,
    summarize_derm_catalog,
)
from medresearch_platform.vision.pipeline import (
    DermFetchReport,
    download_balanced_thumbnails,
    load_cached_derm_samples,
    run_derm_benchmark,
    select_balanced_records,
)
from medresearch_platform.vision.quality import (
    ImageQualityReport,
    QualityThresholds,
    assess_image_quality,
    calibrate_quality_thresholds_on_val,
    compute_laplacian_variance,
    estimate_skin_and_lesion_area_ratio,
)
from medresearch_platform.vision.router import (
    ROUTER_ACTIONS,
    ROUTER_CLASSES,
    PictureTriageRouter,
    TriageDecision,
    build_router_dataset,
    extract_router_features,
    generate_document_scan_image,
    generate_non_skin_other_image,
    generate_synthetic_skin_modality_image,
    train_and_evaluate_router,
)
from medresearch_platform.vision.splits import (
    audit_derm_splits,
    compute_dhash64_from_array,
    compute_dhash64_from_path,
    deduplicate_and_assign_splits,
    hamming_distance64,
    log_locked_test_access,
)
from medresearch_platform.vision.sync import (
    DermMetaSyncReport,
    load_dotenv_credentials,
    sync_derm_metadata,
)
from medresearch_platform.vision.taxonomy import (
    HARMONISED_CLASSES,
    HarmonisedDiagnosisResult,
    LicencePolicyResult,
    evaluate_licence_policy,
    harmonise_diagnosis,
    load_dx_taxonomy,
    summarize_taxonomy_mapping,
)
from medresearch_platform.vision.vision_classifier import (
    DermVisionTrainer,
    RandomInitDermConvNet,
    encode_derm_metadata,
)

__all__ = [
    "DermCatalogRecord",
    "DermFetchReport",
    "DermMetaSyncReport",
    "DermVisionTrainer",
    "HARMONISED_CLASSES",
    "HarmonisedDiagnosisResult",
    "ImageQualityReport",
    "LicencePolicyResult",
    "MALIGNANCY_CLASSES",
    "MODALITIES",
    "PictureTriageRouter",
    "QualityThresholds",
    "ROUTER_ACTIONS",
    "ROUTER_CLASSES",
    "RandomInitDermConvNet",
    "SKIN_TONES",
    "TriageDecision",
    "assess_image_quality",
    "audit_derm_splits",
    "build_router_dataset",
    "calibrate_quality_thresholds_on_val",
    "compute_dhash64_from_array",
    "compute_dhash64_from_path",
    "compute_laplacian_variance",
    "deduplicate_and_assign_splits",
    "download_balanced_thumbnails",
    "encode_derm_metadata",
    "estimate_skin_and_lesion_area_ratio",
    "evaluate_licence_policy",
    "extract_router_features",
    "generate_derm_corpus_card",
    "generate_document_scan_image",
    "generate_non_skin_other_image",
    "generate_synthetic_derm_samples",
    "generate_synthetic_skin_modality_image",
    "hamming_distance64",
    "harmonise_diagnosis",
    "inspect_local_thumbnail_cache",
    "iter_derm_catalog",
    "load_cached_derm_samples",
    "load_dotenv_credentials",
    "load_dx_taxonomy",
    "log_locked_test_access",
    "run_derm_benchmark",
    "select_balanced_records",
    "summarize_derm_catalog",
    "summarize_taxonomy_mapping",
    "sync_derm_metadata",
    "train_and_evaluate_router",
]
