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
    "LicencePolicyResult",
    "MALIGNANCY_CLASSES",
    "MODALITIES",
    "RandomInitDermConvNet",
    "SKIN_TONES",
    "audit_derm_splits",
    "compute_dhash64_from_array",
    "compute_dhash64_from_path",
    "deduplicate_and_assign_splits",
    "download_balanced_thumbnails",
    "encode_derm_metadata",
    "evaluate_licence_policy",
    "generate_derm_corpus_card",
    "generate_synthetic_derm_samples",
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
]
