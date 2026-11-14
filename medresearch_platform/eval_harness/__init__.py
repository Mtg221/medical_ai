"""Phase 2 Evaluation Harness: metrics, significance tests, resource profiler, error taxonomy, and runner."""

from medresearch_platform.eval_harness.error_taxonomy import ErrorExample, classify_corpus_errors
from medresearch_platform.eval_harness.metrics import (
    DocumentPrediction,
    Evaluator,
    PredictedEntity,
    PredictedRelation,
    PredictedTimex,
    bootstrap_f1_ci,
    paired_significance_test,
)
from medresearch_platform.eval_harness.profiler import ResourceMetrics, ResourceProfiler

__all__ = [
    "DocumentPrediction",
    "ErrorExample",
    "Evaluator",
    "PredictedEntity",
    "PredictedRelation",
    "PredictedTimex",
    "ResourceMetrics",
    "ResourceProfiler",
    "bootstrap_f1_ci",
    "classify_corpus_errors",
    "paired_significance_test",
]
