"""Domain FastAPI routers for the Medical Research Platform API."""

from medresearch_platform.interfaces.routers.common import (
    ImageReviewRequest,
    ReviewSubmissionRequest,
    auth_error_response,
    extract_principal,
)
from medresearch_platform.interfaces.routers.documents import build_documents_router
from medresearch_platform.interfaces.routers.experiments import build_experiments_router
from medresearch_platform.interfaces.routers.reviews import build_reviews_router
from medresearch_platform.interfaces.routers.system import build_system_router
from medresearch_platform.interfaces.routers.vision import build_vision_router

__all__ = [
    "ImageReviewRequest",
    "ReviewSubmissionRequest",
    "auth_error_response",
    "build_documents_router",
    "build_experiments_router",
    "build_reviews_router",
    "build_system_router",
    "build_vision_router",
    "extract_principal",
]
