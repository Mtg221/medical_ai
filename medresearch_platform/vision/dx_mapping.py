"""Shared honest diagnosis label mapping for training, evaluation, and serving (W2).

Records without a specific harmonised diagnosis (e.g. ISIC raw ``dx == 'Benign'/'Malignant'``
which map to ``benign_unspecified`` / ``malignant_unspecified`` or ``other``) are masked
(``MASKED_DX_INDEX = -1``) rather than fabricated as ``melanoma`` / ``nevus``.
"""

from __future__ import annotations

from medresearch_platform.vision.catalog import DermCatalogRecord

CORE_DX_CLASSES: tuple[str, ...] = (
    "melanoma",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "actinic_keratosis",
    "nevus",
    "seborrheic_keratosis",
    "dermatofibroma",
    "vascular_lesion",
)
CORE_DX_TO_IDX: dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CORE_DX_CLASSES)}
MASKED_DX_INDEX: int = -1
MIN_DX_EVAL_SUPPORT: int = 30


def map_record_to_core_dx_index(rec: DermCatalogRecord) -> int:
    """Return index in ``CORE_DX_CLASSES`` if ``rec.diagnosis_harmonised`` is a specific core class, else ``-1`` (masked)."""
    return CORE_DX_TO_IDX.get(rec.diagnosis_harmonised, MASKED_DX_INDEX)


def map_record_to_core_dx_label(rec: DermCatalogRecord) -> str | None:
    """Return canonical core diagnosis string if specifically labelled, else ``None`` (masked)."""
    return rec.diagnosis_harmonised if rec.diagnosis_harmonised in CORE_DX_TO_IDX else None
