"""Deterministic random weight initialization and cryptographic provenance verification (Section 7)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Literal
import numpy as np

InitMethod = Literal["he_normal", "xavier_uniform", "normal_small", "zeros"]


@dataclass(frozen=True)
class RandomInitManifest:
    architecture_name: str
    architecture_version: str
    init_method: InitMethod
    seed: int
    parameter_count: int
    tensor_shapes: dict[str, list[int]]
    initial_weights_sha256: str
    pretrained_weights_used: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def initialize_tensor(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    method: InitMethod = "he_normal",
) -> np.ndarray:
    """Initialize a float32 parameter tensor strictly from the provided random number generator."""
    if method == "zeros":
        return np.zeros(shape, dtype=np.float32)

    fan_in = int(np.prod(shape[:-1])) if len(shape) >= 2 else shape[0]
    fan_out = shape[-1] if len(shape) >= 2 else shape[0]

    if method == "he_normal":
        std = float(np.sqrt(2.0 / max(1, fan_in)))
        return rng.normal(loc=0.0, scale=std, size=shape).astype(np.float32)
    if method == "xavier_uniform":
        limit = float(np.sqrt(6.0 / max(1, fan_in + fan_out)))
        return rng.uniform(low=-limit, high=limit, size=shape).astype(np.float32)
    if method == "normal_small":
        return rng.normal(loc=0.0, scale=0.02, size=shape).astype(np.float32)

    raise ValueError(f"Unsupported initialization method: {method}")


def compute_weights_sha256(parameters: dict[str, np.ndarray]) -> str:
    """Compute a deterministic SHA-256 digest over ordered parameter names, shapes, dtypes, and bytes."""
    hasher = hashlib.sha256()
    for name in sorted(parameters.keys()):
        arr = np.ascontiguousarray(parameters[name], dtype=np.float32)
        header = f"{name}:{list(arr.shape)}:{arr.dtype.str}|".encode("utf-8")
        hasher.update(header)
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def count_parameters(parameters: dict[str, np.ndarray]) -> int:
    return int(sum(int(arr.size) for arr in parameters.values()))


def build_random_init_manifest(
    *,
    architecture_name: str,
    architecture_version: str,
    init_method: InitMethod,
    seed: int,
    parameters: dict[str, np.ndarray],
) -> RandomInitManifest:
    return RandomInitManifest(
        architecture_name=architecture_name,
        architecture_version=architecture_version,
        init_method=init_method,
        seed=seed,
        parameter_count=count_parameters(parameters),
        tensor_shapes={k: list(v.shape) for k, v in sorted(parameters.items())},
        initial_weights_sha256=compute_weights_sha256(parameters),
        pretrained_weights_used=False,
    )


def verify_random_init_provenance(
    manifest: RandomInitManifest,
    weight_builder_fn: Any,
) -> bool:
    """Re-run random weight initialization with the recorded seed and verify exact SHA-256 match."""
    if manifest.pretrained_weights_used:
        return False
    regenerated_params = weight_builder_fn(manifest.seed, manifest.init_method)
    regenerated_sha = compute_weights_sha256(regenerated_params)
    return (
        regenerated_sha == manifest.initial_weights_sha256
        and count_parameters(regenerated_params) == manifest.parameter_count
    )
