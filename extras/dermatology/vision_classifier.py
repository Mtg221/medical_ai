"""Pure-NumPy Randomly Initialized Dermatology ConvNet + Metadata Fusion Classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from medresearch_platform.ai.random_init import (
    RandomInitManifest,
    build_random_init_manifest,
    initialize_tensor,
)
from medresearch_platform.vision.catalog import (
    MALIGNANCY_CLASSES,
    MODALITIES,
    SKIN_TONES,
    DermCatalogRecord,
)
from medresearch_platform.vision.vision_classifier import (
    AdamWOptimizer,
    numerical_gradient_check,
    softmax,
)


def encode_derm_metadata(record: DermCatalogRecord | None) -> np.ndarray:
    """Encode optional modality and skin tone into a 11-dim vector (zeros when absent)."""
    vec = np.zeros((len(MODALITIES) + len(SKIN_TONES),), dtype=np.float64)
    if record is None:
        return vec
    if record.modality in MODALITIES:
        vec[MODALITIES.index(record.modality)] = 1.0
    if record.skin_tone in SKIN_TONES:
        vec[len(MODALITIES) + SKIN_TONES.index(record.skin_tone)] = 1.0
    return vec


class RandomInitDermConvNet:
    """
    Pure-NumPy randomly initialized 2D ConvNet + optional metadata fusion head.
    Architecture: Conv2D(3 -> C, 3x3, pad=1) -> tanh -> 2x2 AvgPool -> Global(Avg+Max) ->
    concat(metadata_dim) -> Linear(hidden) -> tanh -> Linear(num_classes).
    """

    architecture_name = "RandomInitDermConvNet"
    architecture_version = "1.0.0"

    def __init__(
        self,
        conv_channels: int = 6,
        hidden_dim: int = 16,
        metadata_dim: int = len(MODALITIES) + len(SKIN_TONES),
        num_classes: int = len(MALIGNANCY_CLASSES),
        seed: int = 42,
    ) -> None:
        self.conv_channels = conv_channels
        self.hidden_dim = hidden_dim
        self.metadata_dim = metadata_dim
        self.num_classes = num_classes
        self.seed = seed

        rng = np.random.default_rng(seed)
        pooled_dim = 2 * conv_channels + metadata_dim
        self.params: dict[str, np.ndarray] = {
            "W_conv": initialize_tensor(rng, (conv_channels, 3, 3, 3), method="xavier_uniform").astype(np.float64),
            "b_conv": np.zeros((conv_channels,), dtype=np.float64),
            "W_fc1": initialize_tensor(rng, (pooled_dim, hidden_dim), method="xavier_uniform").astype(np.float64),
            "b_fc1": np.zeros((hidden_dim,), dtype=np.float64),
            "W_out": initialize_tensor(rng, (hidden_dim, num_classes), method="xavier_uniform").astype(np.float64),
            "b_out": np.zeros((num_classes,), dtype=np.float64),
        }

    @property
    def parameter_count(self) -> int:
        return int(sum(arr.size for arr in self.params.values()))

    def manifest(self) -> RandomInitManifest:
        return build_random_init_manifest(
            architecture_name=self.architecture_name,
            architecture_version=self.architecture_version,
            init_method="xavier_uniform",
            seed=self.seed,
            parameters=self.params,
        )

    def _conv3x3_forward(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # images: [B, H, W, 3]
        bsz, h, w, _ = images.shape
        padded = np.pad(images, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
        out = np.zeros((bsz, h, w, self.conv_channels), dtype=np.float64)
        W = self.params["W_conv"]  # [C_out, 3, 3, 3]
        b = self.params["b_conv"]
        for dy in range(3):
            for dx in range(3):
                patch = padded[:, dy : dy + h, dx : dx + w, :]  # [B, H, W, 3]
                out += np.einsum("bhwi,ci->bhwc", patch, W[:, dy, dx, :])
        out += b[None, None, None, :]
        return out, padded

    def _conv3x3_backward(self, d_conv: np.ndarray, padded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, h, w, _ = d_conv.shape
        dW = np.zeros_like(self.params["W_conv"])
        db = np.sum(d_conv, axis=(0, 1, 2))
        for dy in range(3):
            for dx in range(3):
                patch = padded[:, dy : dy + h, dx : dx + w, :]
                dW[:, dy, dx, :] = np.einsum("bhwc,bhwi->ci", d_conv, patch)
        return dW, db

    def forward(
        self, images: np.ndarray, metadata: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        bsz = images.shape[0]
        if metadata is None:
            metadata = np.zeros((bsz, self.metadata_dim), dtype=np.float64)

        conv_pre, padded = self._conv3x3_forward(images)
        conv_act = np.tanh(conv_pre)

        # 2x2 average pool
        h2 = conv_act.shape[1] // 2
        w2 = conv_act.shape[2] // 2
        cropped = conv_act[:, : h2 * 2, : w2 * 2, :]
        pooled_2x2 = cropped.reshape(bsz, h2, 2, w2, 2, self.conv_channels).mean(axis=(2, 4))

        # Global average + global max pooling
        g_avg = pooled_2x2.mean(axis=(1, 2))  # [B, C]
        flat_spatial = pooled_2x2.reshape(bsz, h2 * w2, self.conv_channels)
        max_idx = np.argmax(flat_spatial, axis=1)  # [B, C]
        g_max = np.take_along_axis(flat_spatial, max_idx[:, None, :], axis=1)[:, 0, :]  # [B, C]

        fused = np.concatenate([g_avg, g_max, metadata], axis=1)
        fc1_pre = fused @ self.params["W_fc1"] + self.params["b_fc1"]
        fc1_act = np.tanh(fc1_pre)
        logits = fc1_act @ self.params["W_out"] + self.params["b_out"]

        ctx = {
            "padded": padded,
            "conv_shape": conv_act.shape,
            "conv_act": conv_act,
            "h2": h2,
            "w2": w2,
            "max_idx": max_idx,
            "fused": fused,
            "fc1_act": fc1_act,
        }
        return logits, ctx

    def backward(self, d_logits: np.ndarray, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
        fc1_act: np.ndarray = ctx["fc1_act"]
        fused: np.ndarray = ctx["fused"]
        conv_act: np.ndarray = ctx["conv_act"]
        bsz, h, w, c = ctx["conv_shape"]
        h2, w2 = ctx["h2"], ctx["w2"]

        dW_out = fc1_act.T @ d_logits
        db_out = np.sum(d_logits, axis=0)
        d_fc1_act = d_logits @ self.params["W_out"].T
        d_fc1_pre = d_fc1_act * (1.0 - fc1_act * fc1_act)

        dW_fc1 = fused.T @ d_fc1_pre
        db_fc1 = np.sum(d_fc1_pre, axis=0)
        d_fused = d_fc1_pre @ self.params["W_fc1"].T

        d_g_avg = d_fused[:, :c]
        d_g_max = d_fused[:, c : 2 * c]

        d_pooled = np.broadcast_to(d_g_avg[:, None, None, :] / float(h2 * w2), (bsz, h2, w2, c)).copy()
        d_flat_max = np.zeros((bsz, h2 * w2, c), dtype=np.float64)
        np.put_along_axis(d_flat_max, ctx["max_idx"][:, None, :], d_g_max[:, None, :], axis=1)
        d_pooled += d_flat_max.reshape(bsz, h2, w2, c)

        # Backprop 2x2 average pool
        d_cropped = np.repeat(np.repeat(d_pooled / 4.0, 2, axis=1), 2, axis=2)
        d_conv_act = np.zeros((bsz, h, w, c), dtype=np.float64)
        d_conv_act[:, : h2 * 2, : w2 * 2, :] = d_cropped
        d_conv_pre = d_conv_act * (1.0 - conv_act * conv_act)

        dW_conv, db_conv = self._conv3x3_backward(d_conv_pre, ctx["padded"])
        return {
            "W_conv": dW_conv,
            "b_conv": db_conv,
            "W_fc1": dW_fc1,
            "b_fc1": db_fc1,
            "W_out": dW_out,
            "b_out": db_out,
        }


def _cross_entropy_and_grad(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    probs = softmax(logits, axis=-1)
    bsz = logits.shape[0]
    chosen = np.clip(probs[np.arange(bsz), labels], 1e-12, 1.0)
    loss = float(-np.mean(np.log(chosen)))
    grad = probs.copy()
    grad[np.arange(bsz), labels] -= 1.0
    grad /= float(max(1, bsz))
    return loss, grad


@dataclass
class DermVisionTrainer:
    """CPU-friendly trainer and subgroup evaluator for RandomInitDermConvNet."""

    conv_channels: int = 6
    hidden_dim: int = 16
    epochs: int = 20
    batch_size: int = 16
    lr: float = 0.02
    seed: int = 42
    include_metadata: bool = True

    def __post_init__(self) -> None:
        self.model = RandomInitDermConvNet(
            conv_channels=self.conv_channels,
            hidden_dim=self.hidden_dim,
            seed=self.seed,
        )
        self.loss_history: list[float] = []

    def fit(self, samples: list[tuple[np.ndarray, int, DermCatalogRecord]]) -> dict[str, Any]:
        images = np.stack([s[0] for s in samples], axis=0)
        labels = np.asarray([s[1] for s in samples], dtype=np.int64)
        metas = (
            np.stack([encode_derm_metadata(s[2]) for s in samples], axis=0)
            if self.include_metadata
            else np.zeros((len(samples), self.model.metadata_dim), dtype=np.float64)
        )

        optimizer = AdamWOptimizer(self.model.params, lr=self.lr)
        rng = np.random.default_rng(self.seed)
        self.loss_history = []

        for _epoch in range(self.epochs):
            order = rng.permutation(len(samples))
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(order), self.batch_size):
                idxs = order[start : start + self.batch_size]
                logits, ctx = self.model.forward(images[idxs], metas[idxs])
                loss, d_logits = _cross_entropy_and_grad(logits, labels[idxs])
                grads = self.model.backward(d_logits, ctx)
                optimizer.step(self.model.params, grads)
                epoch_loss += loss
                batches += 1
            self.loss_history.append(epoch_loss / max(1, batches))

        manifest = self.model.manifest()
        return {
            "initial_loss": round(self.loss_history[0], 4) if self.loss_history else None,
            "final_loss": round(self.loss_history[-1], 4) if self.loss_history else None,
            "parameter_count": self.model.parameter_count,
            "initial_weights_sha256": manifest.initial_weights_sha256,
            "pretrained_weights_used": manifest.pretrained_weights_used,
        }

    def check_gradients(self) -> dict[str, float]:
        rng = np.random.default_rng(self.seed + 5)
        images = rng.uniform(0.1, 0.9, size=(3, 8, 8, 3))
        metas = rng.uniform(0.0, 1.0, size=(3, self.model.metadata_dim))
        labels = np.asarray([0, 1, 0], dtype=np.int64)

        def loss_and_grads() -> tuple[float, dict[str, np.ndarray]]:
            logits, ctx = self.model.forward(images, metas)
            loss, d_logits = _cross_entropy_and_grad(logits, labels)
            grads = self.model.backward(d_logits, ctx)
            return loss, grads

        return numerical_gradient_check(self.model.params, loss_and_grads)

    def evaluate(self, samples: list[tuple[np.ndarray, int, DermCatalogRecord]]) -> dict[str, Any]:
        images = np.stack([s[0] for s in samples], axis=0)
        labels = np.asarray([s[1] for s in samples], dtype=np.int64)
        metas = (
            np.stack([encode_derm_metadata(s[2]) for s in samples], axis=0)
            if self.include_metadata
            else np.zeros((len(samples), self.model.metadata_dim), dtype=np.float64)
        )
        logits, _ = self.model.forward(images, metas)
        probs = softmax(logits, axis=-1)
        preds = np.argmax(probs, axis=-1)
        p_mal = probs[:, 1]

        acc = float(np.mean(preds == labels))
        brier = float(np.mean((p_mal - labels) ** 2))

        by_skin_tone: dict[str, dict[str, Any]] = {}
        for tone in sorted({s[2].skin_tone for s in samples}):
            idxs = [i for i, s in enumerate(samples) if s[2].skin_tone == tone]
            by_skin_tone[tone] = {
                "count": len(idxs),
                "accuracy": round(float(np.mean(preds[idxs] == labels[idxs])), 4),
            }

        by_modality: dict[str, dict[str, Any]] = {}
        for mod in sorted({s[2].modality for s in samples}):
            idxs = [i for i, s in enumerate(samples) if s[2].modality == mod]
            by_modality[mod] = {
                "count": len(idxs),
                "accuracy": round(float(np.mean(preds[idxs] == labels[idxs])), 4),
            }

        return {
            "num_samples": len(samples),
            "accuracy": round(acc, 4),
            "brier_score": round(brier, 4),
            "by_skin_tone": by_skin_tone,
            "by_modality": by_modality,
            "pretrained_weights_used": False,
            "disclaimer": "Research-only feature estimation on synthetic/research splits; not a clinical diagnosis.",
        }
