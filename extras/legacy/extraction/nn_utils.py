"""Pure-NumPy neural primitives (AdamW, batch padding, log-softmax, gradient check)."""

from __future__ import annotations

from typing import Callable

import numpy as np


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    max_val = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_val
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    max_val = np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(logits - max_val)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def pad_sequence_batch(
    token_ids_batch: list[list[int]],
    tag_ids_batch: list[list[int]] | None = None,
    pad_token_id: int = 0,
    pad_tag_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Pad variable-length sequences into (token_ids [B, T], mask [B, T], tag_ids [B, T])."""
    batch_size = len(token_ids_batch)
    max_len = max((len(seq) for seq in token_ids_batch), default=1)
    token_arr = np.full((batch_size, max_len), pad_token_id, dtype=np.int64)
    mask_arr = np.zeros((batch_size, max_len), dtype=np.float64)
    tag_arr = np.full((batch_size, max_len), pad_tag_id, dtype=np.int64) if tag_ids_batch is not None else None

    for i, seq in enumerate(token_ids_batch):
        seq_len = len(seq)
        if seq_len == 0:
            continue
        token_arr[i, :seq_len] = seq
        mask_arr[i, :seq_len] = 1.0
        if tag_arr is not None and tag_ids_batch is not None:
            tag_arr[i, :seq_len] = tag_ids_batch[i]

    return token_arr, mask_arr, tag_arr


def masked_cross_entropy_and_grad(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mean masked token cross-entropy loss and dLoss/dLogits ([B, T, C])."""
    probs = softmax(logits, axis=-1)
    b_idx, t_idx = np.nonzero(mask > 0.0)
    count = max(1, len(b_idx))
    chosen_probs = np.clip(probs[b_idx, t_idx, targets[b_idx, t_idx]], 1e-12, 1.0)
    loss = float(-np.sum(np.log(chosen_probs)) / count)

    grad = probs.copy()
    grad[b_idx, t_idx, targets[b_idx, t_idx]] -= 1.0
    grad *= mask[:, :, None] / float(count)
    return loss, grad


class AdamWOptimizer:
    """In-place AdamW optimizer for NumPy parameter dictionaries."""

    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}
        self.v: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], max_norm: float = 5.0) -> None:
        self.step_count += 1
        total_sq = sum(float(np.sum(g * g)) for g in grads.values())
        norm = float(np.sqrt(total_sq))
        scale = min(1.0, max_norm / (norm + 1e-12)) if norm > max_norm else 1.0

        bc1 = 1.0 - self.beta1 ** self.step_count
        bc2 = 1.0 - self.beta2 ** self.step_count

        for name, param in params.items():
            if name not in grads:
                continue
            grad = grads[name] * scale
            if self.weight_decay > 0.0 and param.ndim >= 2:
                param -= self.lr * self.weight_decay * param
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[name] / bc1
            v_hat = self.v[name] / bc2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def numerical_gradient_check(
    params: dict[str, np.ndarray],
    loss_and_grads_fn: Callable[[], tuple[float, dict[str, np.ndarray]]],
    eps: float = 1e-5,
    max_checks_per_param: int = 4,
    seed: int = 7,
) -> dict[str, float]:
    """Compare analytical gradients against central finite differences; returns max relative error per param."""
    rng = np.random.default_rng(seed)
    _, analytical = loss_and_grads_fn()
    rel_errors: dict[str, float] = {}

    for name, arr in params.items():
        if name not in analytical:
            continue
        flat = arr.reshape(-1)
        flat_grad = analytical[name].reshape(-1)
        if flat.size == 0:
            continue
        sample_count = min(max_checks_per_param, flat.size)
        indices = rng.choice(flat.size, size=sample_count, replace=False)
        max_rel = 0.0
        for idx in indices:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            loss_plus, _ = loss_and_grads_fn()
            flat[idx] = orig - eps
            loss_minus, _ = loss_and_grads_fn()
            flat[idx] = orig
            num_g = (loss_plus - loss_minus) / (2.0 * eps)
            ana_g = float(flat_grad[idx])
            if abs(num_g - ana_g) < 1e-8:
                rel = 0.0
            else:
                denom = max(1e-7, abs(num_g) + abs(ana_g))
                rel = abs(num_g - ana_g) / denom
            if rel > max_rel:
                max_rel = rel
        rel_errors[name] = max_rel

    return rel_errors
