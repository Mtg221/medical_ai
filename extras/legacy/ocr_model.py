"""Approach 1: Custom Random-Initialized 2D Conv + MLP OCR Neural Network and CER/WER Evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from medresearch_platform.ai.random_init import (
    InitMethod,
    RandomInitManifest,
    build_random_init_manifest,
    compute_weights_sha256,
    initialize_tensor,
)

OCR_CHARSET: str = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:/-%()"
CHAR_TO_IDX: dict[str, int] = {ch: idx for idx, ch in enumerate(OCR_CHARSET)}
IDX_TO_CHAR: dict[int, str] = {idx: ch for idx, ch in enumerate(OCR_CHARSET)}


@dataclass(frozen=True)
class OCRBlockPrediction:
    block_index: int
    recognized_text: str
    ground_truth_text: str | None
    bbox: dict[str, int]
    confidence: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    char_confidences: list[float]


def render_glyph_patch(char: str, size: int = 16, noise_std: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Render a single normalized 16x16 grayscale character glyph patch in [0, 1]."""
    clean_char = char.upper() if char.upper() in CHAR_TO_IDX else " "
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((3, 2), clean_char, fill=255, font=font)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if noise_std > 0.0 and rng is not None:
        arr = np.clip(arr + rng.normal(0.0, noise_std, size=arr.shape).astype(np.float32), 0.0, 1.0)
    return arr


def render_scanned_document_image(
    lines: list[str],
    *,
    char_size: int = 16,
    noise_std: float = 0.02,
    seed: int = 42,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Render multi-line text into a synthetic scanned grayscale image with ground-truth line & character bounding boxes."""
    rng = np.random.default_rng(seed)
    max_cols = max((len(line) for line in lines), default=1)
    width = max(320, max_cols * char_size + 32)
    height = max(120, len(lines) * (char_size + 8) + 32)
    canvas = np.zeros((height, width), dtype=np.float32)
    line_metadata: list[dict[str, Any]] = []

    for row_idx, raw_line in enumerate(lines):
        norm_line = "".join(ch.upper() if ch.upper() in CHAR_TO_IDX else " " for ch in raw_line)
        y0 = 16 + row_idx * (char_size + 8)
        x0 = 16
        char_boxes: list[dict[str, int]] = []
        for col_idx, ch in enumerate(norm_line):
            cx0 = x0 + col_idx * char_size
            patch = render_glyph_patch(ch, size=char_size, noise_std=noise_std, rng=rng)
            canvas[y0 : y0 + char_size, cx0 : cx0 + char_size] = patch
            char_boxes.append({"x": cx0, "y": y0, "width": char_size, "height": char_size})

        line_metadata.append({
            "block_index": row_idx,
            "ground_truth_text": norm_line,
            "bbox": {
                "x": x0,
                "y": y0,
                "width": len(norm_line) * char_size,
                "height": char_size,
            },
            "char_boxes": char_boxes,
        })

    pil_img = Image.fromarray((np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    return pil_img, line_metadata


def levenshtein_distance(seq_a: list[Any] | str, seq_b: list[Any] | str) -> int:
    """Compute exact dynamic-programming Levenshtein edit distance."""
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def compute_cer_and_wer(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute Character Error Rate (CER) and Word Error Rate (WER) across document blocks."""
    total_char_edits = 0
    total_chars = 0
    total_word_edits = 0
    total_words = 0

    for pred, ref in zip(predictions, references):
        p_norm = pred.strip().upper()
        r_norm = ref.strip().upper()
        total_char_edits += levenshtein_distance(p_norm, r_norm)
        total_chars += max(1, len(r_norm))

        p_words = p_norm.split()
        r_words = r_norm.split()
        total_word_edits += levenshtein_distance(p_words, r_words)
        total_words += max(1, len(r_words))

    cer = float(total_char_edits / max(1, total_chars))
    wer = float(total_word_edits / max(1, total_words))
    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "char_accuracy": round(max(0.0, 1.0 - cer), 4),
        "word_accuracy": round(max(0.0, 1.0 - wer), 4),
    }


def _build_ocr_weights(seed: int, init_method: InitMethod = "he_normal") -> dict[str, np.ndarray]:
    """Construct randomly initialized parameters for CustomOCRConvNet-v1.
    Architecture:
      - Patch projection + spatial 4x4 patch-conv filters (16, 16) -> 8 channels on 4x4 grid (128 features)
      - Global + local flattened features (256 + 128 = 384 features) -> Linear(384, 96) + ReLU -> Linear(96, num_classes)
    """
    rng = np.random.default_rng(seed)
    num_classes = len(OCR_CHARSET)
    params = {
        "conv_patch_w": initialize_tensor(rng, (16, 8), init_method),
        "conv_patch_b": initialize_tensor(rng, (8,), "zeros"),
        "fc1_w": initialize_tensor(rng, (256 + 128, 96), init_method),
        "fc1_b": initialize_tensor(rng, (96,), "zeros"),
        "fc2_w": initialize_tensor(rng, (96, num_classes), init_method),
        "fc2_b": initialize_tensor(rng, (num_classes,), "zeros"),
    }
    return params


class CustomOCRConvNet:
    """Custom randomly initialized 2D Patch-Conv + MLP Character Recognition Network (Zero Pretrained Weights)."""

    architecture_name = "CustomOCRConvNet"
    architecture_version = "1.0.0"

    def __init__(self, seed: int = 20261112, init_method: InitMethod = "he_normal") -> None:
        self.seed = seed
        self.init_method: InitMethod = init_method
        self.params = _build_ocr_weights(seed, init_method)
        self.init_manifest: RandomInitManifest = build_random_init_manifest(
            architecture_name=self.architecture_name,
            architecture_version=self.architecture_version,
            init_method=init_method,
            seed=seed,
            parameters=self.params,
        )

    @staticmethod
    def _extract_patch_blocks(x_batch: np.ndarray) -> np.ndarray:
        """Reshape (N, 16, 16) into (N, 16_spatial_locations, 16_pixels_per_4x4_patch)."""
        n = x_batch.shape[0]
        reshaped = x_batch.reshape(n, 4, 4, 4, 4).transpose(0, 1, 3, 2, 4).reshape(n, 16, 16)
        return reshaped

    def forward(self, x_batch: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Forward pass returning probabilities (N, C) and intermediate activations for backprop."""
        n = x_batch.shape[0]
        flat_raw = x_batch.reshape(n, 256)
        patch_blocks = self._extract_patch_blocks(x_batch)  # (N, 16, 16)
        conv_pre = np.matmul(patch_blocks, self.params["conv_patch_w"]) + self.params["conv_patch_b"]  # (N, 16, 8)
        conv_act = np.maximum(0.0, conv_pre)
        conv_flat = conv_act.reshape(n, 128)

        concat_feat = np.concatenate([flat_raw, conv_flat], axis=1)  # (N, 384)
        h1_pre = np.matmul(concat_feat, self.params["fc1_w"]) + self.params["fc1_b"]  # (N, 96)
        h1_act = np.maximum(0.0, h1_pre)
        logits = np.matmul(h1_act, self.params["fc2_w"]) + self.params["fc2_b"]  # (N, C)

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        cache = {
            "x_batch": x_batch,
            "patch_blocks": patch_blocks,
            "conv_pre": conv_pre,
            "concat_feat": concat_feat,
            "h1_pre": h1_pre,
            "h1_act": h1_act,
            "logits": logits,
            "probs": probs,
        }
        return probs, cache

    def train_on_synthetic_glyphs(
        self,
        *,
        epochs: int = 140,
        learning_rate: float = 0.012,
        weight_decay: float = 1e-4,
        noise_std: float = 0.02,
    ) -> dict[str, Any]:
        """Train from random initialization on synthetic 16x16 character glyphs using analytical backprop + AdamW."""
        rng = np.random.default_rng(self.seed + 101)

        # Build training set: 4 realizations per character class
        samples_per_class = 4
        x_list: list[np.ndarray] = []
        y_list: list[int] = []
        for idx, ch in enumerate(OCR_CHARSET):
            x_list.append(render_glyph_patch(ch, size=16, noise_std=0.0, rng=rng))
            y_list.append(idx)
            for _ in range(samples_per_class - 1):
                x_list.append(render_glyph_patch(ch, size=16, noise_std=noise_std, rng=rng))
                y_list.append(idx)

        x_train = np.stack(x_list, axis=0).astype(np.float32)
        y_train = np.array(y_list, dtype=np.int64)
        n = x_train.shape[0]

        m_state = {k: np.zeros_like(v) for k, v in self.params.items()}
        v_state = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        loss_history: list[float] = []
        for step in range(1, epochs + 1):
            probs, cache = self.forward(x_train)
            correct_logprobs = -np.log(np.clip(probs[np.arange(n), y_train], 1e-8, 1.0))
            loss = float(np.mean(correct_logprobs))
            loss_history.append(round(loss, 5))

            # Analytical gradients of cross-entropy wrt logits
            dlogits = probs.copy()
            dlogits[np.arange(n), y_train] -= 1.0
            dlogits /= float(n)

            dfc2_w = np.matmul(cache["h1_act"].T, dlogits)
            dfc2_b = np.sum(dlogits, axis=0)

            dh1_act = np.matmul(dlogits, self.params["fc2_w"].T)
            dh1_pre = dh1_act * (cache["h1_pre"] > 0.0).astype(np.float32)

            dfc1_w = np.matmul(cache["concat_feat"].T, dh1_pre)
            dfc1_b = np.sum(dh1_pre, axis=0)

            dconcat = np.matmul(dh1_pre, self.params["fc1_w"].T)
            dconv_flat = dconcat[:, 256:]  # (N, 128)
            dconv_act = dconv_flat.reshape(n, 16, 8)
            dconv_pre = dconv_act * (cache["conv_pre"] > 0.0).astype(np.float32)

            dconv_w = np.einsum("nsi,nsj->ij", cache["patch_blocks"], dconv_pre)
            dconv_b = np.sum(dconv_pre, axis=(0, 1))

            grads = {
                "fc2_w": dfc2_w,
                "fc2_b": dfc2_b,
                "fc1_w": dfc1_w,
                "fc1_b": dfc1_b,
                "conv_patch_w": dconv_w,
                "conv_patch_b": dconv_b,
            }

            for k, grad in grads.items():
                m_state[k] = beta1 * m_state[k] + (1.0 - beta1) * grad
                v_state[k] = beta2 * v_state[k] + (1.0 - beta2) * (grad * grad)
                m_hat = m_state[k] / (1.0 - beta1 ** step)
                v_hat = v_state[k] / (1.0 - beta2 ** step)
                self.params[k] -= learning_rate * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * self.params[k])

        final_probs, _ = self.forward(x_train)
        preds = np.argmax(final_probs, axis=1)
        train_acc = float(np.mean(preds == y_train))
        return {
            "epochs": epochs,
            "initial_loss": loss_history[0] if loss_history else 0.0,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "train_accuracy": round(train_acc, 4),
            "trained_weights_sha256": compute_weights_sha256(self.params),
        }

    def recognize_image_blocks(
        self,
        image: Image.Image,
        line_metadata: list[dict[str, Any]],
    ) -> list[OCRBlockPrediction]:
        """Run custom OCR inference on line blocks and compute confidence + aleatoric/epistemic uncertainty."""
        gray_arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        results: list[OCRBlockPrediction] = []
        max_entropy = float(np.log(len(OCR_CHARSET)))

        for meta in line_metadata:
            char_boxes = meta["char_boxes"]
            if not char_boxes:
                results.append(
                    OCRBlockPrediction(
                        block_index=int(meta["block_index"]),
                        recognized_text="",
                        ground_truth_text=meta.get("ground_truth_text"),
                        bbox=meta["bbox"],
                        confidence=1.0,
                        aleatoric_uncertainty=0.0,
                        epistemic_uncertainty=0.0,
                        char_confidences=[],
                    )
                )
                continue

            patches: list[np.ndarray] = []
            for box in char_boxes:
                x, y, w, h = box["x"], box["y"], box["width"], box["height"]
                crop = gray_arr[y : y + h, x : x + w]
                if crop.shape != (16, 16):
                    pil_crop = Image.fromarray((crop * 255.0).astype(np.uint8)).resize((16, 16))
                    crop = np.asarray(pil_crop, dtype=np.float32) / 255.0
                patches.append(crop)

            batch = np.stack(patches, axis=0)
            probs, _ = self.forward(batch)
            pred_indices = np.argmax(probs, axis=1)
            recognized = "".join(IDX_TO_CHAR[int(i)] for i in pred_indices)

            top1_probs = np.max(probs, axis=1)
            sorted_probs = np.sort(probs, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]

            # Aleatoric uncertainty: normalized Shannon entropy of predictive distribution
            entropies = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1) / max_entropy
            # Epistemic proxy: dispersion in top-2 margin (1 - margin)
            epistemic = 1.0 - margins

            results.append(
                OCRBlockPrediction(
                    block_index=int(meta["block_index"]),
                    recognized_text=recognized,
                    ground_truth_text=meta.get("ground_truth_text"),
                    bbox=meta["bbox"],
                    confidence=round(float(np.mean(top1_probs)), 4),
                    aleatoric_uncertainty=round(float(np.mean(entropies)), 4),
                    epistemic_uncertainty=round(float(np.mean(epistemic)), 4),
                    char_confidences=[round(float(c), 4) for c in top1_probs],
                )
            )

        return results

    def save_checkpoint(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **cast(Any, self.params))
        return compute_weights_sha256(self.params)

    def load_checkpoint(self, path: Path) -> None:
        with np.load(str(path)) as data:
            for key in self.params:
                self.params[key] = data[key].astype(np.float32)
