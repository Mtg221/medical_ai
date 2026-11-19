"""Step 3: Real 5-Class PyTorch Picture Triage Router trained from random init on real images."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from medresearch_platform.vision.quality import assess_image_quality
from medresearch_platform.vision.router import (
    ROUTER_CLASSES,
    ROUTER_LABELS_EN,
    ROUTER_LABELS_FR,
    TriageDecision,
)


def is_torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _apply_phone_photo_augmentations(im: Image.Image, rng: np.random.Generator, *, add_hand: bool) -> Image.Image:
    """Apply realistic smartphone-capture artifacts: warm/dim indoor lighting, table background, perspective tilt, shadow, crumple, and optional hand edge."""
    w, h = im.size
    # Place page onto a warm desk/table/marble canvas (including narrow slip/receipt widths)
    bg_tone = (int(rng.integers(85, 195)), int(rng.integers(70, 175)), int(rng.integers(58, 155)))
    canvas = Image.new("RGB", (w, h), color=bg_tone)
    scale_y = float(rng.uniform(0.74, 0.92))
    scale_x = float(rng.uniform(0.52, 0.88)) if rng.random() < 0.4 else scale_y
    pw, ph = max(64, int(w * scale_x)), max(64, int(h * scale_y))
    page_resized = im.resize((pw, ph), Image.Resampling.BILINEAR)
    ox = (w - pw) // 2 + int(rng.integers(-18, 19))
    oy = (h - ph) // 2 + int(rng.integers(-22, 23))
    canvas.paste(page_resized, (max(0, min(w - pw, ox)), max(0, min(h - ph, oy))))

    # Perspective warp
    dx1, dy1 = float(rng.uniform(-0.06, 0.06)) * w, float(rng.uniform(-0.05, 0.05)) * h
    dx2, dy2 = float(rng.uniform(-0.06, 0.06)) * w, float(rng.uniform(-0.05, 0.05)) * h
    src = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    dst = [(dx1, dy1), (w - dx2, dy1 * 0.5), (w + dx2 * 0.5, h - dy2), (-dx1 * 0.5, h + dy2 * 0.5)]
    # Solve 8-parameter projective transform
    matrix_rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (u, v) in zip(dst, src):
        matrix_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        matrix_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.append(v)
    coeffs = np.linalg.solve(np.asarray(matrix_rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64)).tolist()
    warped = canvas.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR)

    if add_hand:
        d = ImageDraw.Draw(warped)
        skin_rgb = (int(rng.integers(110, 215)), int(rng.integers(75, 165)), int(rng.integers(55, 140)))
        if rng.random() < 0.5:
            d.ellipse([(-int(0.08 * w), int(0.55 * h)), (int(0.18 * w), int(0.88 * h))], fill=skin_rgb)
        else:
            d.ellipse([(int(0.82 * w), int(0.45 * h)), (int(1.08 * w), int(0.82 * h))], fill=skin_rgb)

    arr = np.asarray(warped, dtype=np.float32)
    # Shadow band + crumple fold waves
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shadow_center = float(rng.uniform(0.25, 0.75)) * w + 0.35 * yy
    shadow = 1.0 - float(rng.uniform(0.18, 0.42)) * np.exp(-((xx - shadow_center) ** 2) / (2.0 * (0.24 * w) ** 2))
    crumple = 1.0 + float(rng.uniform(0.02, 0.06)) * np.sin(0.08 * xx + 0.11 * yy + float(rng.uniform(0, 6.28)))
    arr = arr * (shadow * crumple)[:, :, None]

    # Realistic warm indoor incandescent / dim phone camera white-balance shift
    if rng.random() < 0.55:
        wb = np.array(
            [
                float(rng.uniform(0.95, 1.14)),
                float(rng.uniform(0.82, 0.99)),
                float(rng.uniform(0.66, 0.88)),
            ],
            dtype=np.float32,
        )
        exposure = float(rng.uniform(0.62, 0.98))
        arr = arr * wb[None, None, :] * exposure

    noise = rng.normal(0.0, float(rng.uniform(2.0, 7.0)), size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def render_synth_med_document_pages(
    jsonl_path: str | Path,
    out_dir: str | Path,
    *,
    max_docs: int = 120,
    seed: int = 42,
    phone_photo_aug: bool = True,
) -> list[Path]:
    """Render real SYNTH-MED notes as realistic scanned/printed document images (with skew, paper grain, lighting)."""
    p_in = Path(jsonl_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not p_in.exists():
        return []

    rng = np.random.default_rng(seed)
    saved: list[Path] = []
    lines_src = [ln for ln in p_in.read_text(encoding="utf-8").splitlines() if ln.strip()][:max_docs]
    font = ImageFont.load_default()

    for idx, raw in enumerate(lines_src):
        out_path = out / f"synth_doc_{idx:04d}.jpg"
        if out_path.exists() and out_path.stat().st_size > 0:
            saved.append(out_path)
            continue
        rec = json.loads(raw)
        text = str(rec.get("text") or "")
        bg_val = int(rng.integers(212, 250))
        ink_val = int(rng.integers(18, 115)) if (phone_photo_aug and rng.random() < 0.45) else int(rng.integers(18, 42))
        ink_rgb = (ink_val, ink_val + 2, min(255, ink_val + 6))
        im = Image.new("RGB", (480, 640), color=(bg_val, bg_val - int(rng.integers(1, 8)), bg_val - int(rng.integers(3, 14))))
        draw = ImageDraw.Draw(im)
        # Header & rule
        draw.text((28, 22), f"CLINICAL RECORD #{idx+1:04d}  ({rec.get('language', 'en').upper()})", fill=ink_rgb, font=font)
        draw.line([(28, 42), (452, 42)], fill=(min(255, ink_val + 65), min(255, ink_val + 70), min(255, ink_val + 80)), width=1)

        y = 58
        for para in text.splitlines():
            words = para.split()
            line = ""
            for w in words:
                cand = f"{line} {w}".strip()
                if len(cand) > 64:
                    draw.text((28, y), line, fill=ink_rgb, font=font)
                    y += 16
                    line = w
                    if y > 600:
                        break
                else:
                    line = cand
            if line and y <= 600:
                draw.text((28, y), line, fill=ink_rgb, font=font)
                y += 18
            if y > 600:
                break

        # Print-style scan noise or realistic phone-photo capture augmentation
        if phone_photo_aug and (idx % 5 != 0):
            im = _apply_phone_photo_augmentations(im, rng, add_hand=(idx % 3 == 0))
            noisy_im = im.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.35, 1.05))))
            q = int(rng.integers(45, 76))
        else:
            angle = float(rng.uniform(-2.5, 2.5))
            im = im.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(bg_val - 12, bg_val - 14, bg_val - 16))
            arr = np.asarray(im, dtype=np.float32)
            grad_x = np.linspace(float(rng.uniform(-14, 6)), float(rng.uniform(-6, 14)), arr.shape[1], dtype=np.float32)[None, :, None]
            noise = rng.normal(0.0, 5.5, size=arr.shape).astype(np.float32)
            arr = np.clip(arr + grad_x + noise, 0, 255).astype(np.uint8)
            noisy_im = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.7))))
            q = int(rng.integers(75, 92))
        noisy_im.thumbnail((256, 256), Image.Resampling.BILINEAR)
        noisy_im.save(out_path, format="JPEG", quality=q)
        saved.append(out_path)
    return saved


@dataclass
class TorchRouterManifest:
    architecture_name: str
    image_size: int
    parameter_count: int
    seed: int
    temperature: float
    abstain_threshold: float
    initial_weights_sha256: str
    trained_weights_sha256: str
    pretrained_weights_used: bool = False


class PyTorchPictureTriageRouter:
    """Real 5-class PyTorch CNN router (`Conv2d+BatchNorm+ReLU` x4) trained from random init."""

    def __init__(self, *, image_size: int = 96, seed: int = 42, abstain_threshold: float = 0.45) -> None:
        import torch
        import torch.nn as nn

        self.image_size = image_size
        self.seed = seed
        self.abstain_threshold = abstain_threshold
        self.temperature = 1.0
        self.architecture_name = "PyTorchPictureTriageCNN-v1.0.0"

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(128, len(ROUTER_CLASSES)),
        )
        self.parameter_count = sum(p.numel() for p in self.net.parameters())
        self.initial_weights_sha256 = self._weights_sha256()

    def _weights_sha256(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.net.state_dict().keys()):
            t = self.net.state_dict()[k].detach().cpu().numpy().tobytes()
            h.update(k.encode("utf-8"))
            h.update(t)
        return h.hexdigest()

    def _pil_to_tensor(self, im: Image.Image) -> Any:
        import torch

        rgb = im.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def fit(
        self,
        train_items: list[tuple[Path, int]],
        val_items: list[tuple[Path, int]],
        *,
        epochs: int = 12,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> dict[str, Any]:
        import torch
        import torch.nn as nn

        t0 = time.perf_counter()
        rng = random.Random(self.seed)

        def _load_xy(items: list[tuple[Path, int]]) -> tuple[torch.Tensor, torch.Tensor]:
            xs, ys = [], []
            for p, lbl in items:
                try:
                    with Image.open(p) as im:
                        xs.append(self._pil_to_tensor(im))
                        ys.append(lbl)
                except Exception:
                    continue
            return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

        x_tr, y_tr = _load_xy(train_items)
        x_va, y_va = _load_xy(val_items)

        counts = Counter(y_tr.tolist())
        weights = torch.tensor(
            [len(y_tr) / max(1, counts.get(c, 1) * len(ROUTER_CLASSES)) for c in range(len(ROUTER_CLASSES))],
            dtype=torch.float32,
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

        best_val_acc = -1.0
        best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        history: list[dict[str, float]] = []

        for ep in range(1, epochs + 1):
            self.net.train()
            order = list(range(len(x_tr)))
            rng.shuffle(order)
            ep_loss = 0.0
            n_batches = 0
            for i in range(0, len(order), batch_size):
                idx = order[i : i + batch_size]
                xb = x_tr[idx]
                yb = y_tr[idx]
                # Lightweight flip augmentation
                if rng.random() < 0.5:
                    xb = torch.flip(xb, dims=[3])
                opt.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                n_batches += 1
            scheduler.step()

            self.net.eval()
            with torch.no_grad():
                v_logits = self.net(x_va)
                v_acc = float((v_logits.argmax(dim=1) == y_va).float().mean().item())
            history.append({"epoch": float(ep), "train_loss": round(ep_loss / max(1, n_batches), 4), "val_acc": round(v_acc, 4)})
            if v_acc >= best_val_acc:
                best_val_acc = v_acc
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

        self.net.load_state_dict(best_state)
        self.net.eval()

        # Fit scalar temperature on val logits
        with torch.no_grad():
            v_logits = self.net(x_va)
        best_t = 1.0
        best_nll = float("inf")
        ce = nn.CrossEntropyLoss()
        for t in np.linspace(0.5, 2.5, 21):
            nll = float(ce(v_logits / float(t), y_va).item())
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)
        self.temperature = round(best_t, 3)

        return {
            "epochs": epochs,
            "best_val_acc": round(best_val_acc, 4),
            "temperature": self.temperature,
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "history": history,
            "pretrained_weights_used": False,
        }

    def predict_probs(self, image: Image.Image) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            x = self._pil_to_tensor(image).unsqueeze(0)
            logits = self.net(x) / max(0.1, self.temperature)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return np.asarray(probs, dtype=np.float64)

    def triage_image(self, image: Image.Image) -> TriageDecision:
        probs = self.predict_probs(image)
        pred_idx = int(np.argmax(probs))
        pred_cls = ROUTER_CLASSES[pred_idx]
        conf = float(probs[pred_idx])
        quality = assess_image_quality(image, check_skin_area=(pred_cls.startswith("skin_")))

        abstained = conf < self.abstain_threshold
        abstain_en = None
        abstain_fr = None
        if pred_cls == "document_scan":
            action = "ocr_text_pipeline"
        elif pred_cls.startswith("skin_") and not abstained and quality.quality_ok:
            action = "skin_vision_pipeline"
        else:
            action = "reject_unsupported"
            if pred_cls.startswith("skin_") and not quality.quality_ok:
                abstained = True
                abstain_en = "; ".join(quality.reasons_en) or "Image quality insufficient for reliable analysis."
                abstain_fr = "; ".join(quality.reasons_fr) or "Qualité d'image insuffisante pour une analyse fiable."
            elif pred_cls == "other":
                abstained = True
                abstain_en = "Non-skin photograph — no dermatology risk score computed."
                abstain_fr = "Photographie non cutanée — aucun score de risque dermatologique calculé."
            elif abstained:
                abstain_en = f"Uncertain picture type (confidence {conf:.2f} < {self.abstain_threshold:.2f})."
                abstain_fr = f"Type d'image incertain (confiance {conf:.2f} < {self.abstain_threshold:.2f})."

        return TriageDecision(
            predicted_class=pred_cls,
            route_action=action,
            confidence=round(conf, 4),
            probabilities={c: round(float(probs[i]), 4) for i, c in enumerate(ROUTER_CLASSES)},
            label_en=ROUTER_LABELS_EN[pred_cls],
            label_fr=ROUTER_LABELS_FR[pred_cls],
            quality=quality,
            abstained=abstained,
            abstain_reason_en=abstain_en,
            abstain_reason_fr=abstain_fr,
            model_name=self.architecture_name,
            pretrained_weights_used=False,
        )

    def save(self, out_dir: str | Path) -> dict[str, Any]:
        import torch

        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        weights_path = p / "router_torch.pt"
        torch.save(self.net.state_dict(), weights_path)
        trained_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        manifest = {
            "architecture_name": self.architecture_name,
            "backend": "pytorch_cpu",
            "image_size": self.image_size,
            "classes": list(ROUTER_CLASSES),
            "parameter_count": self.parameter_count,
            "seed": self.seed,
            "temperature": self.temperature,
            "abstain_threshold": self.abstain_threshold,
            "initial_weights_sha256": self.initial_weights_sha256,
            "trained_weights_sha256": trained_sha,
            "pretrained_weights_used": False,
            "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (p / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, model_dir: str | Path) -> PyTorchPictureTriageRouter:
        import torch

        p = Path(model_dir)
        manifest_path = p / "manifest.json"
        weights_path = p / "router_torch.pt"
        if not manifest_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"PyTorch router files missing in {p}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = str(manifest.get("trained_weights_sha256") or "")
        actual_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if not expected_sha or expected_sha != actual_sha:
            raise ValueError(f"PyTorch router SHA-256 mismatch in {p}")
        obj = cls(
            image_size=int(manifest.get("image_size", 96)),
            seed=int(manifest.get("seed", 42)),
            abstain_threshold=float(manifest.get("abstain_threshold", 0.45)),
        )
        obj.net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        obj.net.eval()
        obj.temperature = float(manifest.get("temperature", 1.0))
        return obj


def _predict_with_text_safety_guard(router: PyTorchPictureTriageRouter, im: Image.Image) -> tuple[str, str]:
    """Return `(raw_router_class, served_class_after_text_guard)`."""
    from medresearch_platform.vision.quality import detect_dense_text_lines

    dec = router.triage_image(im)
    raw_cls = dec.predicted_class
    served_cls = raw_cls
    if raw_cls in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile"):
        if detect_dense_text_lines(im).get("has_dense_text"):
            served_cls = "document_scan"
    return raw_cls, served_cls


def evaluate_real_router(
    router: PyTorchPictureTriageRouter,
    eval_items: list[tuple[Path, int]],
    *,
    ood_paths: list[Path] | None = None,
    owner_doc_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Evaluate router (+ pre-skin text-presence safety guard) on held-out real images + OOD check."""
    n_cls = len(ROUTER_CLASSES)
    cm = [[0 for _ in range(n_cls)] for _ in range(n_cls)]
    cm_raw = [[0 for _ in range(n_cls)] for _ in range(n_cls)]
    correct = 0
    total = 0

    for path, true_idx in eval_items:
        try:
            with Image.open(path) as im:
                raw_cls, served_cls = _predict_with_text_safety_guard(router, im)
        except Exception:
            continue
        raw_idx = ROUTER_CLASSES.index(raw_cls)
        pred_idx = ROUTER_CLASSES.index(served_cls)
        cm_raw[true_idx][raw_idx] += 1
        cm[true_idx][pred_idx] += 1
        total += 1
        if pred_idx == true_idx:
            correct += 1

    skin_indices = {ROUTER_CLASSES.index(c) for c in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile")}
    doc_idx = ROUTER_CLASSES.index("document_scan")
    other_idx = ROUTER_CLASSES.index("other")

    skin_to_doc = sum(cm[si][doc_idx] for si in skin_indices)
    doc_to_skin = sum(cm[doc_idx][si] for si in skin_indices)
    raw_doc_to_skin = sum(cm_raw[doc_idx][si] for si in skin_indices)
    other_to_skin = sum(cm[other_idx][si] for si in skin_indices)

    ood_summary: dict[str, Any] = {"total": 0, "routed_to_skin": 0, "predictions": {}}
    if ood_paths:
        ood_counts: Counter[str] = Counter()
        for op in ood_paths:
            try:
                with Image.open(op) as im:
                    _, served_cls = _predict_with_text_safety_guard(router, im)
                ood_counts[served_cls] += 1
            except Exception:
                continue
        ood_summary = {
            "total": sum(ood_counts.values()),
            "routed_to_skin": sum(ood_counts[c] for c in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile")),
            "predictions": dict(ood_counts),
        }

    owner_doc_summary: dict[str, Any] = {"total": 0, "correct_document_scan": 0, "routed_to_skin": 0, "raw_router_to_skin": 0}
    if owner_doc_paths:
        o_counts: Counter[str] = Counter()
        o_raw_counts: Counter[str] = Counter()
        for dp in owner_doc_paths:
            try:
                with Image.open(dp) as im:
                    raw_cls, served_cls = _predict_with_text_safety_guard(router, im)
                o_counts[served_cls] += 1
                o_raw_counts[raw_cls] += 1
            except Exception:
                continue
        owner_doc_summary = {
            "total": sum(o_counts.values()),
            "correct_document_scan": o_counts["document_scan"],
            "routed_to_skin": sum(o_counts[c] for c in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile")),
            "raw_router_to_skin": sum(o_raw_counts[c] for c in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile")),
            "predictions": dict(o_counts),
            "raw_predictions": dict(o_raw_counts),
        }

    return {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classes": list(ROUTER_CLASSES),
        "total_eval": total,
        "accuracy": round(correct / max(1, total), 4),
        "confusion_matrix": cm,
        "critical_errors": {
            "skin_to_document_scan": skin_to_doc,
            "document_scan_to_skin": doc_to_skin,
            "raw_router_document_scan_to_skin": raw_doc_to_skin,
            "other_to_skin": other_to_skin,
        },
        "owner_document_photos": owner_doc_summary,
        "ood_check": ood_summary,
    }
