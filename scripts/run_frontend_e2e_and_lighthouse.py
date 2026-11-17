"""Start live FastAPI backend and run Playwright E2E + Lighthouse accessibility audit."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def _wait_for_server(url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"Server at {url} did not become ready within {timeout_s}s")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env["MEDRESEARCH_SQLITE_PATH"] = str(Path(tmp) / "e2e_app.sqlite3")
        env["MEDRESEARCH_STORAGE_DIR"] = str(Path(tmp) / "storage")
        env["MEDRESEARCH_DEMO_PASSWORD"] = env.get("MEDRESEARCH_DEMO_PASSWORD", "e2e-demo-pw-2026")
        env["PLAYWRIGHT_BASE_URL"] = "http://127.0.0.1:8011"

        default_model_dir = Path("models/hybrid_v2_seed1")
        if not (default_model_dir / "manifest.json").exists() and not env.get("MEDRESEARCH_MODEL_DIR"):
            import yaml

            from medresearch_platform.eval_harness.runner import _load_jsonl
            from medresearch_platform.hybrid.pipeline import HybridPipeline

            cfg = dict(yaml.safe_load(Path("configs/exp/v3_2/hybrid_v2.yaml").read_text(encoding="utf-8")))
            cfg["epochs"] = 3
            cfg["aux_epochs"] = 2
            v31_dir = Path("data/processed/v3_1")
            train_docs = _load_jsonl(v31_dir / "train.jsonl")[:60]
            e2e_model_dir = Path(tmp) / "models" / "hybrid_v2_seed1"
            pipe = HybridPipeline(seed=1, cfg=cfg)
            pipe.fit(train_docs)
            pipe.save(e2e_model_dir, corpus_id="SYNTH-MED-V3.1-e2e-60")
            env["MEDRESEARCH_MODEL_DIR"] = str(e2e_model_dir)

        default_router_dir = Path("models/vision_router_v1")
        default_skin_dir = Path("models/derm_cnn_seed1")
        if (not (default_router_dir / "manifest.json").exists() and not env.get("MEDRESEARCH_VISION_ROUTER_DIR")) or (
            not (default_skin_dir / "manifest.json").exists() and not env.get("MEDRESEARCH_VISION_SKIN_DIR")
        ):
            from medresearch_platform.vision.cnn_multitask import DermResNetMultiTask
            from medresearch_platform.vision.router import (
                ROUTER_CLASSES,
                PictureTriageRouter,
                extract_router_features,
                generate_document_scan_image,
                generate_non_skin_other_image,
                generate_synthetic_skin_modality_image,
            )

            e2e_router_dir = Path(tmp) / "models" / "vision_router_v1"
            router = PictureTriageRouter(seed=42)
            samples = []
            for i in range(6):
                for r_cls in ("skin_clinical", "skin_dermoscopic", "skin_tbp_tile"):
                    p24, d28 = extract_router_features(generate_synthetic_skin_modality_image(r_cls, i, seed=42))
                    samples.append((p24, d28, ROUTER_CLASSES.index(r_cls)))
                p24_d, d28_d = extract_router_features(generate_document_scan_image(i, seed=42))
                samples.append((p24_d, d28_d, ROUTER_CLASSES.index("document_scan")))
                p24_o, d28_o = extract_router_features(generate_non_skin_other_image(i, seed=42))
                samples.append((p24_o, d28_o, ROUTER_CLASSES.index("other")))
            router.fit(samples, epochs=35, lr=0.05)
            router.save(e2e_router_dir)
            env["MEDRESEARCH_VISION_ROUTER_DIR"] = str(e2e_router_dir)

            e2e_skin_dir = Path(tmp) / "models" / "derm_cnn_seed1"
            skin = DermResNetMultiTask(image_size=64, seed=42)
            skin.save(e2e_skin_dir)
            env["MEDRESEARCH_VISION_SKIN_DIR"] = str(e2e_skin_dir)

        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "medresearch_platform.interfaces.api:create_app", "--factory", "--host", "127.0.0.1", "--port", "8011"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server("http://127.0.0.1:8011/api/v1/health")
            subprocess.run(["npm", "--prefix", "frontend", "run", "test:e2e"], env=env, check=True)
            if "--skip-lighthouse" not in sys.argv:
                subprocess.run(["node", "frontend/e2e/run_lighthouse.mjs"], env=env, check=True)
        finally:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
