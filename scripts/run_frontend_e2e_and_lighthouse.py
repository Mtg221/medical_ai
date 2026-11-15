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
            val_docs = _load_jsonl(v31_dir / "val.jsonl")[:20]
            e2e_model_dir = Path(tmp) / "models" / "hybrid_v2_seed1"
            pipe = HybridPipeline(seed=1, cfg=cfg)
            pipe.fit(train_docs, val_docs)
            pipe.save(e2e_model_dir, corpus_id="SYNTH-MED-V3.1-e2e-60", corpus_hash="e2e-ci")
            env["MEDRESEARCH_MODEL_DIR"] = str(e2e_model_dir)

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
