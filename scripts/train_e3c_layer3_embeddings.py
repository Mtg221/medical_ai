"""Train from-scratch bilingual EN/FR clinical SVD-PPMI embeddings + clusters on E3C Layer 3."""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

from medresearch_platform.approaches.self_supervised import SelfSupervisedMedicalEmbeddings


def train_from_e3c_zip(
    *,
    zip_path: Path = Path("data/raw/open_clinical/E3C-Corpus-main.zip"),
    out_path: Path = Path("models/e3c_self_supervised_en_fr.npz"),
    max_per_lang: int = 4000,
    max_vocab: int = 16000,
    dim: int = 32,
    permissive_only: bool = True,
    seed: int = 42,
) -> dict[str, object]:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing {zip_path}. Run: python3 scripts/download_open_clinical_corpora.py"
        )
    t0 = time.perf_counter()
    texts: list[str] = []
    lang_counts: Counter[str] = Counter()
    lic_counts: Counter[str] = Counter()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for lang_dir, lang_code in (("English", "en"), ("French", "fr")):
            prefix = f"/data_collection/{lang_dir}/layer3/"
            members = sorted(n for n in zf.namelist() if prefix in n and n.endswith(".json"))
            taken = 0
            for name in members:
                if taken >= max_per_lang:
                    break
                rec = json.loads(zf.read(name).decode("utf-8", errors="replace"))
                lic = str(rec.get("licence", "") or "").strip()
                if permissive_only and ("nc" in lic.lower() or not lic):
                    continue
                txt = str(rec.get("text", "") or "").strip()
                if len(txt) < 80:
                    continue
                texts.append(txt)
                lang_counts[lang_code] += 1
                lic_counts[lic] += 1
                taken += 1

    model = SelfSupervisedMedicalEmbeddings.fit_from_texts(
        texts,
        max_vocab=max_vocab,
        min_count=4,
        window=4,
        dim=dim,
        n_clusters_fine=64,
        n_clusters_coarse=16,
        seed=seed,
        source_label="E3C_layer3_en_fr_permissive" if permissive_only else "E3C_layer3_en_fr",
    )
    model.metadata["lang_counts"] = dict(lang_counts)
    model.metadata["license_counts"] = dict(lic_counts)
    model.metadata["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    saved = model.save(out_path)
    return {**model.metadata, "saved_path": str(saved)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from-scratch EN/FR medical embeddings on E3C Layer 3.")
    parser.add_argument("--zip-path", type=Path, default=Path("data/raw/open_clinical/E3C-Corpus-main.zip"))
    parser.add_argument("--out", type=Path, default=Path("models/e3c_self_supervised_en_fr.npz"))
    parser.add_argument("--max-per-lang", type=int, default=4000)
    parser.add_argument("--max-vocab", type=int, default=16000)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--include-nc", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    info = train_from_e3c_zip(
        zip_path=args.zip_path,
        out_path=args.out,
        max_per_lang=args.max_per_lang,
        max_vocab=args.max_vocab,
        dim=args.dim,
        permissive_only=not args.include_nc,
        seed=args.seed,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
