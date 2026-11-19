"""Download, validate, and split real open-license EN/FR clinical corpora into AnnotatedDocument JSONL.

Supported corpora:
  1. MACCROBAT2020 (EN, CC-BY-4.0) — Figshare DOI 10.6084/m9.figshare.9764942
  2. PxCorpus (FR, CC-BY-4.0) — Zenodo DOI 10.5281/zenodo.6482586 / DrBenchmark/PxCorpus
  3. E3C Layer 1 (EN + FR, CC-BY-NC-4.0 corpus with per-doc licenses) — github.com/hltfbk/E3C-Corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from medresearch_platform.data.converters import (
    from_bc5cdr_iob,
    from_e3c_xmi,
    from_maccrobat_brat,
    from_ncbi_disease_pubtator,
    from_pxcorpus_row,
)
from medresearch_platform.data.schema import AnnotatedDocument

MACCROBAT2020_URL = "https://ndownloader.figshare.com/files/21489405"
E3C_ZIP_URL = "https://github.com/hltfbk/E3C-Corpus/archive/refs/heads/main.zip"
PXCORPUS_ROWS_API = "https://datasets-server.huggingface.co/rows"
NCBI_DISEASE_URLS = {
    "train": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip",
    "val": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip",
    "test": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip",
}
BC5CDR_IOB_URLS = {
    "train": "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-IOB/train.tsv",
    "val": "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-IOB/devel.tsv",
    "test": "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-IOB/test.tsv",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _deterministic_split(doc_id: str) -> str:
    bucket = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _download_file(client: httpx.Client, url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=65536):
                fh.write(chunk)
        tmp.replace(dest)
    return dest


def load_maccrobat(client: httpx.Client, raw_dir: Path) -> list[AnnotatedDocument]:
    zip_path = _download_file(client, MACCROBAT2020_URL, raw_dir / "MACCROBAT2020.zip")
    docs: list[AnnotatedDocument] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        txt_map = {Path(n).stem: n for n in zf.namelist() if n.endswith(".txt")}
        ann_map = {Path(n).stem: n for n in zf.namelist() if n.endswith(".ann")}
        for stem in sorted(set(txt_map) & set(ann_map)):
            txt = zf.read(txt_map[stem]).decode("utf-8", errors="replace")
            ann = zf.read(ann_map[stem]).decode("utf-8", errors="replace")
            doc_id = f"maccrobat_{stem}"
            doc = from_maccrobat_brat(txt, ann, doc_id=doc_id)
            doc.split = _deterministic_split(doc_id)
            errs = doc.validate_spans()
            if errs:
                raise ValueError(f"MACCROBAT span validation failed on {doc_id}: {errs[:3]}")
            docs.append(doc)
    return docs


def load_pxcorpus(client: httpx.Client, raw_dir: Path) -> list[AnnotatedDocument]:
    cache_path = raw_dir / "pxcorpus_rows.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        tag_names: list[str] = cached["tag_names"]
        splits_rows: dict[str, list[dict[str, Any]]] = cached["splits"]
    else:
        tag_names = []
        splits_rows = {}
        for hf_split, canon_split in (("train", "train"), ("validation", "val"), ("test", "test")):
            collected: list[dict[str, Any]] = []
            offset = 0
            page_size = 100
            while True:
                resp = client.get(
                    PXCORPUS_ROWS_API,
                    params={
                        "dataset": "DrBenchmark/PxCorpus",
                        "config": "default",
                        "split": hf_split,
                        "offset": offset,
                        "length": page_size,
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
                if not tag_names:
                    for feat in payload.get("features", []):
                        if feat.get("name") == "ner_tags":
                            tag_names = list(feat["type"]["feature"]["names"])
                rows = [item["row"] for item in payload.get("rows", [])]
                if not rows:
                    break
                collected.extend(rows)
                if len(rows) < page_size:
                    break
                offset += len(rows)
            splits_rows[canon_split] = collected
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"tag_names": tag_names, "splits": splits_rows}, ensure_ascii=False),
            encoding="utf-8",
        )

    docs: list[AnnotatedDocument] = []
    for canon_split, rows in splits_rows.items():
        for row in rows:
            doc = from_pxcorpus_row(row, tag_names, split=canon_split)
            if doc is None:
                continue
            errs = doc.validate_spans()
            if errs:
                raise ValueError(f"PxCorpus span validation failed on {doc.doc_id}: {errs[:3]}")
            docs.append(doc)
    return docs


def load_e3c(client: httpx.Client, raw_dir: Path) -> list[AnnotatedDocument]:
    zip_path = _download_file(client, E3C_ZIP_URL, raw_dir / "E3C-Corpus-main.zip")
    docs: list[AnnotatedDocument] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".xml"):
                continue
            lang: str | None = None
            if "/data_annotation/English/layer1/" in name:
                lang = "en"
            elif "/data_annotation/French/layer1/" in name:
                lang = "fr"
            if lang is None:
                continue
            xmi_text = zf.read(name).decode("utf-8", errors="replace")
            doc_id = f"e3c_{lang}_{Path(name).stem}"
            doc = from_e3c_xmi(xmi_text, doc_id=doc_id, language=lang)
            if doc is None:
                continue
            doc.split = _deterministic_split(doc_id)
            errs = doc.validate_spans()
            if errs:
                raise ValueError(f"E3C span validation failed on {doc_id}: {errs[:3]}")
            docs.append(doc)
    return docs


def load_ncbi_disease(client: httpx.Client, raw_dir: Path) -> list[AnnotatedDocument]:
    docs: list[AnnotatedDocument] = []
    for sp, url in NCBI_DISEASE_URLS.items():
        zip_path = _download_file(client, url, raw_dir / f"ncbi_disease_{sp}.zip")
        with zipfile.ZipFile(zip_path, "r") as zf:
            txt_names = [n for n in zf.namelist() if n.endswith(".txt")]
            if not txt_names:
                continue
            raw_txt = zf.read(txt_names[0]).decode("utf-8", errors="replace")
            parsed = from_ncbi_disease_pubtator(raw_txt, split=sp)
            for doc in parsed:
                errs = doc.validate_spans()
                if errs:
                    raise ValueError(f"NCBI Disease span validation failed on {doc.doc_id}: {errs[:3]}")
                docs.append(doc)
    return docs


def load_bc5cdr(client: httpx.Client, raw_dir: Path) -> list[AnnotatedDocument]:
    docs: list[AnnotatedDocument] = []
    for sp, url in BC5CDR_IOB_URLS.items():
        tsv_path = _download_file(client, url, raw_dir / f"bc5cdr_{sp}.tsv")
        raw_txt = tsv_path.read_text(encoding="utf-8", errors="replace")
        parsed = from_bc5cdr_iob(raw_txt, split=sp)
        for doc in parsed:
            errs = doc.validate_spans()
            if errs:
                raise ValueError(f"BC5CDR span validation failed on {doc.doc_id}: {errs[:3]}")
            docs.append(doc)
    return docs


def _write_jsonl(path: Path, docs: list[AnnotatedDocument]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
    return _sha256_file(path)


def build_open_clinical_corpora(
    *,
    raw_dir: Path = Path("data/raw/open_clinical"),
    out_dir: Path = Path("data/processed/open_clinical"),
    corpora: tuple[str, ...] = ("maccrobat", "pxcorpus", "e3c"),
    permissive_only: bool = False,
) -> dict[str, Any]:
    all_docs: list[AnnotatedDocument] = []
    by_corpus: dict[str, list[AnnotatedDocument]] = {}

    with httpx.Client(timeout=60.0, follow_redirects=True, headers={"User-Agent": "medical_ai/0.2.0"}) as client:
        if "maccrobat" in corpora:
            mc_docs = load_maccrobat(client, raw_dir)
            by_corpus["maccrobat"] = mc_docs
            all_docs.extend(mc_docs)
        if "pxcorpus" in corpora:
            px_docs = load_pxcorpus(client, raw_dir)
            by_corpus["pxcorpus"] = px_docs
            all_docs.extend(px_docs)
        if "e3c" in corpora and not permissive_only:
            e3c_docs = load_e3c(client, raw_dir)
            by_corpus["e3c"] = e3c_docs
            all_docs.extend(e3c_docs)
        if "bc5cdr" in corpora:
            bc_docs = load_bc5cdr(client, raw_dir)
            by_corpus["bc5cdr"] = bc_docs
            all_docs.extend(bc_docs)
        if "ncbi_disease" in corpora:
            ncbi_docs = load_ncbi_disease(client, raw_dir)
            by_corpus["ncbi_disease"] = ncbi_docs
            all_docs.extend(ncbi_docs)

    splits_out: dict[str, list[AnnotatedDocument]] = {"train": [], "val": [], "test": []}
    for d in all_docs:
        sp = d.split if d.split in splits_out else "train"
        splits_out[sp].append(d)

    file_hashes: dict[str, str] = {}
    for sp, sp_docs in splits_out.items():
        p = out_dir / f"{sp}.jsonl"
        file_hashes[f"{sp}.jsonl"] = _write_jsonl(p, sp_docs)

    for cname, cdocs in by_corpus.items():
        for sp in ("train", "val", "test"):
            sub = [d for d in cdocs if (d.split or "train") == sp]
            p = out_dir / "by_corpus" / f"{cname}_{sp}.jsonl"
            file_hashes[f"by_corpus/{cname}_{sp}.jsonl"] = _write_jsonl(p, sub)

    ent_counts = Counter(e.type for d in all_docs for e in d.entities)
    lang_counts = Counter(d.language for d in all_docs)
    lic_counts = Counter(str(d.metadata.get("license", "unknown")) for d in all_docs)

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "permissive_only": permissive_only,
        "total_docs": len(all_docs),
        "split_counts": {sp: len(v) for sp, v in splits_out.items()},
        "by_corpus_counts": {k: len(v) for k, v in by_corpus.items()},
        "by_language_counts": dict(sorted(lang_counts.items())),
        "by_license_counts": dict(sorted(lic_counts.items())),
        "entity_counts": dict(sorted(ent_counts.items())),
        "total_entities": sum(ent_counts.values()),
        "total_relations": sum(len(d.relations) for d in all_docs),
        "total_timex": sum(len(d.timex) for d in all_docs),
        "files_sha256": file_hashes,
    }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert open-license EN/FR clinical corpora.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/open_clinical"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/open_clinical"))
    parser.add_argument(
        "--corpora",
        type=str,
        default="maccrobat,pxcorpus,e3c",
        help="Comma-separated list of corpora: maccrobat,pxcorpus,e3c,bc5cdr,ncbi_disease",
    )
    parser.add_argument(
        "--permissive-only",
        action="store_true",
        help="Include ONLY CC-BY-4.0 / Public Domain corpora (excludes CC-BY-NC-4.0 E3C)",
    )
    args = parser.parse_args()
    selected = tuple(c.strip().lower() for c in args.corpora.split(",") if c.strip())
    manifest = build_open_clinical_corpora(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        corpora=selected,
        permissive_only=args.permissive_only,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
