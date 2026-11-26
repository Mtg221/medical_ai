"""Permanent corpus audit script: detects doubled text, offset mismatches, and train/test overlap."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medresearch_platform.data.schema import AnnotatedDocument

_DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|PAT-[A-Za-z]{2}-\d+|\w+|[^\w\s]", re.UNICODE)


def _load_split(path: Path) -> list[AnnotatedDocument]:
    docs: list[AnnotatedDocument] = []
    if not path.exists():
        return docs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(AnnotatedDocument.from_dict(json.loads(line)))
    return docs


def _mask_digits(sentence: str) -> str:
    return _DIGIT_RE.sub("<NUM>", " ".join(sentence.strip().split())).lower()


def _find_doubled_spans(doc: AnnotatedDocument) -> list[str]:
    hits: list[str] = []
    for ent in doc.entities:
        if len(ent.text.strip()) >= 3 and (ent.text + ent.text) in doc.text:
            hits.append(f"entity:{ent.type}:{ent.text!r}")
    for tmx in doc.timex:
        if len(tmx.text.strip()) >= 3 and (tmx.text + tmx.text) in doc.text:
            hits.append(f"timex:{tmx.type}:{tmx.text!r}")
    return hits


def _find_adjacent_duplicate_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    hits: list[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - 2 * n + 1):
            g1 = tokens[i : i + n]
            g2 = tokens[i + n : i + 2 * n]
            if g1 == g2 and any(t.isalnum() for t in g1):
                hits.append(" ".join(g1))
    return hits


def audit_corpus(data_dir: Path) -> dict[str, Any]:
    """Run integrity, doubled-text, and train/test overlap checks across all splits in `data_dir`."""
    split_files = sorted(data_dir.glob("*.jsonl"))
    splits: dict[str, list[AnnotatedDocument]] = {
        p.stem: _load_split(p) for p in split_files
    }

    report: dict[str, Any] = {
        "data_dir": str(data_dir),
        "splits": {},
        "overlap_vs_train": {},
    }

    for split_name, docs in splits.items():
        total_spans = 0
        offset_ok = 0
        rel_ok = 0
        total_rels = 0
        doubled_span_docs = 0
        doubled_span_examples: list[str] = []
        ngram_dup_docs = 0
        vocab: Counter[str] = Counter()
        unique_ent_strings: set[str] = set()

        for d in docs:
            ent_ids = {e.id for e in d.entities}
            for e in d.entities:
                total_spans += 1
                if d.text[e.start : e.end] == e.text:
                    offset_ok += 1
                unique_ent_strings.add(e.text.lower())
            for t in d.timex:
                total_spans += 1
                if d.text[t.start : t.end] == t.text:
                    offset_ok += 1
            for r in d.relations:
                total_rels += 1
                if r.head in ent_ids and r.tail in ent_ids:
                    rel_ok += 1

            ds = _find_doubled_spans(d)
            if ds:
                doubled_span_docs += 1
                if len(doubled_span_examples) < 10:
                    doubled_span_examples.append(f"{d.doc_id}: {ds[0]}")
            if _find_adjacent_duplicate_ngrams(d.text):
                ngram_dup_docs += 1

            for tok in _TOKEN_RE.findall(d.text.lower()):
                vocab[tok] += 1

        total_tokens = sum(vocab.values())
        report["splits"][split_name] = {
            "docs": len(docs),
            "offset_match_pct": round(100.0 * offset_ok / total_spans, 2) if total_spans else 100.0,
            "relation_id_match_pct": round(100.0 * rel_ok / total_rels, 2) if total_rels else 100.0,
            "doubled_span_docs": doubled_span_docs,
            "adjacent_ngram_dup_docs": ngram_dup_docs,
            "doubled_examples": doubled_span_examples,
            "unique_entity_strings": len(unique_ent_strings),
            "vocab_size": len(vocab),
            "total_tokens": total_tokens,
            "type_token_ratio": round(len(vocab) / total_tokens, 4) if total_tokens else 0.0,
        }

    train_docs = splits.get("train", [])
    if train_docs:
        train_ents_all = {e.text.lower() for d in train_docs for e in d.entities}
        train_ents_by_type: dict[str, set[str]] = defaultdict(set)
        train_ents_by_lang: dict[str, set[str]] = defaultdict(set)
        train_sents: set[str] = set()
        for d in train_docs:
            for e in d.entities:
                train_ents_by_type[e.type].add(e.text.lower())
                train_ents_by_lang[d.language].add(e.text.lower())
            for s in _SENT_SPLIT_RE.split(d.text):
                masked = _mask_digits(s)
                if masked:
                    train_sents.add(masked)

        for split_name, docs in splits.items():
            if split_name == "train" or not docs:
                continue
            eval_ents = {e.text.lower() for d in docs for e in d.entities}
            seen_ents = eval_ents & train_ents_all

            by_type: dict[str, float] = {}
            eval_by_type: dict[str, set[str]] = defaultdict(set)
            eval_by_lang: dict[str, set[str]] = defaultdict(set)
            eval_sents: list[str] = []
            for d in docs:
                for e in d.entities:
                    eval_by_type[e.type].add(e.text.lower())
                    eval_by_lang[d.language].add(e.text.lower())
                for s in _SENT_SPLIT_RE.split(d.text):
                    masked = _mask_digits(s)
                    if masked:
                        eval_sents.append(masked)

            for etype, sset in sorted(eval_by_type.items()):
                by_type[etype] = round(100.0 * len(sset & train_ents_by_type[etype]) / len(sset), 1) if sset else 0.0
            by_lang = {
                lang: round(100.0 * len(sset & train_ents_by_lang[lang]) / len(sset), 1) if sset else 0.0
                for lang, sset in sorted(eval_by_lang.items())
            }
            sent_overlap = (
                round(100.0 * sum(1 for s in eval_sents if s in train_sents) / len(eval_sents), 1)
                if eval_sents
                else 0.0
            )
            report["overlap_vs_train"][split_name] = {
                "entity_string_overlap_pct": round(100.0 * len(seen_ents) / len(eval_ents), 1) if eval_ents else 0.0,
                "by_type_pct": by_type,
                "by_language_pct": by_lang,
                "masked_sentence_overlap_pct": sent_overlap,
                "total_eval_sentences": len(eval_sents),
            }

    return report


def format_audit_report(report: dict[str, Any]) -> str:
    lines: list[str] = [
        f"=== Corpus Audit: {report['data_dir']} ===",
        "",
        "Split Integrity & Lexical Diversity:",
        f"{'Split':<22} {'Docs':>5} {'Offset%':>8} {'RelID%':>7} {'DoubledDocs':>12} {'NGramDupDocs':>13} {'UniqEnts':>9} {'Vocab':>7} {'TTR':>7}",
        "-" * 98,
    ]
    for split_name, info in report["splits"].items():
        lines.append(
            f"{split_name:<22} {info['docs']:>5} {info['offset_match_pct']:>7.2f}% "
            f"{info['relation_id_match_pct']:>6.2f}% {info['doubled_span_docs']:>12} "
            f"{info['adjacent_ngram_dup_docs']:>13} {info['unique_entity_strings']:>9} "
            f"{info['vocab_size']:>7} {info['type_token_ratio']:>7.4f}"
        )
        for ex in info["doubled_examples"][:3]:
            lines.append(f"  [doubled example] {ex}")

    if report["overlap_vs_train"]:
        lines.extend(["", "Overlap vs. `train` Split:"])
        for split_name, ov in report["overlap_vs_train"].items():
            lines.append(
                f"- {split_name}: unique_entity_overlap={ov['entity_string_overlap_pct']:.1f}% | "
                f"masked_sentence_overlap={ov['masked_sentence_overlap_pct']:.1f}% "
                f"({ov['total_eval_sentences']} sentences)"
            )
            lines.append(f"  by_language (%): {ov['by_language_pct']}")
            lines.append(f"  by_type (%):     {ov['by_type_pct']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit corpus integrity, doubled text, and train/test overlap")
    parser.add_argument("--data-dir", default="data/processed/v1", help="Directory containing split JSONL files")
    parser.add_argument("--out", default=None, help="Optional path to write text report")
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON report")
    parser.add_argument("--fail-on-doubled", action="store_true", help="Exit 1 if any doubled text is found")
    args = parser.parse_args(argv)

    rep = audit_corpus(Path(args.data_dir))
    txt = format_audit_report(rep)
    print(txt)
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(txt, encoding="utf-8")
    if args.json_out:
        j_p = Path(args.json_out)
        j_p.parent.mkdir(parents=True, exist_ok=True)
        j_p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")

    if args.fail_on_doubled:
        total_doubled = sum(v["doubled_span_docs"] for v in rep["splits"].values())
        if total_doubled > 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
