"""Orchestrator for the 5-stage Hybrid Medical Extraction Pipeline + ablation & JSON contract export."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from medresearch_platform.approaches.context_negex import (
    apply_context_and_relations,
    classify_assertion_context,
)
from medresearch_platform.approaches.neural_seq import NeuralBiLSTMCRFTagger
from medresearch_platform.approaches.tokenizer import tokenize_with_offsets
from medresearch_platform.data.schema import ASSERTION_CLASSES, AnnotatedDocument
from medresearch_platform.eval_harness.metrics import (
    DocumentPrediction,
    PredictedEntity,
    PredictedRelation,
    PredictedTimex,
)
from medresearch_platform.hybrid.persistence import (
    compute_corpus_hash,
    git_commit_hash,
    load_crf_model,
    load_npz_params,
    save_crf_model,
    save_npz_params,
    verify_manifest,
    write_manifest,
)
from medresearch_platform.hybrid.stage1_doc_understanding import Stage1DocumentUnderstanding
from medresearch_platform.hybrid.stage2_ner import CandidateEntity, Stage2EntityExtractor
from medresearch_platform.hybrid.stage3_context_relations import (
    BiLSTMAssertionClassifier,
    PiecewiseCNNRelationClassifier,
)
from medresearch_platform.hybrid.stage4_temporal import CRFTemporalTagger
from medresearch_platform.hybrid.stage5_integration import (
    TemperatureScaler,
    TransformerCandidateReranker,
    check_consistency,
    detect_missing_sections,
)


class HybridPipeline:
    """5-stage, >=6-technique Hybrid Medical Extraction Pipeline with ablation toggles."""

    def __init__(self, *, seed: int = 42, cfg: dict[str, Any] | None = None) -> None:
        self.seed = seed
        self.cfg = cfg or {}
        self.stage1 = Stage1DocumentUnderstanding(seed=seed)
        raw_max = self.cfg.get("max_train_docs", None)
        self.stage2 = Stage2EntityExtractor(
            hidden_dim=int(self.cfg.get("hidden_dim", 32)),
            epochs=int(self.cfg.get("epochs", 10)),
            lr=float(self.cfg.get("lr", 0.018)),
            max_train_docs=int(raw_max) if raw_max is not None else None,
            use_crf_ensemble=bool(self.cfg.get("use_crf_ensemble", False)),
            seed=seed,
        )
        self.assertion_model: BiLSTMAssertionClassifier | None = None
        self.relation_model: PiecewiseCNNRelationClassifier | None = None
        self.temporal_crf = CRFTemporalTagger(seed=seed)
        self.reranker = TransformerCandidateReranker(seed=seed)
        self.scaler = TemperatureScaler()
        self.corpus_id = str(self.cfg.get("dataset_dir", "in_memory"))
        self.corpus_hash = ""
        self.manifest: dict[str, Any] = {}
        self._fitted = False

    def fit(self, train_docs: list[AnnotatedDocument]) -> HybridPipeline:
        self.corpus_hash = compute_corpus_hash(train_docs)
        self.stage1.fit(train_docs)
        self.stage2.fit(train_docs)
        vocab = self.stage2.vocab

        if bool(self.cfg.get("use_assertion_bilstm", True)):
            self.assertion_model = BiLSTMAssertionClassifier(len(vocab), seed=self.seed).fit(
                train_docs, vocab, epochs=int(self.cfg.get("aux_epochs", 6)), seed=self.seed
            )
        else:
            self.assertion_model = None
        self.relation_model = PiecewiseCNNRelationClassifier(len(vocab), seed=self.seed).fit(
            train_docs, vocab, epochs=int(self.cfg.get("aux_epochs", 6)), seed=self.seed
        )
        self.temporal_crf.fit(train_docs)

        # Fit Transformer reranker & temperature scaler on shuffled EN+FR slices of train_docs
        rng = np.random.default_rng(self.seed)
        shuffled_docs = list(train_docs)
        rng.shuffle(shuffled_docs)
        self.reranker = TransformerCandidateReranker(vocab_size=len(vocab), seed=self.seed)
        if bool(self.cfg.get("use_transformer_fusion", True)):
            cand_pairs = [(d, self.stage2.predict(d, use_rules_layer=True)) for d in shuffled_docs[:80]]
            self.reranker.fit_on_candidates(
                cand_pairs, vocab=vocab, epochs=int(self.cfg.get("aux_epochs", 6)), seed=self.seed
            )

        # Fit temperature scaler on held-out slice of train_docs
        cal_confs: list[float] = []
        cal_hits: list[int] = []
        for d in shuffled_docs[80:120]:
            gold_keys = {(e.type, e.start, e.end) for e in d.entities}
            for c in self.stage2.predict(d, use_rules_layer=True):
                cal_confs.append(c.entity.confidence)
                cal_hits.append(1 if (c.entity.type, c.entity.start, c.entity.end) in gold_keys else 0)
        self.scaler.fit(cal_confs, cal_hits)
        self._fitted = True
        return self

    def extract_document(
        self,
        doc: AnnotatedDocument,
        *,
        use_rules_layer: bool | None = None,
        use_assertion_bilstm: bool | None = None,
        use_relation_cnn: bool | None = None,
        use_temporal_crf: bool | None = None,
        use_transformer_fusion: bool | None = None,
    ) -> tuple[DocumentPrediction, dict[str, Any]]:
        if not self._fitted or self.relation_model is None:
            raise RuntimeError("HybridPipeline must be fitted before extract_document()")
        use_rules_layer = bool(self.cfg.get("use_rules_layer", True)) if use_rules_layer is None else use_rules_layer
        use_assertion_bilstm = (
            bool(self.cfg.get("use_assertion_bilstm", True))
            if use_assertion_bilstm is None
            else use_assertion_bilstm
        )
        use_relation_cnn = bool(self.cfg.get("use_relation_cnn", True)) if use_relation_cnn is None else use_relation_cnn
        use_temporal_crf = bool(self.cfg.get("use_temporal_crf", True)) if use_temporal_crf is None else use_temporal_crf
        use_transformer_fusion = (
            bool(self.cfg.get("use_transformer_fusion", True))
            if use_transformer_fusion is None
            else use_transformer_fusion
        )
        if use_assertion_bilstm and self.assertion_model is None:
            use_assertion_bilstm = False

        if len(doc.text) > 4000:
            return self._extract_long_document_chunked(
                doc,
                use_rules_layer=use_rules_layer,
                use_assertion_bilstm=use_assertion_bilstm,
                use_relation_cnn=use_relation_cnn,
                use_temporal_crf=use_temporal_crf,
                use_transformer_fusion=use_transformer_fusion,
            )

        timings: list[dict[str, Any]] = []

        # Stage 1: Document Understanding
        t0 = time.perf_counter()
        s1 = self.stage1.predict(doc.text)
        timings.append({"name": "stage1_doc_understanding", "ms": round((time.perf_counter() - t0) * 1000, 2)})

        # Stage 2: Entity Extraction
        t0 = time.perf_counter()
        cands = self.stage2.predict(doc, use_rules_layer=use_rules_layer)
        timings.append({"name": "stage2_ner", "ms": round((time.perf_counter() - t0) * 1000, 2)})

        # Stage 5a (optional text-reading Transformer reranker) or simple union
        t0 = time.perf_counter()
        vocab = self.stage2.vocab
        if use_transformer_fusion:
            fused_cands = self.reranker.rerank(
                cands, doc_text=doc.text, vocab=vocab, threshold=float(self.cfg.get("rerank_threshold", 0.25))
            )
        else:
            fused_cands = cands
        rerank_ms = (time.perf_counter() - t0) * 1000

        # Stage 3: Context (Assertion BiLSTM + ConText fusion) & Relations (CNN or proximity fallback)
        t0 = time.perf_counter()
        doc_toks = tokenize_with_offsets(doc.text)
        enriched_cands: list[CandidateEntity] = []
        for c in fused_cands:
            ctx_lbl = classify_assertion_context(doc.text, c.entity.start, c.entity.end, language=s1.language)
            if ctx_lbl == "present" and doc.language in ("en", "fr") and doc.language != s1.language:
                ctx_lbl = classify_assertion_context(doc.text, c.entity.start, c.entity.end, language=doc.language)
            if ctx_lbl != "present":
                assert_lbl = ctx_lbl
            elif use_assertion_bilstm and self.assertion_model is not None:
                probs = self.assertion_model.predict_proba(
                    doc.text, c.entity.start, c.entity.end, vocab, toks=doc_toks
                )
                best_idx = int(probs.argmax(axis=0))
                assert_lbl = ASSERTION_CLASSES[best_idx] if float(probs[best_idx]) >= 0.75 else ctx_lbl
            else:
                assert_lbl = ctx_lbl
            cal_conf = self.scaler.calibrate(c.entity.confidence)
            enriched_cands.append(
                CandidateEntity(
                    entity=PredictedEntity(
                        type=c.entity.type,
                        start=c.entity.start,
                        end=c.entity.end,
                        text=c.entity.text,
                        assertion=assert_lbl,
                        confidence=cal_conf,
                        needs_review=(cal_conf < 0.65),
                    ),
                    source_stage=c.source_stage,
                )
            )
        final_entities = [c.entity for c in enriched_cands]

        if use_relation_cnn:
            relations = self.relation_model.predict_relations(doc.text, final_entities, vocab)
        else:
            _, relations, _ = apply_context_and_relations(doc, final_entities)
        timings.append({"name": "stage3_context_relations", "ms": round((time.perf_counter() - t0) * 1000, 2)})

        # Stage 4: Temporal Analysis (CRF or regex fallback)
        t0 = time.perf_counter()
        if use_temporal_crf:
            timex = self.temporal_crf.predict(doc.text, doc.document_date)
        else:
            _, _, timex = apply_context_and_relations(doc, final_entities)
        timings.append({"name": "stage4_temporal", "ms": round((time.perf_counter() - t0) * 1000, 2)})

        # Stage 5b: Consistency checks & contract payload
        t0 = time.perf_counter()
        warnings = check_consistency(final_entities, relations, timex, doc.document_date)
        missing = detect_missing_sections(doc.text, s1.language)
        timings.append(
            {"name": "stage5_integration_validation", "ms": round(rerank_ms + (time.perf_counter() - t0) * 1000, 2)}
        )

        pred = DocumentPrediction(
            doc_id=doc.doc_id,
            language=doc.language,
            doc_type=doc.doc_type,
            predicted_doc_type=s1.doc_type,
            entities=final_entities,
            relations=relations,
            timex=timex,
        )

        ent_id_by_span = {(c.entity.type, c.entity.start, c.entity.end): f"e{i+1}" for i, c in enumerate(enriched_cands)}
        contract_json = {
            "schema_version": "1.0",
            "document": {
                "id": doc.doc_id,
                "language": s1.language,
                "language_confidence": s1.language_confidence,
                "doc_type": s1.doc_type,
                "doc_type_confidence": s1.doc_type_confidence,
                "document_date": doc.document_date,
                "sections": s1.sections,
            },
            "entities": [
                {
                    "id": f"e{i+1}",
                    "type": c.entity.type,
                    "text": c.entity.text,
                    "assertion": c.entity.assertion,
                    "confidence": c.entity.confidence,
                    "char_start": c.entity.start,
                    "char_end": c.entity.end,
                    "entity_type": c.entity.type,
                    "provenance": {"source_stage": c.source_stage},
                    "source": {"page": 1, "start": c.entity.start, "end": c.entity.end},
                    "produced_by": {"approach": "hybrid", "stage": c.source_stage, "model_version": "2.0.0"},
                    "review": {"status": "flagged" if c.entity.confidence < 0.65 else "unreviewed", "by": None},
                }
                for i, c in enumerate(enriched_cands)
            ],
            "relations": [
                {
                    "id": f"r{i+1}",
                    "type": r.type,
                    "head": next(
                        (eid for (et, s, e), eid in ent_id_by_span.items() if s == r.head_start and e == r.head_end),
                        None,
                    ),
                    "tail": next(
                        (eid for (et, s, e), eid in ent_id_by_span.items() if s == r.tail_start and e == r.tail_end),
                        None,
                    ),
                    "confidence": r.confidence,
                    "evidence": {"start": min(r.head_start, r.tail_start), "end": max(r.head_end, r.tail_end)},
                }
                for i, r in enumerate(relations)
            ],
            "temporal": [
                {
                    "id": f"t{i+1}",
                    "type": tx.type,
                    "text": tx.text,
                    "normalized": tx.normalized,
                    "anchor": "document_date",
                    "source": {"start": tx.start, "end": tx.end},
                }
                for i, tx in enumerate(timex)
            ],
            "missing_information": missing,
            "validation": {"passed": len(warnings) == 0, "warnings": warnings},
            "processing": {
                "pipeline": "hybrid-v2",
                "duration_ms": round(sum(t["ms"] for t in timings), 2),
                "stages": timings,
            },
        }
        contract_json["records"] = contract_json["entities"]
        return pred, contract_json

    def _extract_long_document_chunked(
        self,
        doc: AnnotatedDocument,
        *,
        use_rules_layer: bool,
        use_assertion_bilstm: bool,
        use_relation_cnn: bool,
        use_temporal_crf: bool,
        use_transformer_fusion: bool,
        chunk_max_chars: int = 3500,
    ) -> tuple[DocumentPrediction, dict[str, Any]]:
        """Process documents > 4,000 chars in sentence-window chunks to avoid sequence length limits."""
        t0 = time.perf_counter()
        s1 = self.stage1.predict(doc.text[:4000])
        all_entities: list[PredictedEntity] = []
        all_relations: list[PredictedRelation] = []
        all_timex: list[PredictedTimex] = []
        all_records: list[dict[str, Any]] = []
        warnings: list[str] = []

        # Split text on newline or sentence boundary <= chunk_max_chars
        chunks: list[tuple[int, str]] = []
        n = len(doc.text)
        pos = 0
        while pos < n:
            end = min(pos + chunk_max_chars, n)
            if end < n:
                nl = doc.text.rfind("\n", pos + 500, end)
                if nl != -1:
                    end = nl + 1
                else:
                    dot = doc.text.rfind(". ", pos + 500, end)
                    if dot != -1:
                        end = dot + 2
            chunks.append((pos, doc.text[pos:end]))
            pos = end

        # Deduplicate identical chunks to process repeated boilerplate in O(1) while preserving all span offsets
        chunk_cache: dict[str, tuple[DocumentPrediction, dict[str, Any]]] = {}
        is_huge = n > 25_000
        for offset, chunk_txt in chunks:
            if not chunk_txt.strip():
                continue
            cached = chunk_cache.get(chunk_txt)
            if cached is None:
                sub_doc = AnnotatedDocument(
                    doc_id=f"{doc.doc_id}_chunk_{offset}",
                    patient_id=doc.patient_id,
                    language=doc.language,
                    doc_type=doc.doc_type,
                    document_date=doc.document_date,
                    source=doc.source,
                    text=chunk_txt,
                )
                cached = self.extract_document(
                    sub_doc,
                    use_rules_layer=use_rules_layer,
                    use_assertion_bilstm=use_assertion_bilstm and not is_huge,
                    use_relation_cnn=use_relation_cnn and not is_huge,
                    use_temporal_crf=use_temporal_crf,
                    use_transformer_fusion=use_transformer_fusion and not is_huge,
                )
                chunk_cache[chunk_txt] = cached
            c_pred, c_json = cached
            for ent in c_pred.entities:
                all_entities.append(
                    PredictedEntity(
                        type=ent.type,
                        start=ent.start + offset,
                        end=ent.end + offset,
                        text=ent.text,
                        assertion=ent.assertion,
                        confidence=ent.confidence,
                    )
                )
            for rel in c_pred.relations:
                all_relations.append(
                    PredictedRelation(
                        type=rel.type,
                        head_start=rel.head_start + offset,
                        head_end=rel.head_end + offset,
                        tail_start=rel.tail_start + offset,
                        tail_end=rel.tail_end + offset,
                        confidence=rel.confidence,
                    )
                )
            for tx in c_pred.timex:
                all_timex.append(
                    PredictedTimex(
                        type=tx.type,
                        start=tx.start + offset,
                        end=tx.end + offset,
                        text=tx.text,
                        normalized=tx.normalized,
                    )
                )
            for rec in c_json.get("records", []):
                shifted = dict(rec)
                shifted["char_start"] = int(rec.get("char_start", 0)) + offset
                shifted["char_end"] = int(rec.get("char_end", 0)) + offset
                all_records.append(shifted)

        dur_ms = round((time.perf_counter() - t0) * 1000, 2)
        pred = DocumentPrediction(
            doc_id=doc.doc_id,
            language=doc.language,
            doc_type=doc.doc_type,
            predicted_doc_type=s1.doc_type,
            entities=all_entities,
            relations=all_relations,
            timex=all_timex,
        )
        contract_json = {
            "schema_version": "1.0",
            "document": {
                "id": doc.doc_id,
                "language": s1.language,
                "language_confidence": s1.language_confidence,
                "doc_type": s1.doc_type,
                "doc_type_confidence": s1.doc_type_confidence,
                "document_date": doc.document_date,
                "sections": s1.sections,
            },
            "entities": all_records,
            "records": all_records,
            "relations": [],
            "temporal": [],
            "missing_information": [],
            "validation": {"passed": True, "warnings": warnings},
            "processing": {
                "pipeline": "hybrid-v2-chunked",
                "duration_ms": dur_ms,
                "chunks": len(chunks),
            },
        }
        return pred, contract_json

    def save(self, model_dir: Path | str, *, corpus_id: str | None = None) -> Path:
        if not self._fitted or self.stage2.model is None or self.relation_model is None:
            raise RuntimeError("HybridPipeline must be fitted before save()")
        out_dir = Path(model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.stage1, out_dir / "stage1.joblib")
        vocab_payload = {
            "max_size": self.stage2.vocab.max_size,
            "min_freq": self.stage2.vocab.min_freq,
            "word_to_id": self.stage2.vocab.word_to_id,
        }
        (out_dir / "vocab.json").write_text(json.dumps(vocab_payload, ensure_ascii=False), encoding="utf-8")

        save_npz_params(
            out_dir / "stage2_bilstm_crf.npz",
            self.stage2.model.params,
            {
                "vocab_size": int(self.stage2.model.params["E"].shape[0]),
                "emb_dim": int(self.stage2.model.params["E"].shape[1]),
                "hidden_dim": int(self.stage2.hidden_dim),
            },
        )
        if self.stage2.crf_model is not None:
            save_crf_model(self.stage2.crf_model, out_dir / "stage2_crf.crfsuite")
        if self.assertion_model is not None:
            save_npz_params(
                out_dir / "stage3_assertion.npz",
                self.assertion_model.params,
                {
                    "vocab_size": self.assertion_model.vocab_size,
                    "emb_dim": self.assertion_model.emb_dim,
                    "hidden_dim": self.assertion_model.hidden_dim,
                },
            )
        save_npz_params(
            out_dir / "stage3_relation.npz",
            self.relation_model.params,
            {
                "vocab_size": int(self.relation_model.params["E"].shape[0]),
                "emb_dim": self.relation_model.emb_dim,
                "pos_dim": self.relation_model.pos_dim,
                "n_filters": self.relation_model.n_filters,
            },
        )
        save_crf_model(self.temporal_crf.crf, out_dir / "stage4_temporal.crfsuite")
        save_npz_params(
            out_dir / "stage5_reranker.npz",
            self.reranker.params,
            {
                "vocab_size": self.reranker.vocab_size,
                "emb_dim": self.reranker.emb_dim,
                "d_model": self.reranker.d_model,
            },
        )

        manifest_path = write_manifest(
            out_dir,
            {
                "model_type": "hybrid",
                "seed": self.seed,
                "cfg": self.cfg,
                "corpus_id": corpus_id or self.corpus_id,
                "corpus_hash": self.corpus_hash,
                "git_commit": git_commit_hash(),
                "temperature": float(self.scaler.temperature),
                "has_stage2_crf": self.stage2.crf_model is not None,
                "has_assertion_bilstm": self.assertion_model is not None,
            },
        )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return out_dir

    @classmethod
    def load(cls, model_dir: Path | str) -> HybridPipeline:
        in_dir = Path(model_dir)
        manifest = verify_manifest(in_dir)
        if manifest.get("model_type") != "hybrid":
            raise ValueError(f"Expected model_type='hybrid', got {manifest.get('model_type')!r}")

        pipe = cls(seed=int(manifest["seed"]), cfg=dict(manifest.get("cfg", {})))
        pipe.corpus_id = str(manifest.get("corpus_id", "unknown"))
        pipe.corpus_hash = str(manifest.get("corpus_hash", ""))
        pipe.manifest = manifest

        pipe.stage1 = joblib.load(in_dir / "stage1.joblib")
        for clf_attr in ("lang_clf", "doc_type_clf"):
            clf_obj = getattr(pipe.stage1, clf_attr, None)
            if clf_obj is not None and not hasattr(clf_obj, "multi_class"):
                setattr(clf_obj, "multi_class", "auto")

        vocab_data = json.loads((in_dir / "vocab.json").read_text(encoding="utf-8"))
        pipe.stage2.vocab.max_size = int(vocab_data["max_size"])
        pipe.stage2.vocab.min_freq = int(vocab_data["min_freq"])
        pipe.stage2.vocab.word_to_id = {str(k): int(v) for k, v in vocab_data["word_to_id"].items()}

        s2_params, s2_meta = load_npz_params(in_dir / "stage2_bilstm_crf.npz")
        s2_model = NeuralBiLSTMCRFTagger(
            int(s2_meta["vocab_size"]),
            emb_dim=int(s2_meta["emb_dim"]),
            hidden_dim=int(s2_meta["hidden_dim"]),
            seed=pipe.seed,
        )
        s2_model.params = s2_params
        pipe.stage2.model = s2_model

        s2_crf_path = in_dir / "stage2_crf.crfsuite"
        pipe.stage2.crf_model = load_crf_model(s2_crf_path) if s2_crf_path.exists() else None

        s3a_path = in_dir / "stage3_assertion.npz"
        if s3a_path.exists():
            a_params, a_meta = load_npz_params(s3a_path)
            a_model = BiLSTMAssertionClassifier(
                int(a_meta["vocab_size"]),
                emb_dim=int(a_meta["emb_dim"]),
                hidden_dim=int(a_meta["hidden_dim"]),
                seed=pipe.seed,
            )
            a_model.params = a_params
            pipe.assertion_model = a_model
        else:
            pipe.assertion_model = None

        r_params, r_meta = load_npz_params(in_dir / "stage3_relation.npz")
        r_model = PiecewiseCNNRelationClassifier(
            int(r_meta["vocab_size"]),
            emb_dim=int(r_meta["emb_dim"]),
            pos_dim=int(r_meta["pos_dim"]),
            n_filters=int(r_meta["n_filters"]),
            seed=pipe.seed,
        )
        r_model.params = r_params
        pipe.relation_model = r_model

        pipe.temporal_crf.crf = load_crf_model(in_dir / "stage4_temporal.crfsuite")
        pipe.temporal_crf._fitted = True

        rr_params, rr_meta = load_npz_params(in_dir / "stage5_reranker.npz")
        rr_model = TransformerCandidateReranker(
            vocab_size=int(rr_meta["vocab_size"]),
            emb_dim=int(rr_meta["emb_dim"]),
            d_model=int(rr_meta["d_model"]),
            seed=pipe.seed,
        )
        rr_model.params = rr_params
        pipe.reranker = rr_model

        pipe.scaler.temperature = float(manifest.get("temperature", 1.0))
        pipe._fitted = True
        return pipe


def predict_hybrid(
    train_docs: list[AnnotatedDocument],
    eval_docs: list[AnnotatedDocument],
    seed: int,
    cfg: dict[str, Any],
) -> list[DocumentPrediction]:
    """Runner-compatible entrypoint for the 5-stage Hybrid pipeline."""
    pipe = HybridPipeline(seed=seed, cfg=cfg).fit(train_docs)
    return [
        pipe.extract_document(
            d,
            use_rules_layer=bool(cfg.get("use_rules_layer", True)),
            use_assertion_bilstm=bool(cfg.get("use_assertion_bilstm", True)),
            use_relation_cnn=bool(cfg.get("use_relation_cnn", True)),
            use_temporal_crf=bool(cfg.get("use_temporal_crf", True)),
            use_transformer_fusion=bool(cfg.get("use_transformer_fusion", True)),
        )[0]
        for d in eval_docs
    ]
