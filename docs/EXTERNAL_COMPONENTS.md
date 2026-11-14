# External Components & Document Utilities

Per `MASTER_PROMPT.md` Part B.1:
> **Primary research models are trained from scratch. No pretrained weights** (no BERT/CamemBERT, no pretrained word vectors, no pretrained tokenizers) in the primary research track.
> OCR engines (Tesseract) and document utilities (`pypdf`, `pdfplumber`, `PyMuPDF`, `python-docx`) **may** be used as tools, clearly documented here.

---

## Allowed External Tools (Non-Research-Model Utilities)

| Component | Purpose | Track / Role | Status |
|---|---|---|---|
| `pypdf` | Digital PDF text extraction | Document parsing utility | Active (`pyproject.toml`) |
| `python-docx` | DOCX paragraph & table extraction | Document parsing utility | Active (`pyproject.toml`) |
| `Pillow` | Image decoding & format conversion | Document/OCR utility | Active (`pyproject.toml`) |
| `reportlab` | Synthetic PDF rendering for scan/digital tests | Test & synthetic data utility | Active (`pyproject.toml`) |
| **Tesseract 5 (`eng+fra`) + `pytesseract`** | Real-world scanned PDF/image OCR | Document OCR tool (Phase 5) | Planned (ADR-0007) |
| **PyMuPDF (`fitz`)** | Page rendering (300 dpi) & word bounding boxes | Document layout/highlight tool (Phase 5) | Planned (ADR-0007) |
| **OpenCV (`opencv-python-headless`)** | Scan preprocessing (deskew, denoise, adaptive threshold) | OCR preprocessing tool (Phase 5) | Planned (ADR-0007) |
| **`pdfplumber`** | Simple lab report table extraction | Table parsing tool (Phase 5) | Planned |
| **spaCy (`spacy.blank("en")`, `spacy.blank("fr")`)** | Blank rule-based tokenization only (never pretrained `en_core_*`/`fr_core_*` models) | Rule-based tokenizer option (Phase 3) | Planned |

---

## Strictly Prohibited in Primary Research Track

- Pretrained language models (BERT, RoBERTa, CamemBERT, DrBERT, BioBERT, PubMedBERT, LLMs) as feature extractors or fine-tuned backbones.
- Pretrained word embeddings (Word2Vec, GloVe, fastText) or pretrained subword tokenizers.
- Cloud inference APIs for clinical extraction.
