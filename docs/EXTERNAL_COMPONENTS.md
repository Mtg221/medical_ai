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
| **Tesseract 5 (`eng+fra`) + `pytesseract`** | Real-world scanned PDF/image OCR | Document OCR tool (Phase 5) | Active (`pyproject.toml`) |
| **`pypdfium2` + `pdfplumber`** | 300 dpi PDF page rendering (`pypdfium2`), word bounding boxes & lab table extraction (`pdfplumber`). Replaces `PyMuPDF` on macOS x86_64 Python 3.13 where `pymupdf` has no prebuilt wheel. | Document layout/render/table tool (Phase 5) | Active (`pyproject.toml`) |
| **OpenCV (`opencv-python-headless`)** | Scan preprocessing (upscale, deskew, denoise, adaptive threshold, border pad) | OCR preprocessing tool (Phase 5) | Active (`pyproject.toml`) |
| **spaCy (`spacy.blank("en")`, `spacy.blank("fr")`)** | Blank rule-based tokenization only (never pretrained `en_core_*`/`fr_core_*` models) | Rule-based tokenizer option (Phase 3) | Planned |

---

## Strictly Prohibited in Primary Research Track

- Pretrained language models (BERT, RoBERTa, CamemBERT, DrBERT, BioBERT, PubMedBERT, LLMs) as feature extractors or fine-tuned backbones.
- Pretrained word embeddings (Word2Vec, GloVe, fastText) or pretrained subword tokenizers.
- Cloud inference APIs for clinical extraction.
