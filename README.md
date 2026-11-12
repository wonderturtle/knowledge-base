# Document Search & Classroom RAG Lab

A lightweight local Flask application designed for a Big Data classroom exercise. Students upload their own documents, inspect extracted text, compare keyword search (SQLite FTS5) with multilingual semantic search (FastEmbed), and generate grounded answers using a local Ollama model.

---

## Architecture & Project Structure

```text
knowledge-base/
├── app.py                      # Flask entry point (app = create_app())
├── requirements.txt            # Project dependencies
├── README.md                   # Documentation, schema, and verification log
├── schema.sql                  # Database schema definition
├── .gitignore                  # Git ignore rules (.env, instance/, caches)
├── .env.example                # Ollama environment configuration template
├── .env                        # Local configuration (WSL-to-Windows host IP)
├── document_search/
│   ├── __init__.py             # Flask factory & SQLAlchemy init
│   ├── config.py               # App configuration & classroom limits
│   ├── models.py               # SQLAlchemy models (Document, Passage)
│   ├── services.py             # Text extraction, passage splitting, search
│   ├── ollama_client.py        # Local Ollama HTTP chat & tags client
│   ├── blueprints/
│   │   ├── __init__.py         # Blueprint package exports
│   │   ├── documents.py        # /documents: list, upload, parsed view, index, delete
│   │   ├── search.py           # /search: keyword/semantic retrieval & Ollama answer
│   │   └── chat.py             # /chat: grounded & general chat with local model
│   └── templates/
│       ├── base.html           # Shared layout & navigation
│       ├── documents/
│       │   ├── index.html      # Document list & upload form
│       │   └── parsed.html     # Dedicated parsed document inspection view
│       ├── search/
│       │   └── index.html      # Search comparison & grounded answer screen
│       └── chat/
│           └── index.html      # Interactive chat with document toggle
├── static/
│   └── style.css               # Clean, responsive plain CSS
└── instance/                   # Local database, uploads, and model cache
    ├── documents.db            # SQLite database with FTS5 index
    ├── uploads/                # Stored files (UUID names)
    └── model_cache/            # FastEmbed ONNX model cache
```

---

## Environment Setup inside WSL

The project lives in `~/knowledge-base` while the Python virtual environment is kept in `~/venv_knowledge-base` outside the repository:

```bash
mkdir -p ~/knowledge-base
if [ ! -d ~/venv_knowledge-base ]; then
    python3 -m venv ~/venv_knowledge-base
fi
source ~/venv_knowledge-base/bin/activate
cd ~/knowledge-base
python -c "import sys; print(sys.executable)"
```

Confirm that the interpreter output points to `~/venv_knowledge-base/bin/python`.

Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependency Versions (Tested on Python 3.14.4)
- **Flask**: 3.1.3
- **Flask-SQLAlchemy**: 3.1.1
- **SQLAlchemy**: 2.1.8
- **fastembed**: 0.9.1
- **pypdf**: 7.0.1
- **python-docx**: 1.2.0
- **beautifulsoup4**: 4.15.0
- **requests**: 2.34.2
- **python-dotenv**: 1.2.4

---

## Configuration & Local Ollama Connection

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Settings in `.env`:
```ini
# When Ollama runs on the Windows host and Flask runs inside WSL:
OLLAMA_BASE_URL=http://172.22.224.1:11434
OLLAMA_MODEL=ornith-1.5:9b
```
*(If Ollama runs natively inside Linux loopback, use `http://127.0.0.1:11434`)*.

---

## Model Warmup & Running the Application

Before class, warm the embedding model so local inference runs without network access during the lab:
```bash
source ~/venv_knowledge-base/bin/activate
cd ~/knowledge-base
flask --app app warm-model
```

Run the development server on port 5001:
```bash
python -m flask --app app run --host 127.0.0.1 --port 5001
```
Open your browser at: `http://127.0.0.1:5001` (redirects to `/documents`).

---

## Database Schema & Models

Data is managed with SQLAlchemy 2.x ORM models mapping to SQLite:

### `Document` Model (`documents` table)
- `id` (INTEGER, PK, Autoincrement)
- `original_name` (TEXT): Original filename for display
- `storage_name` (TEXT, UNIQUE): Generated UUID filename to prevent collisions and path injection
- `content_hash` (TEXT): SHA-256 digest to reject duplicate uploads
- `file_type` (TEXT): `pdf`, `docx`, `html`, or `txt`
- `parser_used` (TEXT): e.g. `pypdf`, `python-docx`, `BeautifulSoup`, `utf-8 plain text`
- `extracted_text` (TEXT): Persistent raw extracted text with section boundaries
- `extraction_warnings` (TEXT): JSON array of warnings (empty pages, unreadable tables)
- `status` (TEXT): `pending`, `indexed`, or `empty`
- `page_count` (INTEGER): Page count for PDFs (or NULL)
- `created_at` (DATETIME): Timestamp

### `Passage` Model (`passages` table)
- `id` (INTEGER, PK, Autoincrement)
- `document_id` (INTEGER, FK -> `documents.id` ON DELETE CASCADE)
- `seq` (INTEGER): Sequence number within document
- `text` (TEXT): Clean passage text
- `page_or_section` (TEXT): `page 1` (PDF) or `para-1`, `table-2`, `h1-1`
- `embedding` (TEXT): JSON-serialized 384-dimensional float vector
- `model_id` (TEXT): `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `embedding_dim` (INTEGER): 384
- `UNIQUE(document_id, seq)`

### FTS5 Full-Text Search Table (`passages_fts`)
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    text,
    content='passages',
    content_rowid='id'
);
```
Automatic triggers (`passages_ai`, `passages_ad`, `passages_au`) keep the FTS5 index synchronized with the `passages` table on insert, delete, and update. Foreign keys (`PRAGMA foreign_keys=ON`) ensure document deletion cascades to passages and FTS5.

---

## Passage Sizing & Embedding Strategy

- **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Supported Languages**: English, Italian, and ~50 other languages
- **Context Limit**: 128 tokens
- **Special Tokens**: 2 tokens (`<s>` and `</s>`)
- **Passage Size**: Set to **100 tokens max** with **20 token overlap**
- **Rationale**: 100 tokens guarantees passage content never exceeds the 126 available token slots. Sentences are preserved across boundaries through 20-token sliding overlap, eliminating silent tail truncation.

---

## Classroom Limits
- **File size**: Max 10 MiB per upload
- **PDF page count**: Max 50 pages per PDF
- **Passage corpus**: Max 500 total indexed passages across all documents

---

## Checkpoint Verification Log & Observed Results

### Supported Formats & Extraction
- **PDF** (`.pdf`): 1-based page numbers preserved via `pypdf`.
- **Word** (`.docx`): Paragraphs and table reading order preserved via `python-docx`.
- **HTML** (`.html`): Scripts, styles, and remote links stripped via `BeautifulSoup`.
- **Plain Text** (`.txt`): Honest `para-N` paragraph segmentation.
- **Markdown** (`.md`): Structured section extraction preserving Markdown headers (`# Header`, `## Subheader`).
- **Excel** (`.xlsx`, `.xls`): Tabular worksheet extraction in row-by-row format via `openpyxl`.
- **PowerPoint** (`.pptx`, `.ppt`): Slide-by-slide text and table extraction preserving 1-based slide numbers via `python-pptx`.

### Chat Modes
- **📚 Grounded Document Chat** (`/chat?use_documents=1`): Retrieves top relevant passages from indexed documents, answers strictly with source citations (`[Passage #ID]`), and displays verified citation links to the parsed document view.
- **💬 Normal Direct Chat** (`/chat?use_documents=0`): Conversational direct chat with the local Ollama model (`ornith-1.5:9b`) without document retrieval. Works immediately even with zero uploaded documents. Mode switcher tabs allow 1-click toggling between modes.
- Uploaded supported formats:
  - `.txt` (`instructor_note.txt`, `test_document.txt`): parsed with honest `para-N` markers.
  - `.html` (`test_page.html`): parsed with `BeautifulSoup` (scripts/styles stripped, tags removed).
  - `.docx` (`test_ml.docx`): parsed with `python-docx` (paragraphs and tables in reading order).
  - `.pdf` (`test_db.pdf`): parsed with `pypdf` with 1-based page numbers (`page 1`).
- Empty file (`empty.txt`): identified with "empty or contains only whitespace" warning; marked as `empty`.
- Unsupported format (`test.jpg`): rejected with HTTP 302 and flash error message.
- Duplicate detection: re-uploading an identical file is caught by SHA-256 hash and rejected.
- Dedicated view `documents.parsed` (`/documents/<id>/parsed`) displays metadata, parser name, extraction warnings, scrollable extracted text, and indexing confirmation button.

### Checkpoint 2: Passages and Keyword Search
- Exact word search: Query `"superposition"` matches Passage #21 from `instructor_note.txt`.
- Punctuation handling: Queries like `"B-tree"` and punctuation-only `"!!!???"` execute safely without FTS syntax errors.
- Persistence: Restarting the Flask server preserved all documents, passages, and FTS5 entries.
- Cascading delete: Deleting document #1 cleanly removed its 5 passages from SQLite and FTS5, leaving other documents searchable.

### Checkpoint 3: Semantic Search
- Produced real 384-dimensional vectors stored in SQLite.
- FastEmbed model initialized lazily outside Flask import path.
- Exact vs Paraphrase Comparison:
  - Exact query `"superposition"`: found by keyword FTS5 and semantic search.
  - Paraphrased query `"How do subatomic computing units process parallel possibilities instead of binary values"`:
    - Keyword FTS5: **0 matches** (none of the exact words appear in the text).
    - Semantic search: **Retrieved Passage #20 (score: 0.4067) and Passage #21 (score: 0.3995)**.
- Plain-text copyable context box provides formatted references and passage text for downstream model use.

### Checkpoint 4: Answers & Small Chat with Local Ollama
- Reachable Ollama Server: `http://172.22.224.1:11434`
- Tested Model: `ornith-1.5:9b` (on-device local inference)
- Grounded Search Answer:
  - Question: *"What are qubits and how do they differ from classical bits?"*
  - Generated Answer: Correctly stated that qubits exist in superposition and differ from classical binary bits, citing `[Passage #21]`.
  - Citation chip verified against supplied passages and links directly to `/documents/6/parsed#passage-21`.
- Missing Fact Test:
  - Question: *"What is the capital of Australia according to the documents?"*
  - Generated Answer: Correctly reported that no evidence exists in the provided passages.
- Interactive Chat (`/chat`):
  - Multi-turn conversation tested with document grounding enabled (retaining context and citing `[Passage #22]`).
  - Toggling "Use my documents" off switches to ungrounded General Chat mode and starts a fresh conversation.
  - "New Chat" clears the server-side in-memory session.
- Unreachable Server Test:
  - When pointed to an invalid port (e.g. 9999), clean errors are displayed while upload, parsed view, and retrieval stay 100% operational.

---

## How to Inspect SQLite Rows using SQLAlchemy

From WSL using the external virtual environment:

```bash
cd ~/knowledge-base
~/venv_knowledge-base/bin/python -c "
from document_search import create_app, db
from document_search.models import Document, Passage

app = create_app()
with app.app_context():
    print('=== Documents ===')
    for doc in Document.query.all():
        print(f'Doc {doc.id}: {doc.original_name} [{doc.file_type}] (status: {doc.status}, passages: {len(doc.passages)}, parser: {doc.parser_used})')
    
    print('\n=== Sample Passages ===')
    for p in Passage.query.limit(5).all():
        print(f'Passage {p.id} (Doc {p.document_id}, {p.page_or_section}): {p.text[:60]}... [dim: {p.embedding_dim}]')
"
```
