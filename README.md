# Document Search & Classroom RAG Lab

A local Flask app for uploading and inspecting documents, comparing SQLite FTS5
keyword search with multilingual FastEmbed semantic search, and chatting with
an on-device Ollama model. Python identifiers and the interface use English.

## Install

Use Python 3.10 or newer supported by FastEmbed. The project is in
`~/knowledge-base`; keep its virtual environment outside the repository:

```bash
cd ~/knowledge-base
if [ ! -d ~/venv_knowledge-base ]; then
    python3 -m venv ~/venv_knowledge-base
fi
source ~/venv_knowledge-base/bin/activate
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements.txt
```

The interpreter should be `~/venv_knowledge-base/bin/python`.

No database download, SQL import or separate database server is needed.
**Starting the app creates `instance/documents.db`, its tables, and the FTS5
index automatically.** Later starts preserve existing data. Uploads, the model
cache, SQLite databases and journal files, and `.env` are ignored by Git.
Keep `schema.sql` as a readable schema reference; the app initializes its actual
schema through SQLAlchemy and does not require you to execute that file.

## Run the app

```bash
cd ~/knowledge-base
source ~/venv_knowledge-base/bin/activate
python -m flask --app app run --host 0.0.0.0 --port 5001 --no-debugger --no-reload
```

Open **http://127.0.0.1:5001** on the server, or
**http://YOUR_SERVER_IP:5001** from another computer on your network.
`0.0.0.0` makes Flask listen on all network interfaces; use the server's actual
IP address in your browser. Stop it with Ctrl+C. This is a classroom development
server, intended for a trusted local network.

To allow only this computer, replace `--host 0.0.0.0` with `--host 127.0.0.1`.

## Connect Ollama

1. Open **Ollama settings** in the navigation (`/chat/settings`).
2. Enter the server address reachable **from the Flask machine**, for example
   `http://192.168.1.10:11434`. Loopback works only if Ollama runs on the same
   machine. A Windows/WSL address may differ from a LAN address.
3. Click **Check server and load models**.
4. Choose an installed on-device model and click **Save settings**.
5. Open **Chat**, which defaults to **General chat** and needs no documents.

Settings are saved in the local, Git-ignored `.env` file and apply to both
Search and Chat immediately. Saving clears existing conversations. The app
never installs models or changes Ollama's network configuration. Disable
Ollama's cloud features on the server for this local-only exercise; models
identified as remote/cloud are rejected by the settings form.

Alternatively, create the environment file only if it does not already exist:

```bash
if [ ! -f .env ]; then
    cp .env.example .env
fi
```

Edit `OLLAMA_BASE_URL` and `OLLAMA_MODEL`, then restart Flask. The example leaves
the model blank deliberately: choose one actually installed on your server.
Connection failures, a missing model, and timeouts produce visible errors.
Parsing and keyword retrieval remain available without Ollama.

## Prepare semantic search before class

```bash
cd ~/knowledge-base
source ~/venv_knowledge-base/bin/activate
python -m flask --app app warm-model
```

The first download requires network access (roughly 220 MB). Inference then
runs locally. Optional environment settings:

```dotenv
# Defaults to instance/model_cache; reuse an existing cache if desired.
EMBEDDING_CACHE_DIR=/absolute/path/to/model_cache
# Require existing model files after the first successful warmup.
EMBEDDING_OFFLINE=1
```

Set these in `.env` or the shell before starting Flask. Model loading is lazy,
so starting Flask or opening General chat does not download an embedding model.
The current server on 12 November 2026 uses the already warmed cache at
`/home/codex/document-lab-model-cache` with offline loading enabled.

## Classroom workflow

1. Upload a document under **Documents**. Inspect its full, escaped **Parsed
   text** before choosing **Confirm & Index Document** (a POST action).
2. The approved extraction and its source boundaries are persisted. Indexing
   uses that stored extraction, rather than reparsing a possibly changed file.
3. Inspect every stored passage and its source locator in the parsed view.
4. Open **Search** to compare keyword and semantic results on one screen.
5. Choose a retrieval method and **Search and ask Ollama** to request an answer.
   If semantic search is unavailable, explicitly choose keyword retrieval.
6. Use **General chat** for ordinary conversation, or **Ask my documents** for
   retrieval with sources. Switching modes and **New chat** clear the history.

Generated answers are separate from their sources. Source chips resolve only
passage IDs actually supplied for that answer; unknown IDs are flagged.
Inspect the expandable **Context sent to the model** and check whether each
passage supports the answer. A citation does not guarantee accuracy.

## Supported files and limits

| Files | Local parser | Source references |
| --- | --- | --- |
| PDF | pypdf | 1-based page |
| DOCX | python-docx | Paragraph or table, in document order |
| TXT | UTF-8 text | Paragraph |
| MD / MARKDOWN | UTF-8 text, retaining Markdown | Heading/section |
| HTML / HTM | BeautifulSoup, excluding scripts/styles | Heading or element |
| XLSX | openpyxl, read-only | Worksheet, row, cell addresses; formulas and available cached results |
| XLS | xlrd | Worksheet, row, cell addresses; stored results, not formula text |
| PPTX | python-pptx, including grouped text shapes and tables | 1-based slide |
| PPT | Installed headless LibreOffice converts a temporary copy, then python-pptx | 1-based slide |

Legacy PPT requires local LibreOffice. If it is unavailable, save the file as
PPTX. The app does not install system software or use an online converter.
Image-only PDFs need OCR, which is outside this lab. Charts, images, artwork,
complex layout and table relationships may not survive text extraction.
Formulas are never calculated, and embedded macros/scripts are not executed.

Limits: 10 MiB per file; 50 PDF pages or presentation slides; 50 worksheets and
50,000 processed cells per workbook; 500 indexed passages across the corpus.
Over-limit or unreadable uploads produce errors. Identical bytes are rejected
by SHA-256; generated storage names avoid filename collisions.

## Storage and retrieval

`Document` has an integer primary key and stores original/stored filenames,
hash, format, persistent extraction and boundaries, warnings and status.
`Passage` has an integer primary key, a foreign key to `Document`, sequence,
source locator, complete passage text, optional JSON vector, model ID and
vector dimension. A document has many passages. Deletion cascades to passages;
SQLite triggers keep FTS5 synchronized in the same database transaction.

The embedding model is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: 384 dimensions,
128 input tokens including special tokens. An untruncated copy of its tokenizer
splits source text into windows of 100 content tokens with 20-token overlap.
Original character slices preserve text, every non-whitespace character is
checked for coverage, and each slice is checked against the 128-token limit.
Overlap helps continuity; it does not guarantee complete sentences. Semantic
queries over 128 tokens are rejected with guidance to shorten them.

When embeddings are unavailable, a separate character-window fallback permits
keyword-only indexing. These passages have no vector and are marked accordingly;
this is not semantic search. Documents already indexed without vectors must be
deleted and uploaded/indexed again after model preparation to obtain vectors.

SQLite stores vectors as JSON; Python scans compatible vectors and calculates
exact cosine similarity. There is no approximate vector index. Cosine scores
are similarity values, not probabilities, and are not comparable to FTS ranks.

Chat history stays in server memory, separate for each browser session; cookies
contain an identifier and mode, not document text. History is limited by turn
count and size, with at most three recent exchanges and 4,000 history characters
sent to Ollama. Messages are limited to 4,000 characters; generation requests
512 output tokens. Histories clear on restart. Use one Flask process for this lab.

## Verify

```bash
cd ~/knowledge-base
source ~/venv_knowledge-base/bin/activate
python -m pip check
python -m flask --app app routes
# Use a prepared cache to include real-vector checks:
EMBEDDING_CACHE_DIR=/absolute/path/to/model_cache python -m unittest discover -s tests -v
```

Without `EMBEDDING_CACHE_DIR`, the real-vector test is explicitly skipped.
Tests use temporary databases/uploads. Ollama transport in these automated tests
is mocked to verify request construction and history isolation; it is not proof
of a real model round trip.

Verified here on 12 November 2026: four workflow tests passed, including actual
384-dimensional vectors, complete long-document tail coverage, indexing the
approved extraction after the original file changes, database reconnection,
FTS deletion, duplicates, malformed files, upload limits, empty extraction,
settings validation, General chat isolation, mode resets, and explicit retrieval
selection. A real Italian query retrieved the password passage at cosine
`0.7141523166570964`. Separate real XLS and PPT conversion/parsing checks passed
with installed LibreOffice. The settings page was rendered and visually inspected
in Chromium. No real Ollama answer/chat round trip has yet been verified here;
configure a reachable server and model to finish that check.

Versions used: Python 3.13.5, Flask 3.1.3, Flask-SQLAlchemy 3.1.1,
SQLAlchemy 2.1.8, FastEmbed 0.9.1, pypdf 7.0.1, python-docx 1.2.0,
BeautifulSoup 4.15.0, openpyxl 3.1.5, xlrd 2.0.2, python-pptx 1.0.2,
requests 2.34.2, python-dotenv 1.2.4.

Inspect stored rows without a separate SQLite client:

```bash
~/venv_knowledge-base/bin/python - <<'PY'
from document_search import create_app, db
from document_search.models import Document, Passage
app = create_app()
with app.app_context():
    for doc in db.session.scalars(db.select(Document)):
        print(doc.id, doc.original_name, doc.status, doc.parser_used)
    for passage in db.session.scalars(db.select(Passage).limit(5)):
        print(passage.id, passage.document_id, passage.page_or_section,
              passage.embedding_dim, passage.text)
PY
```
