"""Business logic: extraction, passage splitting, search, embedding and SQLAlchemy operations."""

import hashlib
import json
import os
import re
import uuid
import numpy as np
from sqlalchemy import text

from . import db
from .config import Config
from .models import Document, Passage

# ---------------------------------------------------------------------------
# Lazy-loaded embedding model (singleton, avoids reloader double-download)
# ---------------------------------------------------------------------------
_embedding_model = None
_embedding_error = None


def _get_embedding_model():
    """Return the FastEmbed TextEmbedding model, or None if unavailable."""
    global _embedding_model, _embedding_error
    if _embedding_model is not None:
        return _embedding_model
    if _embedding_error is not None:
        return None
    try:
        from fastembed import TextEmbedding

        _embedding_model = TextEmbedding(
            model_name=Config.EMBEDDING_MODEL,
            cache_dir=Config.MODEL_CACHE_DIR,
        )
        return _embedding_model
    except Exception as exc:
        _embedding_error = str(exc)
        return None


def get_embedding_error():
    """Return the embedding model error message, or None."""
    return _embedding_error


def get_tokenizer():
    """Return the tokenizer from the embedding model, or None."""
    model = _get_embedding_model()
    if model is None:
        return None
    return model.model.tokenizer


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def file_content_hash(filepath):
    """Return SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_storage_name(original_name):
    """Generate a unique storage filename preserving the extension."""
    ext = os.path.splitext(original_name)[1].lower()
    return uuid.uuid4().hex + ext


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath):
    """Extract text from a PDF file preserving 1-based page numbers."""
    from pypdf import PdfReader

    warnings = []
    reader = PdfReader(filepath)
    num_pages = len(reader.pages)
    if num_pages > Config.MAX_PDF_PAGES:
        return None, [
            f"This PDF has {num_pages} pages, exceeding the classroom limit "
            f"of {Config.MAX_PDF_PAGES} pages. Please upload a shorter file."
        ]

    pages = {}
    empty_pages = []
    for i, page in enumerate(reader.pages, start=1):
        text_content = (page.extract_text() or "").strip()
        pages[i] = text_content
        if not text_content:
            empty_pages.append(i)

    if empty_pages and len(empty_pages) == num_pages:
        warnings.append(
            "No text could be extracted from any page. This may be a "
            "scanned or image-only PDF that requires OCR (which is outside "
            "the scope of this exercise)."
        )
    elif empty_pages:
        pages_str = ", ".join(str(p) for p in empty_pages[:10])
        if len(empty_pages) > 10:
            pages_str += f" … and {len(empty_pages) - 10} more"
        warnings.append(
            f"Pages with no extractable text: {pages_str}. "
            "These may be scanned images requiring OCR."
        )

    return pages, warnings


def extract_text_from_docx(filepath):
    """Extract text from a DOCX file with honest paragraph and table identifiers."""
    from docx import Document as DocxDocument

    warnings = []
    doc = DocxDocument(filepath)

    sections = {}
    idx = 1

    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            text_content = element.text.strip() if element.text else ""
            if not text_content:
                runs_text = []
                for node in element.iter():
                    ntag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
                    if ntag == "t" and node.text:
                        runs_text.append(node.text)
                text_content = "".join(runs_text).strip()
            if text_content:
                sections[f"para-{idx}"] = text_content
                idx += 1

        elif tag == "tbl":
            rows_text = []
            for row_el in element.iter():
                rtag = row_el.tag.split("}")[-1] if "}" in row_el.tag else row_el.tag
                if rtag == "tr":
                    cells = []
                    for cell_el in row_el.iter():
                        ctag = cell_el.tag.split("}")[-1] if "}" in cell_el.tag else cell_el.tag
                        if ctag == "tc":
                            cell_texts = []
                            for t_el in cell_el.iter():
                                ttag = t_el.tag.split("}")[-1] if "}" in t_el.tag else t_el.tag
                                if ttag == "t" and t_el.text:
                                    cell_texts.append(t_el.text)
                            cells.append(" ".join(cell_texts).strip())
                    if cells:
                        rows_text.append(" | ".join(cells))
            if rows_text:
                table_text = "\n".join(rows_text)
                sections[f"table-{idx}"] = table_text
                idx += 1

    if not sections:
        warnings.append("No text could be extracted from this DOCX file.")

    has_tables = any(k.startswith("table-") for k in sections)
    if has_tables:
        warnings.append(
            "Note: Table text is included in simple row-by-row reading order. "
            "Complex table layouts and cell relationships may be lost during "
            "extraction."
        )

    return sections, warnings


def extract_text_from_html(filepath):
    """Extract text from an uploaded HTML file. Scripts and remote resources are stripped."""
    from bs4 import BeautifulSoup

    warnings = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup.find_all(["script", "style", "link", "iframe", "object", "embed"]):
        tag.decompose()

    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr.lower().startswith("on"):
                del tag[attr]

    sections = {}
    idx = 1
    for elem in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6",
                                "li", "td", "th", "blockquote", "pre", "div"]):
        text_content = elem.get_text(separator=" ", strip=True)
        if text_content:
            tag_name = elem.name
            sections[f"{tag_name}-{idx}"] = text_content
            idx += 1

    if not sections:
        all_text = soup.get_text(separator="\n", strip=True)
        if all_text:
            sections["section-1"] = all_text
        else:
            warnings.append("No text could be extracted from this HTML file.")

    return sections, warnings


def extract_text_from_txt(filepath):
    """Extract text from a plain UTF-8 text file."""
    warnings = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if not content.strip():
        warnings.append(
            "This text file is empty or contains only whitespace. "
            "There is no text to index."
        )
        return {}, warnings

    paragraphs = re.split(r"\n\s*\n", content)
    sections = {}
    idx = 1
    for para in paragraphs:
        text_content = para.strip()
        if text_content:
            sections[f"para-{idx}"] = text_content
            idx += 1

    return sections, warnings


def extract_text_from_md(filepath):
    """Extract text from a Markdown (.md) file preserving section headers."""
    warnings = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if not content.strip():
        warnings.append("This Markdown file is empty or contains only whitespace.")
        return {}, warnings

    # Split by Markdown headers (# Header) or double newlines
    lines = content.splitlines()
    sections = {}
    current_title = "Introduction"
    current_lines = []
    sec_idx = 1

    for line in lines:
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            if current_lines:
                text_block = "\n".join(current_lines).strip()
                if text_block:
                    sections[f"sec-{sec_idx} ({current_title})"] = text_block
                    sec_idx += 1
                current_lines = []
            current_title = header_match.group(2).strip()
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        text_block = "\n".join(current_lines).strip()
        if text_block:
            sections[f"sec-{sec_idx} ({current_title})"] = text_block

    if not sections:
        # Fallback to paragraph splitting if no headers
        paragraphs = re.split(r"\n\s*\n", content)
        for idx, p in enumerate(paragraphs, start=1):
            if p.strip():
                sections[f"para-{idx}"] = p.strip()

    return sections, warnings


def extract_text_from_excel(filepath):
    """Extract tabular text from an Excel (.xlsx / .xls) file."""
    import openpyxl

    warnings = [
        "Note: Excel spreadsheet data is extracted in row-by-row table format. "
        "Formulas, charts, and cell styling are flattened to plain text."
    ]
    sections = {}

    try:
        wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
        sheet_count = len(wb.sheetnames)
        for s_idx, sheet_name in enumerate(wb.sheetnames, start=1):
            sheet = wb[sheet_name]
            sheet_rows = []
            for row in sheet.iter_rows(values_only=True):
                non_empty = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                if non_empty:
                    sheet_rows.append(" | ".join(non_empty))
            if sheet_rows:
                sections[f"sheet-{s_idx} ({sheet_name})"] = "\n".join(sheet_rows)

        wb.close()
        if not sections:
            warnings.append("No readable data rows found in this spreadsheet.")
        return sections, warnings, sheet_count
    except Exception as exc:
        return {}, [f"Failed to read Excel file: {exc}"], None


def extract_text_from_pptx(filepath):
    """Extract slide text from a PowerPoint (.pptx / .ppt) presentation."""
    from pptx import Presentation

    warnings = [
        "Note: Presentation slides are extracted in reading order. "
        "Visual layouts, animations, and non-text graphics are omitted."
    ]
    sections = {}

    try:
        prs = Presentation(filepath)
        slide_count = len(prs.slides)
        if slide_count > Config.MAX_PDF_PAGES:
            return None, [
                f"This presentation has {slide_count} slides, exceeding the classroom limit "
                f"of {Config.MAX_PDF_PAGES} slides/pages."
            ], None

        for s_idx, slide in enumerate(prs.slides, start=1):
            slide_texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text_val = para.text.strip()
                        if text_val:
                            slide_texts.append(text_val)
                elif shape.has_table:
                    for row in shape.table.rows:
                        row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_cells:
                            slide_texts.append(" | ".join(row_cells))

            if slide_texts:
                sections[s_idx] = "\n".join(slide_texts)

        if not sections:
            warnings.append("No text could be extracted from these presentation slides.")

        return sections, warnings, slide_count
    except Exception as exc:
        return {}, [f"Failed to read PowerPoint presentation: {exc}"], None


def extract_text(filepath, file_type):
    """Dispatch extraction by file type.

    Returns (sections_or_pages, warnings, page_count_or_none, parser_name).
    """
    ft = file_type.lower()
    if ft == "pdf":
        pages, warnings = extract_text_from_pdf(filepath)
        if pages is None:
            return None, warnings, None, "pypdf"
        return pages, warnings, len(pages), "pypdf"
    elif ft == "docx":
        sections, warnings = extract_text_from_docx(filepath)
        return sections, warnings, None, "python-docx"
    elif ft == "html":
        sections, warnings = extract_text_from_html(filepath)
        return sections, warnings, None, "BeautifulSoup (html.parser)"
    elif ft == "txt":
        sections, warnings = extract_text_from_txt(filepath)
        return sections, warnings, None, "utf-8 plain text"
    elif ft == "md":
        sections, warnings = extract_text_from_md(filepath)
        return sections, warnings, None, "markdown parser"
    elif ft in ("xlsx", "xls"):
        sections, warnings, sheet_count = extract_text_from_excel(filepath)
        return sections, warnings, sheet_count, "openpyxl (excel)"
    elif ft in ("pptx", "ppt"):
        sections, warnings, slide_count = extract_text_from_pptx(filepath)
        if sections is None:
            return None, warnings, None, "python-pptx"
        return sections, warnings, slide_count, "python-pptx (slides)"
    else:
        return None, [f"Unsupported file type: {ft}"], None, "unknown"


# ---------------------------------------------------------------------------
# Passage splitting
# ---------------------------------------------------------------------------

def _tokenize_text(text_input, tokenizer):
    """Tokenize text and return list of (token_id, token_str) pairs."""
    enc = tokenizer.encode(text_input)
    ids = enc.ids
    tokens = enc.tokens
    if len(ids) >= 2 and ids[0] == 0 and ids[-1] == 2:
        ids = ids[1:-1]
        tokens = tokens[1:-1]
    return ids, tokens


def split_into_passages(sections_or_pages, file_type, tokenizer=None):
    """Split extracted text into overlapping passages."""
    max_tokens = Config.PASSAGE_MAX_TOKENS
    overlap_tokens = Config.PASSAGE_OVERLAP_TOKENS
    passages = []
    seq = 0

    for key in sorted(sections_or_pages.keys(), key=_sort_key):
        text_content = sections_or_pages[key]
        if not text_content.strip():
            continue

        if file_type == "pdf":
            label = f"page {key}"
        elif file_type in ("pptx", "ppt") and isinstance(key, int):
            label = f"slide {key}"
        else:
            label = str(key)

        if tokenizer is not None:
            ids, tokens = _tokenize_text(text_content, tokenizer)
            start = 0
            while start < len(ids):
                end = min(start + max_tokens, len(ids))
                chunk_text = tokenizer.decode(ids[start:end]).strip()
                if chunk_text:
                    seq += 1
                    passages.append({
                        "text": chunk_text,
                        "page_or_section": label,
                        "seq": seq,
                    })
                if end >= len(ids):
                    break
                start = end - overlap_tokens
        else:
            words = text_content.split()
            max_words = 75
            overlap_words = 15
            start = 0
            while start < len(words):
                end = min(start + max_words, len(words))
                chunk_text = " ".join(words[start:end]).strip()
                if chunk_text:
                    seq += 1
                    passages.append({
                        "text": chunk_text,
                        "page_or_section": label,
                        "seq": seq,
                    })
                if end >= len(words):
                    break
                start = end - overlap_words

    return passages


def _sort_key(key):
    """Sort key: integers first (for PDF pages), then strings."""
    if isinstance(key, int):
        return (0, key, "")
    return (1, 0, key)


# ---------------------------------------------------------------------------
# SQLAlchemy Database operations
# ---------------------------------------------------------------------------

def check_duplicate(content_hash):
    """Check if a file with the same content hash already exists."""
    return Document.query.filter_by(content_hash=content_hash).first()


def insert_document(original_name, storage_name, content_hash,
                    file_type, extracted_text, status, page_count,
                    parser_used="", extraction_warnings=None):
    """Insert a document record and return its ID."""
    warnings_json = json.dumps(extraction_warnings or [])
    doc = Document(
        original_name=original_name,
        storage_name=storage_name,
        content_hash=content_hash,
        file_type=file_type,
        parser_used=parser_used,
        extracted_text=extracted_text,
        extraction_warnings=warnings_json,
        status=status,
        page_count=page_count,
    )
    try:
        db.session.add(doc)
        db.session.commit()
        return doc.id
    except Exception:
        db.session.rollback()
        raise


def update_document_status(doc_id, status):
    """Update document status."""
    try:
        doc = db.session.get(Document, doc_id)
        if doc:
            doc.status = status
            db.session.commit()
    except Exception:
        db.session.rollback()
        raise


def get_document(doc_id):
    """Get a single document by ID."""
    return db.session.get(Document, doc_id)


def get_all_documents():
    """Get all documents ordered by creation date."""
    return Document.query.order_by(Document.created_at.desc()).all()


def count_indexed_passages():
    """Return total number of indexed passages."""
    return Passage.query.count()


def insert_passages(doc_id, passages, model_id=None):
    """Insert passages into the database with optional embeddings."""
    model = _get_embedding_model()

    current_count = count_indexed_passages()
    if current_count + len(passages) > Config.MAX_INDEXED_PASSAGES:
        remaining = Config.MAX_INDEXED_PASSAGES - current_count
        return False, (
            f"Adding {len(passages)} passages would exceed the classroom limit "
            f"of {Config.MAX_INDEXED_PASSAGES} total indexed passages "
            f"({current_count} already indexed, {remaining} remaining). "
            f"Please delete some documents first or use a shorter file."
        )

    texts = [p["text"] for p in passages]
    embeddings = None
    actual_model_id = None
    dim = None
    if model is not None:
        try:
            emb_list = list(model.embed(texts))
            embeddings = emb_list
            actual_model_id = Config.EMBEDDING_MODEL
            dim = Config.EMBEDDING_DIM
        except Exception as exc:
            print(f"Warning: embedding failed: {exc}")

    try:
        for i, p in enumerate(passages):
            emb_json = None
            if embeddings is not None:
                emb_json = json.dumps(embeddings[i].tolist())

            passage = Passage(
                document_id=doc_id,
                seq=p["seq"],
                text=p["text"],
                page_or_section=p["page_or_section"],
                embedding=emb_json,
                model_id=actual_model_id,
                embedding_dim=dim,
            )
            db.session.add(passage)

        db.session.commit()
        return True, None
    except Exception:
        db.session.rollback()
        raise


def delete_document(doc_id):
    """Delete a document and its passages (SQLAlchemy cascade and SQLite foreign keys)."""
    doc = db.session.get(Document, doc_id)
    if doc is None:
        return False

    storage_name = doc.storage_name
    try:
        db.session.delete(doc)
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise

    filepath = os.path.join(Config.UPLOAD_FOLDER, storage_name)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass

    return True


def get_document_passages(doc_id):
    """Get all passages for a document ordered by seq."""
    return Passage.query.filter_by(document_id=doc_id).order_by(Passage.seq).all()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def keyword_search(query, limit=5):
    """Search passages using FTS5 keyword matching via SQLAlchemy session."""
    if not query or not query.strip():
        return []

    words = re.findall(r"\w+", query, re.UNICODE)
    if not words:
        return []

    fts_query = " ".join(f'"{w}"' for w in words)

    rows = db.session.execute(
        text("""
            SELECT p.id AS passage_id, p.document_id, p.seq, p.text,
                   p.page_or_section,
                   d.original_name,
                   rank
            FROM passages_fts
            JOIN passages p ON p.id = passages_fts.rowid
            JOIN documents d ON d.id = p.document_id
            WHERE passages_fts MATCH :query
            ORDER BY rank
            LIMIT :limit
        """),
        {"query": fts_query, "limit": limit},
    ).mappings().all()

    return [dict(r) for r in rows]


def semantic_search(query, limit=5):
    """Search passages using cosine similarity with FastEmbed via SQLAlchemy."""
    model = _get_embedding_model()
    if model is None:
        return None, get_embedding_error() or "Embedding model not available."

    if not query or not query.strip():
        return [], None

    try:
        query_emb = list(model.embed([query]))[0]
    except Exception as exc:
        return None, f"Failed to embed query: {exc}"

    rows = db.session.execute(
        text("""
            SELECT p.id AS passage_id, p.document_id, p.seq, p.text,
                   p.page_or_section, p.embedding,
                   d.original_name
            FROM passages p
            JOIN documents d ON d.id = p.document_id
            WHERE p.embedding IS NOT NULL
        """)
    ).mappings().all()

    if not rows:
        return [], None

    results = []
    query_norm = np.linalg.norm(query_emb)
    if query_norm == 0:
        return [], None

    for row in rows:
        emb = np.array(json.loads(row["embedding"]))
        emb_norm = np.linalg.norm(emb)
        if emb_norm == 0:
            continue
        similarity = float(np.dot(query_emb, emb) / (query_norm * emb_norm))
        result = dict(row)
        del result["embedding"]
        result["similarity"] = similarity
        results.append(result)

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:limit], None
