"""Documents blueprint: list, upload, inspection preview, indexing, deletion."""

import os
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from ..services import (
    check_duplicate,
    count_indexed_passages,
    delete_document,
    extract_text,
    file_content_hash,
    generate_storage_name,
    get_all_documents,
    get_document,
    get_document_passages,
    get_embedding_error,
    get_tokenizer,
    insert_document,
    insert_passages,
    split_into_passages,
    update_document_status,
    stored_sections,
)

documents_bp = Blueprint("documents", __name__, url_prefix="/documents")


@documents_bp.route("/", methods=["GET"])
@documents_bp.route("", methods=["GET"])
def index():
    """List all uploaded documents."""
    docs = get_all_documents()
    total_passages = count_indexed_passages()
    return render_template(
        "documents/index.html",
        documents=docs,
        total_passages=total_passages,
        max_passages=current_app.config["MAX_INDEXED_PASSAGES"],
    )


@documents_bp.route("/upload", methods=["POST"])
def upload():
    """Upload document, extract text, and redirect to parsed view."""
    if "file" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("documents.index"))

    file = request.files["file"]
    if not file or not file.filename:
        flash("No file selected.", "error")
        return redirect(url_for("documents.index"))

    original_name = file.filename
    ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""
    allowed = current_app.config["ALLOWED_EXTENSIONS"]
    if ext not in allowed:
        flash(
            f"Unsupported file type: .{ext}. Allowed: {', '.join('.' + e for e in sorted(allowed))}.",
            "error",
        )
        return redirect(url_for("documents.index"))

    storage_name = generate_storage_name(original_name)
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], storage_name)
    file.save(filepath)

    # Validate file size
    max_bytes = current_app.config["MAX_UPLOAD_SIZE_BYTES"]
    if os.path.getsize(filepath) > max_bytes:
        try:
            os.remove(filepath)
        except OSError:
            pass
        flash(
            f"File too large. Maximum upload size is {max_bytes // (1024 * 1024)} MiB.",
            "error",
        )
        return redirect(url_for("documents.index"))

    # Duplicate content check
    content_hash = file_content_hash(filepath)
    existing = check_duplicate(content_hash)
    if existing:
        try:
            os.remove(filepath)
        except OSError:
            pass
        flash(
            f'This file is identical to an already uploaded document: "{existing.original_name}". Duplicate rejected.',
            "warning",
        )
        return redirect(url_for("documents.index"))

    # Extract text
    sections, warnings, page_count, parser_name = extract_text(filepath, ext)
    if sections is None:
        try:
            os.remove(filepath)
        except OSError:
            pass
        for w in warnings:
            flash(w, "error")
        return redirect(url_for("documents.index"))

    # Build full structured extracted text with page/section headers
    full_text = ""
    for key in sections:
        label = f"page {key}" if isinstance(key, int) else str(key)
        full_text += f"=== [{label}] ===\n{sections[key]}\n\n"

    status = "pending" if any(value.strip() for value in sections.values()) else "empty"
    doc_id = insert_document(
        original_name=original_name,
        storage_name=storage_name,
        content_hash=content_hash,
        file_type=ext,
        extracted_text=full_text.strip(),
        status=status,
        page_count=page_count,
        parser_used=parser_name,
        extraction_warnings=warnings,
        sections=sections,
    )

    for w in warnings:
        flash(w, "warning")

    return redirect(url_for("documents.parsed", document_id=doc_id))


@documents_bp.route("/<int:document_id>/parsed", methods=["GET"])
def parsed(document_id):
    """Dedicated inspection view for parsed document and its stored passages."""
    doc = get_document(document_id)
    if doc is None:
        flash("Document not found.", "error")
        return redirect(url_for("documents.index"))

    passages = get_document_passages(document_id)
    embedding_error = get_embedding_error()

    return render_template(
        "documents/parsed.html",
        doc=doc,
        passages=passages,
        embedding_error=embedding_error,
    )


@documents_bp.route("/<int:document_id>/index", methods=["POST"])
def index_doc(document_id):
    """Confirm indexing: split approved text into passages and store with embeddings."""
    doc = get_document(document_id)
    if doc is None:
        flash("Document not found.", "error")
        return redirect(url_for("documents.index"))

    if doc.status == "indexed":
        flash("Document is already indexed.", "info")
        return redirect(url_for("documents.parsed", document_id=document_id))

    sections = stored_sections(doc)

    if not sections:
        flash("No text could be extracted to index.", "error")
        update_document_status(document_id, "empty")
        return redirect(url_for("documents.parsed", document_id=document_id))

    tokenizer = get_tokenizer()
    try:
        passages = split_into_passages(sections, doc.file_type, tokenizer)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("documents.parsed", document_id=document_id))

    if not passages:
        flash("No passages could be generated from the extracted text.", "error")
        update_document_status(document_id, "empty")
        return redirect(url_for("documents.parsed", document_id=document_id))

    ok, error_msg = insert_passages(document_id, passages)
    if not ok:
        flash(error_msg, "error")
        return redirect(url_for("documents.parsed", document_id=document_id))

    if get_embedding_error():
        flash("Keyword indexing completed; semantic search unavailable: " + get_embedding_error(), "warning")
    flash(
        f"Document successfully indexed into {len(passages)} passages.",
        "success",
    )
    return redirect(url_for("documents.parsed", document_id=document_id))


@documents_bp.route("/<int:document_id>/delete", methods=["POST"])
def delete_doc(document_id):
    """Delete a document and cascade remove passages and FTS entries."""
    ok = delete_document(document_id)
    if ok:
        flash("Document and all associated passages deleted.", "success")
    else:
        flash("Document not found.", "error")
    return redirect(url_for("documents.index"))
