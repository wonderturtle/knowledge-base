"""Chat blueprint: grounded or general conversation with local Ollama."""

import uuid
from flask import (
    Blueprint,
    current_app,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from ..ollama_client import (
    build_grounded_messages,
    call_chat_completion,
    get_installed_models,
    resolve_citations,
)
from ..services import (
    count_indexed_passages,
    get_embedding_error,
    keyword_search,
    semantic_search,
)

chat_bp = Blueprint("chat", __name__, url_prefix="/chat")

# Server-side in-memory chat session store: session_id -> list of message dicts
# Format: {"role": "user"|"assistant", "content": ..., "use_documents": bool, "sources": [...], "context_sent": ...}
_CHAT_SESSIONS = {}


@chat_bp.route("", methods=["GET", "POST"])
@chat_bp.route("/", methods=["GET", "POST"])
def index():
    """Chat page with grounded document retrieval or general chat."""
    if "chat_id" not in session:
        session["chat_id"] = uuid.uuid4().hex

    chat_id = session["chat_id"]
    if chat_id not in _CHAT_SESSIONS:
        _CHAT_SESSIONS[chat_id] = []

    history = _CHAT_SESSIONS[chat_id]

    ollama_base_url = current_app.config["OLLAMA_BASE_URL"]
    ollama_model = current_app.config["OLLAMA_MODEL"]
    ollama_timeout = current_app.config["OLLAMA_TIMEOUT"]

    # Check available models
    installed_models, server_status_error = get_installed_models(ollama_base_url, timeout=3)

    # Current document toggle (default ON)
    # Check if mode switched
    current_doc_mode = request.values.get("use_documents", "1") == "1"

    # Action check: new_chat
    if request.method == "POST" and "new_chat" in request.form:
        _CHAT_SESSIONS[chat_id] = []
        return redirect(url_for("chat.index", use_documents="1" if current_doc_mode else "0"))

    error_msg = None

    if request.method == "POST" and "message" in request.form:
        user_message = request.form.get("message", "").strip()
        doc_mode_submitted = request.form.get("use_documents") == "1"

        # If document mode switched, start fresh conversation
        last_mode = history[-1].get("use_documents") if history else doc_mode_submitted
        if history and last_mode != doc_mode_submitted:
            _CHAT_SESSIONS[chat_id] = []
            history = _CHAT_SESSIONS[chat_id]

        if user_message:
            # Append user message
            user_entry = {
                "role": "user",
                "content": user_message,
                "use_documents": doc_mode_submitted,
            }
            history.append(user_entry)

            # Build messages for Ollama
            sources = []
            context_sent = ""
            valid_citations = {}
            unknown_citations = []

            if doc_mode_submitted:
                # Retrieve relevant passages (try semantic, fallback to keyword)
                sem_res, sem_err = semantic_search(user_message, limit=5)
                if not sem_err and sem_res:
                    sources = sem_res
                else:
                    sources = keyword_search(user_message, limit=5)

                if not sources:
                    assistant_text = "No relevant document passages were found to answer your question."
                elif not ollama_model:
                    assistant_text = "Error: OLLAMA_MODEL is not configured in .env."
                    error_msg = assistant_text
                else:
                    # Filter history to same mode
                    prior_turns = [
                        {"role": m["role"], "content": m["content"]}
                        for m in history[:-1]
                        if m.get("use_documents") == doc_mode_submitted
                    ]
                    messages = build_grounded_messages(user_message, sources, history=prior_turns)
                    context_sent = messages[-1]["content"]

                    ans, err = call_chat_completion(
                        base_url=ollama_base_url,
                        model=ollama_model,
                        messages=messages,
                        timeout=ollama_timeout,
                    )
                    if err:
                        assistant_text = f"Error from Ollama: {err}"
                        error_msg = err
                    else:
                        assistant_text = ans
                        valid_citations, unknown_citations = resolve_citations(ans, sources)
            else:
                # General chat without documents
                if not ollama_model:
                    assistant_text = "Error: OLLAMA_MODEL is not configured in .env."
                    error_msg = assistant_text
                else:
                    prior_turns = [
                        {"role": m["role"], "content": m["content"]}
                        for m in history[:-1]
                        if not m.get("use_documents")
                    ]
                    system_msg = {
                        "role": "system",
                        "content": "You are a helpful general-purpose AI assistant in a classroom setting.",
                    }
                    messages = [system_msg] + prior_turns + [{"role": "user", "content": user_message}]

                    ans, err = call_chat_completion(
                        base_url=ollama_base_url,
                        model=ollama_model,
                        messages=messages,
                        timeout=ollama_timeout,
                    )
                    if err:
                        assistant_text = f"Error from Ollama: {err}"
                        error_msg = err
                    else:
                        assistant_text = ans

            # Append assistant message
            history.append({
                "role": "assistant",
                "content": assistant_text,
                "use_documents": doc_mode_submitted,
                "sources": sources,
                "context_sent": context_sent,
                "valid_citations": valid_citations,
                "unknown_citations": unknown_citations,
            })

            # Keep bounded history (last 20 messages)
            if len(history) > 20:
                _CHAT_SESSIONS[chat_id] = history[-20:]

            return redirect(url_for("chat.index", use_documents="1" if doc_mode_submitted else "0"))

    total_passages = count_indexed_passages()
    embedding_error = get_embedding_error()

    return render_template(
        "chat/index.html",
        history=history,
        use_documents=current_doc_mode,
        total_passages=total_passages,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        installed_models=installed_models,
        server_status_error=server_status_error,
        embedding_error=embedding_error,
        error_msg=error_msg,
    )
