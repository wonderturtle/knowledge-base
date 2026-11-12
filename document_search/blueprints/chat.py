"""Chat blueprint: grounded or general conversation with local Ollama."""

import uuid
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from ..ollama_client import (
    build_grounded_messages,
    bounded_history,
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

    # General chat is the default; changing mode clears prior context.
    # Check if mode switched
    current_doc_mode = request.values.get("use_documents", "0") == "1"
    previous_mode = session.get("chat_mode", current_doc_mode)
    if previous_mode != current_doc_mode:
        _CHAT_SESSIONS[chat_id] = []
        history = _CHAT_SESSIONS[chat_id]
    session["chat_mode"] = current_doc_mode
    # Bound the process store as well as each browser conversation.
    if len(_CHAT_SESSIONS) > 100:
        for old_id in list(_CHAT_SESSIONS):
            if old_id != chat_id:
                del _CHAT_SESSIONS[old_id]
                break

    # Action check: new_chat
    if request.method == "POST" and "new_chat" in request.form:
        _CHAT_SESSIONS[chat_id] = []
        return redirect(url_for("chat.index", use_documents="1" if current_doc_mode else "0"))

    error_msg = None

    if request.method == "POST" and "message" in request.form:
        user_message = request.form.get("message", "").strip()
        if len(user_message) > 4000:
            return "Message exceeds the 4,000 character limit.", 400
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
                method = request.form.get("retrieval_method", "semantic")
                if method == "keyword":
                    sources = keyword_search(user_message, limit=5)
                    sem_err = None
                else:
                    sources, sem_err = semantic_search(user_message, limit=5)
                    sources = sources or []
                if sem_err:
                    error_msg = "Semantic retrieval unavailable. Choose keyword retrieval explicitly. " + sem_err

                if sem_err:
                    assistant_text = error_msg
                elif not sources:
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
                    messages = [system_msg] + bounded_history(prior_turns) + [{"role": "user", "content": user_message}]

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

            # Bound retained source/context text as well as turn count.
            while len(history) > 2 and sum(len(str(m)) for m in history) > 80000:
                del history[:2]
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


@chat_bp.route("/settings", methods=["GET", "POST"])
def settings():
    """Configure the instructor's local Ollama server without editing files."""
    from urllib.parse import urlsplit
    from pathlib import Path
    from dotenv import set_key
    server = current_app.config["OLLAMA_BASE_URL"]
    model = current_app.config["OLLAMA_MODEL"]
    models, error = [], None
    if request.method == "POST":
        server = request.form.get("server", "").strip().rstrip("/")
        model = request.form.get("model", "").strip()
        try:
            parsed = urlsplit(server)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment
                    or parsed.path not in {"", "/"}):
                raise ValueError("Enter a server URL such as http://192.168.1.10:11434, without an API path or credentials.")
            parsed.port  # Validate the port syntax.
        except ValueError as exc:
            error = str(exc)
        if not error:
            models, error = get_installed_models(server, timeout=5)
            models = models or []
        if request.form.get("action") == "save" and not error:
            names = {item["name"] for item in models if isinstance(item.get("name"), str)}
            if model and model not in names:
                error = "Choose a model listed by this server, or leave the model blank."
            elif model and ("cloud" in model.lower() or any(item.get("remote_model") or item.get("remote_host") for item in models if item.get("name") == model)):
                error = "Choose an installed on-device model. Cloud models are outside this classroom exercise."
            else:
                env_path = Path(current_app.root_path).parent / ".env"
                try:
                    set_key(str(env_path), "OLLAMA_BASE_URL", server)
                    set_key(str(env_path), "OLLAMA_MODEL", model)
                except OSError as exc:
                    error = f"Could not save local settings: {exc}"
                else:
                    current_app.config.update(OLLAMA_BASE_URL=server, OLLAMA_MODEL=model)
                    _CHAT_SESSIONS.clear()
                    flash("Ollama settings saved. Conversations have been cleared for the new configuration.", "success")
                    return redirect(url_for("chat.settings"))
    return render_template("chat/settings.html", server=server, selected_model=model, models=models, error=error)
