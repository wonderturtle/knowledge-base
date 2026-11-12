"""Search blueprint: compare keyword vs semantic retrieval, and ask local Ollama."""

from flask import (
    Blueprint,
    current_app,
    render_template,
    request,
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

search_bp = Blueprint("search", __name__, url_prefix="/search")


@search_bp.route("", methods=["GET", "POST"])
@search_bp.route("/", methods=["GET", "POST"])
def index():
    """Search screen with side-by-side keyword/semantic comparison and Ollama answer."""
    query = request.values.get("q", "").strip()
    retrieval_method = request.values.get("retrieval_method", "semantic")
    is_ask_action = (request.method == "POST")

    keyword_results = []
    semantic_results = []
    semantic_error = None
    ollama_answer = None
    ollama_error = None
    context_sent = ""
    valid_citations = {}
    unknown_citations = []
    retrieved_for_model = []

    ollama_base_url = current_app.config["OLLAMA_BASE_URL"]
    ollama_model = current_app.config["OLLAMA_MODEL"]
    ollama_timeout = current_app.config["OLLAMA_TIMEOUT"]

    # Check available Ollama models if configured
    installed_models, server_status_error = get_installed_models(ollama_base_url, timeout=3)

    if query:
        # Run both retrieval methods for comparison
        keyword_results = keyword_search(query, limit=5)
        sem_res, sem_err = semantic_search(query, limit=5)
        if sem_err:
            semantic_error = sem_err
        elif sem_res is not None:
            semantic_results = sem_res

        # If user asked Ollama via POST
        if is_ask_action:
            # Select passages based on user's chosen retrieval method
            if retrieval_method == "semantic" and semantic_error:
                ollama_error = "Semantic retrieval unavailable. Choose keyword retrieval explicitly to ask Ollama."
            if retrieval_method == "semantic":
                retrieved_for_model = semantic_results
            else:
                retrieved_for_model = keyword_results
                retrieval_method = "keyword"

            if ollama_error:
                pass
            elif not retrieved_for_model:
                ollama_answer = "No evidence found in uploaded documents for this query. No passages were retrieved."
            elif not ollama_model:
                ollama_error = "No Ollama model configured. Please set OLLAMA_MODEL in .env."
            else:
                messages = build_grounded_messages(query, retrieved_for_model)
                # Save the prompt context for display
                context_sent = messages[-1]["content"]

                answer_text, err = call_chat_completion(
                    base_url=ollama_base_url,
                    model=ollama_model,
                    messages=messages,
                    timeout=ollama_timeout,
                )
                if err:
                    ollama_error = err
                else:
                    ollama_answer = answer_text
                    valid_citations, unknown_citations = resolve_citations(
                        answer_text, retrieved_for_model
                    )

    # Build copyable plain-text context box
    context_box_lines = [f"Question: {query}", ""]
    if semantic_results:
        context_box_lines.append("=== Semantic Search Results ===")
        for r in semantic_results:
            context_box_lines.append(
                f"[{r['original_name']} | Passage #{r['passage_id']} | {r['page_or_section']}] "
                f"(cosine similarity: {r['similarity']:.4f})"
            )
            context_box_lines.append(r["text"])
            context_box_lines.append("")
    if keyword_results:
        context_box_lines.append("=== Keyword Search Results (FTS5) ===")
        for r in keyword_results:
            context_box_lines.append(
                f"[{r['original_name']} | Passage #{r['passage_id']} | {r['page_or_section']}]"
            )
            context_box_lines.append(r["text"])
            context_box_lines.append("")

    context_box = "\n".join(context_box_lines) if query else ""
    total_passages = count_indexed_passages()
    embedding_error = get_embedding_error()

    return render_template(
        "search/index.html",
        query=query,
        retrieval_method=retrieval_method,
        keyword_results=keyword_results,
        semantic_results=semantic_results,
        semantic_error=semantic_error,
        embedding_error=embedding_error,
        ollama_answer=ollama_answer,
        ollama_error=ollama_error,
        context_sent=context_sent,
        valid_citations=valid_citations,
        unknown_citations=unknown_citations,
        retrieved_for_model=retrieved_for_model,
        context_box=context_box,
        total_passages=total_passages,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        installed_models=installed_models,
        server_status_error=server_status_error,
    )
