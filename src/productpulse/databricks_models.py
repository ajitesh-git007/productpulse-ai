"""Databricks Foundation Model API helpers.

This file is the only place where the project directly calls Databricks-hosted
AI models. It uses the Databricks SDK serving endpoint client instead of an
OpenAI-compatible helper so notebook jobs, Databricks Apps, and local runs all
follow the same Databricks authentication chain.

Models used by default:
- Embeddings: `databricks-qwen3-embedding-0-6b`
- Chat response: `databricks-meta-llama-3-3-70b-instruct`

Both values come from environment variables through `load_settings()`, so an
instructor can switch endpoints without changing source code.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from productpulse.config import load_settings


def embedding_model_candidates(primary_model: str) -> list[str]:
    """Return embedding endpoint names to try, in order.

    Databricks exposes some foundation models through slightly different names
    in different UI/API surfaces. The user's workspace test confirmed the
    endpoint-style Qwen name works, so it is tried first. The `system.ai` name
    remains as a last alias in case another workspace exposes that path.
    """

    candidates = [primary_model]
    fallback_models = [
        "databricks-qwen3-embedding-0-6b",
        "system.ai.qwen3-embedding-0-6b",
    ]
    for model in fallback_models:
        if model not in candidates:
            candidates.append(model)
    return candidates


def response_as_dict(response: Any) -> dict[str, Any]:
    """Convert Databricks SDK response objects into plain dictionaries.

    The SDK returns typed response objects in Databricks, while tests or future
    helper code may return dictionaries directly. Normalizing once here makes
    the embedding and chat parsers small and readable.
    """

    if isinstance(response, dict):
        return response
    if hasattr(response, "as_dict"):
        return response.as_dict()
    result: dict[str, Any] = {}
    for key in ["data", "predictions", "choices"]:
        value = getattr(response, key, None)
        if value is not None:
            result[key] = value
    return result


def extract_embedding_vectors(response: Any) -> list[list[float]]:
    """Pull embedding vectors out of a serving endpoint response.

    Foundation Model embedding responses normally look like OpenAI responses:
    `{"data": [{"embedding": [...]}]}`. This helper also accepts a few common
    SDK shapes so the code remains robust across Databricks runtime versions.
    """

    payload = response_as_dict(response)
    data = payload.get("data") or payload.get("predictions") or []
    if isinstance(data, dict):
        data = data.get("data") or data.get("predictions") or []

    vectors: list[list[float]] = []
    for item in data:
        if isinstance(item, dict):
            vector = item.get("embedding")
        else:
            vector = getattr(item, "embedding", None)
        if vector is not None:
            vectors.append([float(value) for value in vector])
    return vectors


def extract_chat_text(response: Any) -> str:
    """Pull the assistant text out of a serving endpoint chat response."""

    payload = response_as_dict(response)
    choices = payload.get("choices") or []
    if not choices:
        return str(payload)

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        first_choice = first_choice.as_dict() if hasattr(first_choice, "as_dict") else {}

    message = first_choice.get("message") or {}
    if not isinstance(message, dict):
        message = message.as_dict() if hasattr(message, "as_dict") else {}
    return message.get("content") or first_choice.get("text") or ""


def embed_texts(texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
    """Create semantic vectors for a list of text chunks.

    Where this is used:
    - Notebook `03_embed_and_sync_lakebase.py` embeds every silver RAG chunk.
    - `retrieval.py` embeds the rewritten user query during chat.

    Input:
    - `texts`: any iterable of strings. The function converts it to a list
      because the Databricks API expects a batch-like payload.

    Output:
    - A list of embedding vectors in the same order as the input texts. This
      positional alignment is important because the ingestion notebook zips
      chunk rows and vectors together before writing to Lakebase.

    Why batching exists:
    - The dataset is small enough for a class project, but PDF chunking can
      still create hundreds of inputs.
    - Sending all chunks in one model request can create large payloads and make
      serverless notebook sessions less stable.
    - Small batches are easier to retry and friendlier to endpoint limits.
    """

    text_list = list(texts)
    if not text_list:
        return []

    # Read the configured endpoint name at runtime. This keeps the source code
    # portable across Free Edition workspaces and paid workspaces.
    settings = load_settings()
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as exc:
        raise RuntimeError(
            "databricks-sdk is required for real embeddings. "
            "Install requirements.txt before running the notebook or app."
        ) from exc

    # WorkspaceClient uses the identity available in the current Databricks
    # context. In a workflow job this is the job run identity; locally it uses
    # DATABRICKS_HOST and DATABRICKS_TOKEN if they are set.
    workspace = WorkspaceClient()
    embeddings: list[list[float]] = []
    models_to_try = embedding_model_candidates(settings.embedding_model)
    total_batches = (len(text_list) + batch_size - 1) // batch_size
    print(
        f"Embedding {len(text_list)} text chunks in {total_batches} batches "
        f"using candidates={models_to_try}",
        flush=True,
    )
    for start in range(0, len(text_list), batch_size):
        batch = text_list[start : start + batch_size]
        batch_number = (start // batch_size) + 1
        last_error: Exception | None = None
        response = None
        for model_name in models_to_try:
            try:
                started_at = time.monotonic()
                print(
                    f"Embedding batch {batch_number}/{total_batches} "
                    f"with model={model_name}, rows={len(batch)}",
                    flush=True,
                )
                response = workspace.serving_endpoints.query(
                    name=model_name,
                    input=batch,
                )
                batch_embeddings = extract_embedding_vectors(response)
                if len(batch_embeddings) != len(batch):
                    raise RuntimeError(
                        f"Endpoint '{model_name}' returned {len(batch_embeddings)} "
                        f"embeddings for {len(batch)} input texts."
                    )
                embeddings.extend(batch_embeddings)
                elapsed = time.monotonic() - started_at
                print(
                    f"Finished batch {batch_number}/{total_batches} "
                    f"with {len(batch_embeddings)} embeddings in {elapsed:.1f}s",
                    flush=True,
                )
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"Embedding batch {batch_number}/{total_batches} failed "
                    f"with model={model_name}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        if response is None:
            raise RuntimeError(
                "Embedding endpoint call failed. Tried these endpoint names: "
                f"{models_to_try}. In Databricks, open system.ai or Serving and "
                "confirm which embedding model exists in your workspace. If it "
                "has a different name, pass the correct `embedding_model` job "
                "parameter or set PRODUCTPULSE_EMBEDDING_MODEL. Last Databricks "
                f"error: {type(last_error).__name__}: {last_error}"
            ) from last_error

    return embeddings


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 900,
) -> str:
    """Generate one chat response from a Databricks-hosted LLM.

    Where this is used:
    - `retrieval.rewrite_query()` asks the model to convert a vague business
      question into a better search query.
    - `agent.answer_question()` asks the model to synthesize the final answer
      from retrieved evidence, metrics, and memory.

    The function accepts separate system and user prompts because production
    RAG systems usually keep durable behavior instructions in the system prompt
    and request-specific context in the user prompt.
    """

    settings = load_settings()
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    except ImportError as exc:
        raise RuntimeError(
            "databricks-sdk is required for LLM responses. "
            "Install requirements.txt before running the notebook or app."
        ) from exc

    # Keep temperature low for an enterprise copilot. The goal is grounded,
    # repeatable support guidance, not creative writing.
    workspace = WorkspaceClient()
    try:
        response = workspace.serving_endpoints.query(
            name=settings.llm_endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=ChatMessageRole.USER, content=user_prompt),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(
            "Chat endpoint call failed. The configured endpoint is "
            f"'{settings.llm_endpoint}'. Confirm this endpoint exists in your "
            "Databricks workspace/region, or set PRODUCTPULSE_LLM_ENDPOINT to "
            f"an available chat endpoint. Last Databricks error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return extract_chat_text(response)
