"""Lakebase storage functions for embeddings and agent memory.

Lakebase is Postgres-compatible. This module uses `pg8000`, a pure-Python
Postgres driver, because Databricks serverless can run on different CPU
architectures and native driver wheels can occasionally make the notebook
kernel unstable. Keeping this layer pure Python makes the class project more
reliable on Free Edition serverless jobs.
"""

from __future__ import annotations

import json
import ssl
from contextlib import contextmanager
from typing import Any, Iterator

import pg8000.dbapi

from productpulse.config import get_env, lakebase_connection_info


def lakebase_password() -> str:
    """Return the password/token used for a Lakebase connection.

    There are two supported modes:

    1. Static password mode:
       If `PGPASSWORD` is set, use it directly. This is useful for local demos
       or password-based Postgres roles.

    2. Databricks Apps + Lakebase Autoscaling OAuth mode:
       If `PGPASSWORD` is not set, use `LAKEBASE_ENDPOINT` and the Databricks SDK
       to generate a short-lived database credential. In a Databricks App, the
       SDK authenticates as the app service principal because Databricks injects
       `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET`.

    The generated token is intentionally not stored. Lakebase OAuth database
    credentials expire, so a fresh token is requested when opening a connection.
    """

    static_password = get_env("PGPASSWORD")
    if static_password:
        return static_password

    endpoint = get_env("LAKEBASE_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "Lakebase password is not configured. Set PGPASSWORD for a password "
            "role, or attach a Lakebase database resource to the Databricks App "
            "and expose LAKEBASE_ENDPOINT with valueFrom: postgres."
        )

    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as exc:
        raise RuntimeError(
            "databricks-sdk>=0.89.0 is required to generate Lakebase OAuth "
            "database credentials. Install requirements.txt."
        ) from exc

    credential = WorkspaceClient().postgres.generate_database_credential(endpoint=endpoint)
    return credential.token


def vector_literal(values: list[float]) -> str:
    """Format a Python list as a pgvector-compatible literal.

    Explicit vector literals make the project easier to run in notebooks because
    we do not need a database-driver-specific pgvector adapter. Example output:
    `[0.12,-0.03,0.44]`.
    """

    return "[" + ",".join(str(float(value)) for value in values) + "]"


def ssl_context_from_mode(sslmode: str) -> bool | ssl.SSLContext | None:
    """Translate familiar Postgres sslmode values into pg8000 settings.

    Lakebase normally requires SSL, so the project default is `require`. pg8000
    accepts `True` to require SSL with its standard context. We keep `disable`
    available for local Postgres demos where SSL is not configured.
    """

    mode = (sslmode or "require").lower()
    if mode == "disable":
        return False
    if mode in {"require", "prefer", "allow"}:
        return True
    if mode in {"verify-ca", "verify-full"}:
        return ssl.create_default_context()
    return True


def rows_as_dicts(cursor: Any) -> list[dict[str, object]]:
    """Convert DB-API result rows into dictionaries.

    pg8000 follows the DB-API default and returns tuples. This helper keeps the
    rest of the application readable: retrieval and memory code can work with
    dictionaries even though the database driver returns positional rows.
    """

    column_names = [column[0] for column in cursor.description]
    rows = []
    for row in cursor.fetchall():
        row_dict = dict(zip(column_names, row))
        for json_column in ["metadata"]:
            if isinstance(row_dict.get(json_column), str):
                row_dict[json_column] = json.loads(row_dict[json_column])
        rows.append(row_dict)
    return rows


@contextmanager
def connect() -> Iterator[Any]:
    """Open one Lakebase connection and manage the transaction lifecycle.

    Callers use this as:

        with connect() as conn:
            ...

    If the block succeeds, changes are committed. If anything fails, changes are
    rolled back so partial writes do not leave Lakebase in an inconsistent
    state.
    """

    connection_info = lakebase_connection_info()
    connection_info["password"] = lakebase_password()
    sslmode = connection_info.pop("sslmode", "require")
    connection_info["port"] = int(connection_info["port"])
    conn = pg8000.dbapi.connect(
        **connection_info,
        ssl_context=ssl_context_from_mode(sslmode),
        timeout=60,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_lakebase_schema(conn: Any, embedding_dimensions: int = 1024) -> None:
    """Create the Lakebase tables required by retrieval and memory.

    The schema uses the standard `vector` extension from pgvector. Lakebase also
    offers the separate `lakebase_vector` ANN extension when Lakebase Search is
    enabled, but that feature requires project-level preload settings. For this
    Free Edition bootcamp project, exact pgvector search is simpler and reliable
    at the small demo scale.
    """

    cur = conn.cursor()
    try:
        # Enable the pgvector data type and distance operators. This does not
        # require Lakebase Search to be enabled.
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # RAG chunks table:
        # - chunk_id is deterministic, so ingestion can upsert safely.
        # - metadata is JSONB so different source types can carry different
        #   details such as rating, sentiment, priority, or effective date.
        # - embedding is the vector used for similarity search.
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS rag_chunks (
              chunk_id TEXT PRIMARY KEY,
              source_id TEXT NOT NULL,
              source_type TEXT NOT NULL,
              product_id TEXT,
              event_date DATE NOT NULL,
              chunk_position INT NOT NULL,
              title TEXT NOT NULL,
              chunk_text TEXT NOT NULL,
              metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
              embedding VECTOR({embedding_dimensions}) NOT NULL,
              loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        # Conversation memory table:
        # This stores only compact memory items, not the full chat transcript.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_memory (
              memory_id BIGSERIAL PRIMARY KEY,
              session_id TEXT NOT NULL,
              memory_type TEXT NOT NULL,
              memory_text TEXT NOT NULL,
              metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        # Filtering by product and date is common in retail support questions.
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS rag_chunks_product_date_idx
            ON rag_chunks (product_id, event_date);
            """
        )
        # Session memory should be read by latest item first.
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS conversation_memory_session_idx
            ON conversation_memory (session_id, created_at DESC);
            """
        )
    finally:
        cur.close()


def upsert_chunks(
    conn: Any,
    chunks: list[dict[str, object]],
    embeddings: list[list[float]],
) -> int:
    """Insert or update chunk embeddings in Lakebase.

    This is used by notebook `03_embed_and_sync_lakebase.py` after chunks have
    been embedded. The upsert behavior is important for scheduled pipelines:
    rerunning the same day should refresh existing rows, not duplicate them.
    """

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        # Convert the Python dictionary into the exact positional tuple expected
        # by the SQL INSERT statement below.
        rows.append(
            (
                chunk["chunk_id"],
                chunk["source_id"],
                chunk["source_type"],
                chunk.get("product_id"),
                chunk["event_date"],
                chunk["chunk_position"],
                chunk["title"],
                chunk["chunk_text"],
                json.dumps(chunk.get("metadata", {})),
                vector_literal(embedding),
            )
        )

    cur = conn.cursor()
    try:
        # ON CONFLICT turns the insert into an idempotent upsert keyed by
        # chunk_id. If the text or embedding changes, the existing row is
        # refreshed and loaded_at is updated.
        cur.executemany(
            """
            INSERT INTO rag_chunks (
              chunk_id, source_id, source_type, product_id, event_date,
              chunk_position, title, chunk_text, metadata, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (chunk_id) DO UPDATE SET
              title = EXCLUDED.title,
              chunk_text = EXCLUDED.chunk_text,
              metadata = EXCLUDED.metadata,
              embedding = EXCLUDED.embedding,
              loaded_at = now();
            """,
            rows,
        )
    finally:
        cur.close()
    return len(rows)


def search_chunks(
    conn: Any,
    query_embedding: list[float],
    top_k: int = 8,
    product_id: str | None = None,
) -> list[dict[str, object]]:
    """Retrieve nearest chunks from Lakebase using pgvector-compatible syntax.

    Input:
    - `query_embedding`: vector for the rewritten user question.
    - `top_k`: number of candidate chunks to return before reranking.
    - `product_id`: optional metadata filter for product-specific questions.

    Output:
    - Rows with chunk text, metadata, and vector distance. Lower distance means
      the chunk is semantically closer to the query.
    """

    query_vector = vector_literal(query_embedding)
    params: list[object] = [query_vector]
    filter_sql = ""
    if product_id:
        filter_sql = "WHERE product_id = %s"
        params.append(product_id)
    params.extend([query_vector, top_k])

    cur = conn.cursor()
    try:
        # `<->` is the vector distance operator. It is used both in SELECT so we
        # can display the distance later, and in ORDER BY so Lakebase returns the
        # nearest chunks first.
        cur.execute(
            f"""
            SELECT
              chunk_id,
              source_id,
              source_type,
              product_id,
              event_date,
              title,
              chunk_text,
              metadata,
              embedding <-> %s::vector AS distance
            FROM rag_chunks
            {filter_sql}
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
            """,
            params,
        )
        return rows_as_dicts(cur)
    finally:
        cur.close()


def read_memory(conn: Any, session_id: str, limit: int = 5) -> list[dict[str, object]]:
    """Read recent memory facts for the current conversation.

    Memory is scoped by session_id so one user's preferences do not leak into
    another user's conversation.
    """

    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT memory_type, memory_text, metadata, created_at
            FROM conversation_memory
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s;
            """,
            (session_id, limit),
        )
        return rows_as_dicts(cur)
    finally:
        cur.close()


def write_memory(
    conn: Any,
    session_id: str,
    memory_type: str,
    memory_text: str,
    metadata: dict[str, object] | None = None,
) -> None:
    """Persist one compact memory item for future turns.

    The project stores two simple memory types in the same table:
    - `conversation_turn`: a compact summary of a normal user/assistant turn.
    - `user_preference`: a durable instruction or preference the user gave.

    It should not be used for raw sensitive support transcripts.
    """

    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO conversation_memory (session_id, memory_type, memory_text, metadata)
            VALUES (%s, %s, %s, %s::jsonb);
            """,
            (session_id, memory_type, memory_text, json.dumps(metadata or {})),
        )
    finally:
        cur.close()
