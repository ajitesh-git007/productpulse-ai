"""Query rewrite, retrieval, and reranking for the copilot.

This module owns the RAG retrieval sequence. The agent asks this module for
evidence, and this module returns both:
- `context`: formatted text blocks that can be placed into the final LLM prompt.
- `evidence`: structured rows that can be displayed in the Streamlit trace.

Keeping retrieval separate from the agent makes the project easier to teach:
students can first understand RAG mechanics, then look at agent tool selection.
"""

from __future__ import annotations

from productpulse.config import load_settings
from productpulse.databricks_models import chat_completion, embed_texts
from productpulse.lakebase import search_chunks
from productpulse.text_processing import keyword_score


QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite retail analytics questions for retrieval.
Return one concise search query. Keep product ids, issue words, dates, and brand terms."""


def rewrite_query(question: str) -> str:
    """Rewrite conversational wording into a retrieval-friendly query.

    Users rarely type search-optimized text. They ask business questions like
    "why are customers returning this shoe?" The rewrite step turns that into a
    compact query that preserves product ids, dates, and issue phrases. Better
    query text generally improves embedding retrieval quality.
    """

    prompt = (
        "Rewrite this user question for searching customer reviews, support "
        f"tickets, FAQs, manuals, and policy chunks:\n\n{question}"
    )
    return chat_completion(
        QUERY_REWRITE_SYSTEM_PROMPT,
        prompt,
        temperature=0.0,
        max_tokens=80,
    ).strip()


def rerank_chunks(question: str, rows: list[dict[str, object]], rerank_k: int) -> list[dict[str, object]]:
    """Blend vector distance with a small keyword score.

    This is intentionally transparent for teaching. Lakebase supplies semantic
    nearest-neighbor candidates, and this function adds a small lexical signal
    that is easy for learners to inspect and explain.
    """

    scored = []
    for row in rows:
        # Lakebase returns smaller vector distance for closer semantic matches.
        # We convert that distance into a positive score with 1 / (1 + distance).
        distance = float(row.get("distance") or 0.0)

        # The lexical score is deliberately simple and transparent. It rewards
        # evidence that contains exact business terms like "return", "toe box",
        # "wide feet", or the product id.
        lexical = keyword_score(question, f"{row.get('title', '')} {row.get('chunk_text', '')}")
        score = (1.0 / (1.0 + distance)) + (0.25 * lexical)

        # Copy the row before adding a score so callers still receive all source
        # metadata plus the new rerank score used by the agent trace.
        row_with_score = dict(row)
        row_with_score["rerank_score"] = round(score, 6)
        scored.append(row_with_score)

    return sorted(scored, key=lambda row: row["rerank_score"], reverse=True)[:rerank_k]


def retrieve_context(
    conn,
    question: str,
    product_id: str | None = None,
) -> dict[str, object]:
    """Run the full RAG retrieval sequence.

    Steps performed:
    1. Rewrite the user's question for better search.
    2. Embed the rewritten query with the Databricks embedding endpoint.
    3. Search Lakebase `rag_chunks` for nearest text chunks.
    4. Rerank candidates with semantic distance plus keyword overlap.
    5. Format top evidence into numbered citation blocks.

    The final answer model sees numbered blocks such as `[1]`, `[2]`, so it can
    cite where each claim came from.
    """

    settings = load_settings()
    rewritten_query = rewrite_query(question)
    query_embedding = embed_texts([rewritten_query])[0]
    candidates = search_chunks(conn, query_embedding, top_k=settings.top_k, product_id=product_id)
    evidence = rerank_chunks(question, candidates, rerank_k=settings.rerank_k)

    context_blocks = []
    for index, row in enumerate(evidence, start=1):
        # Each block carries human-readable source metadata. This is valuable
        # for both final answer citations and classroom debugging.
        context_blocks.append(
            "[{idx}] {title} | source={source_type} | product={product_id} | date={event_date}\n{text}".format(
                idx=index,
                title=row.get("title"),
                source_type=row.get("source_type"),
                product_id=row.get("product_id"),
                event_date=row.get("event_date"),
                text=row.get("chunk_text"),
            )
        )

    return {
        "rewritten_query": rewritten_query,
        "evidence": evidence,
        "context": "\n\n".join(context_blocks),
    }
