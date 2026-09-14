"""Text cleaning and chunking utilities for the RAG corpus.

The goal is not to create a fancy NLP framework. We want predictable chunks
that preserve enough retail context for citations: product, channel, date,
issue tags, and source type.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize whitespace without changing the meaning of the content.

    RAG chunks should not contain accidental line breaks, tabs, or repeated
    spaces because those make prompts harder to read and can slightly hurt
    retrieval quality. This function keeps the words unchanged and only cleans
    formatting noise.
    """

    return WHITESPACE_RE.sub(" ", text or "").strip()


def stable_id(*parts: object, length: int = 24) -> str:
    """Create a deterministic id from business keys.

    Stable ids make the pipeline idempotent. If the same review arrives again,
    we upsert the same chunk id instead of creating duplicates.
    """

    joined = "||".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def split_words(text: str, max_words: int) -> Iterable[str]:
    """Yield word windows of roughly max_words.

    Product reviews and tickets are short, but manuals and policies can be
    longer. Word windows keep the implementation easy to explain in class.
    """

    # Word-based chunking is easy to explain in a bootcamp and is sufficient
    # for these short reviews, tickets, FAQs, policies, and product notes.
    words = clean_text(text).split(" ")
    for start in range(0, len(words), max_words):
        window = words[start : start + max_words]
        if window:
            yield " ".join(window)


def make_chunk_records(
    source_id: str,
    source_type: str,
    product_id: str | None,
    event_date: str,
    title: str,
    body: str,
    metadata: dict[str, object] | None = None,
    max_words: int = 140,
) -> list[dict[str, object]]:
    """Convert one source record into one or more RAG chunks.

    Every review, ticket, FAQ, policy, or manual is converted into the same
    schema. This makes retrieval simple because Lakebase only needs one table:
    `rag_chunks`.

    Key fields:
    - `chunk_id`: deterministic primary key for idempotent upserts.
    - `source_id`: original review_id, ticket_id, or doc_id.
    - `source_type`: review, support_ticket, faq, policy, or manual.
    - `product_id`: optional filter for product-specific retrieval.
    - `metadata`: source-specific details such as rating or priority.
    """

    metadata = metadata or {}

    # Clean title/body once before splitting. This avoids repeating cleanup work
    # for every chunk created from the same source record.
    title = clean_text(title)
    body = clean_text(body)
    chunks: list[dict[str, object]] = []

    for position, chunk_text in enumerate(split_words(body, max_words)):
        # The position is part of the id so a long document can produce several
        # stable chunks without collisions.
        chunk_id = stable_id(source_type, source_id, position)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "source_id": source_id,
                "source_type": source_type,
                "product_id": product_id,
                "event_date": event_date,
                "chunk_position": position,
                "title": title,
                "chunk_text": chunk_text,
                "metadata": metadata,
            }
        )

    return chunks


def keyword_score(query: str, text: str) -> float:
    """Small lexical scoring helper used by the transparent reranker.

    Lakebase vector retrieval gives us semantic similarity search. This helper
    provides a simple additional lexical signal when teaching reranking without
    another service dependency.
    """

    # Extract simple alphanumeric tokens from both strings. This intentionally
    # avoids stemming or NLP libraries so students can see the reranking signal.
    query_terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", query)}
    text_terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", text)}
    if not query_terms:
        return 0.0

    # Score is the fraction of query terms that appear in the candidate text.
    # Example: if the query has 4 terms and 2 appear in the chunk, score = 0.5.
    return len(query_terms & text_terms) / len(query_terms)
