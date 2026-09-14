-- ProductPulse AI Lakebase schema
--
-- Run this in the Lakebase SQL editor if you want to initialize Lakebase
-- manually before executing notebook 03. The notebook also creates these
-- objects automatically when the app/notebook has permissions.

-- Use standard pgvector. This works without enabling Lakebase Search.
-- Lakebase Search/lakebase_vector is optional and only needed for ANN indexing
-- at much larger scale.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
  chunk_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  source_type TEXT NOT NULL,
  product_id TEXT,
  event_date DATE NOT NULL,
  chunk_position INT NOT NULL,
  title TEXT NOT NULL,
  chunk_text TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  embedding VECTOR(1024) NOT NULL,
  loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_memory (
  memory_id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL,
  memory_type TEXT NOT NULL,
  memory_text TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rag_chunks_product_date_idx
ON rag_chunks (product_id, event_date);

CREATE INDEX IF NOT EXISTS conversation_memory_session_idx
ON conversation_memory (session_id, created_at DESC);

