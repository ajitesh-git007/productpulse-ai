# Databricks notebook source
# ProductPulse AI - 03 embed chunks and sync to Lakebase
#
# Purpose:
# Embed new/changed chunks with a Databricks Foundation Model API embedding
# model and upsert them into Lakebase for vector search. The Delta table keeps a
# governed lakehouse copy of the embeddings, while Lakebase powers low-latency
# retrieval and agent memory.

import json
import os
import sys
from pathlib import Path

dbutils.widgets.text("catalog", "adidas_retail_ai", "Unity Catalog catalog")
dbutils.widgets.text("schema", "productpulse", "Unity Catalog schema")
dbutils.widgets.text("pg_host", "", "Lakebase host")
dbutils.widgets.text("pg_port", "5432", "Lakebase port")
dbutils.widgets.text("pg_database", "databricks_postgres", "Lakebase database")
dbutils.widgets.text("pg_user", "", "Lakebase user")
dbutils.widgets.text("pg_password", "", "Lakebase password or token")
dbutils.widgets.text("pg_sslmode", "require", "Lakebase SSL mode")
dbutils.widgets.text("embedding_model", "databricks-qwen3-embedding-0-6b", "Embedding serving endpoint")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
embedding_model = dbutils.widgets.get("embedding_model")

# Job widgets become environment variables because the reusable package reads
# configuration from the environment. This keeps the same code usable in:
# - Databricks notebook jobs
# - Databricks Apps
# - local development with `.env`
for env_name, widget_name in [
    ("PGHOST", "pg_host"),
    ("PGPORT", "pg_port"),
    ("PGDATABASE", "pg_database"),
    ("PGUSER", "pg_user"),
    ("PGPASSWORD", "pg_password"),
    ("PGSSLMODE", "pg_sslmode"),
    ("PRODUCTPULSE_EMBEDDING_MODEL", "embedding_model"),
]:
    widget_value = dbutils.widgets.get(widget_name)
    if widget_value:
        os.environ[env_name] = widget_value

# Databricks can execute this file from the project root or the notebooks
# folder, depending on how it was imported. Add whichever src path exists.
# This avoids requiring a wheel build for the bootcamp project.
for candidate in [Path.cwd(), Path.cwd().parent]:
    if (candidate / "src" / "productpulse").exists():
        sys.path.insert(0, str(candidate / "src"))
        break

from productpulse.lakebase import connect, init_lakebase_schema, upsert_chunks

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

print(
    "Starting embedding sync with "
    f"catalog={catalog}, schema={schema}, "
    f"embedding_model={embedding_model}",
    flush=True,
)

chunks_df = spark.table("silver_document_chunks").orderBy(
    "event_date",
    "source_type",
    "source_id",
    "chunk_position",
)

# Count before embedding so failures are easier to diagnose. If this is zero,
# the issue is earlier in the pipeline, not Lakebase or the model endpoint.
chunk_count = chunks_df.count()
print(f"Prepared {chunk_count} chunks from silver_document_chunks", flush=True)

if chunk_count == 0:
    raise RuntimeError(
        "No rows found in silver_document_chunks. Run 02_build_silver_chunks.py before this task."
    )

chunks_df.createOrReplaceTempView("productpulse_chunks_to_embed")
embedding_model_sql = embedding_model.replace("'", "''")

# Embeddings are generated with Databricks SQL `ai_query` instead of a Python
# loop. This was chosen because serverless notebook kernels can become unstable
# when many model calls happen inside one Python process. `ai_query` delegates
# batch inference to Databricks and writes the result directly to Delta.
print(
    "Writing embeddings to Delta table silver_document_embeddings with ai_query. "
    "This uses Databricks serverless batch inference instead of Python serving endpoint calls.",
    flush=True,
)
spark.sql(
    f"""
    CREATE OR REPLACE TABLE silver_document_embeddings AS
    SELECT
      chunk_id,
      source_id,
      source_type,
      product_id,
      event_date,
      chunk_position,
      title,
      chunk_text,
      COALESCE(metadata_json, '{{}}') AS metadata_json,
      CAST(ai_query('{embedding_model_sql}', chunk_text) AS ARRAY<FLOAT>) AS embedding,
      current_timestamp() AS embedded_at
    FROM productpulse_chunks_to_embed
    """
)
print("Finished writing silver_document_embeddings", flush=True)

# COMMAND ----------

# Lakebase is Postgres-compatible, so we collect the demo-sized embedding table
# to the driver and upsert rows through the pg8000 connection layer. This keeps
# the Lakebase write path explicit and easy to inspect in class.
embedded_rows = [
    row.asDict()
    for row in spark.table("silver_document_embeddings")
    .orderBy("event_date", "source_type", "source_id", "chunk_position")
    .collect()
]
print(f"Collected {len(embedded_rows)} embedded rows for Lakebase sync", flush=True)

lakebase_chunks = []
for row in embedded_rows:
    # Lakebase stores metadata as JSONB. The Silver table stores it as a JSON
    # string because Spark tables and CSV-like display handle strings cleanly.
    lakebase_chunks.append(
        {
            "chunk_id": row["chunk_id"],
            "source_id": row["source_id"],
            "source_type": row["source_type"],
            "product_id": row.get("product_id"),
            "event_date": str(row["event_date"]),
            "chunk_position": row["chunk_position"],
            "title": row["title"],
            "chunk_text": row["chunk_text"],
            "metadata": json.loads(row.get("metadata_json") or "{}"),
        }
    )

print("Initializing Lakebase schema and upserting chunk embeddings", flush=True)
with connect() as conn:
    # The embedding dimension is inferred from the actual model output. This
    # prevents schema mismatch if an instructor switches from Qwen to BGE/GTE.
    init_lakebase_schema(conn, embedding_dimensions=len(embedded_rows[0]["embedding"]))
    inserted = upsert_chunks(conn, lakebase_chunks, [row["embedding"] for row in embedded_rows])

print(f"Embedded and synced {inserted} chunks to Lakebase", flush=True)
