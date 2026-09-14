# Databricks notebook source
# ProductPulse AI - 00 setup Unity Catalog objects
#
# Purpose:
# Create the catalog/schema, a Unity Catalog volume for raw files, and Delta
# tables used by the rest of the project. This notebook is safe to rerun.
#
# Why this is separated as its own job:
# Governance objects should be created deliberately, not hidden inside the
# scheduled daily ingestion flow. Run this setup job manually first, then run it
# again only when the physical data model changes.

# COMMAND ----------

dbutils.widgets.text("catalog", "adidas_retail_ai", "Unity Catalog catalog")
dbutils.widgets.text("schema", "productpulse", "Unity Catalog schema")

# Widgets make the notebook reusable from both places:
# - the Databricks Jobs UI, where parameters can be changed before a run
# - the Databricks Asset Bundle, where the same values are passed automatically
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# Unity Catalog hierarchy used by the project:
# - catalog: project-level governance boundary
# - schema: database-like namespace for ProductPulse tables
# - volume: governed object storage path for raw CSV, JSONL, and PDF files
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS  `{catalog}`.`{schema}`")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_data")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# Product master is a small dimension table. It gives downstream metrics and
# RAG metadata human-readable names, categories, fit profiles, and known launch
# risk themes.
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS bronze_products (
      product_id STRING,
      product_name STRING,
      category STRING,
      gender STRING,
      launch_date DATE,
      price_usd DOUBLE,
      fit_profile STRING,
      known_risk_theme STRING,
      ingestion_ts TIMESTAMP
    )
    USING DELTA
    """
)

# Reviews are both analytics data and RAG evidence:
# - rating, sentiment, issue_tag, channel, and market feed Gold metrics
# - review_title and review_body feed Silver RAG chunks
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS bronze_reviews (
      review_id STRING,
      event_date DATE,
      product_id STRING,
      rating INT,
      review_title STRING,
      review_body STRING,
      sentiment STRING,
      issue_tag STRING,
      channel STRING,
      locale STRING,
      market STRING,
      verified_purchase BOOLEAN,
      ingestion_ts TIMESTAMP
    )
    USING DELTA
    PARTITIONED BY (event_date)
    """
)

# Support tickets are also hybrid data:
# - structured columns feed issue counts and support workload metrics
# - ticket_body becomes unstructured evidence for the RAG path
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS bronze_support_tickets (
      ticket_id STRING,
      event_date DATE,
      order_id STRING,
      product_id STRING,
      reason STRING,
      ticket_body STRING,
      status STRING,
      priority STRING,
      channel STRING,
      customer_segment STRING,
      ingestion_ts TIMESTAMP
    )
    USING DELTA
    PARTITIONED BY (event_date)
    """
)

# Orders are structured facts only. They help calculate demand, units sold, and
# return rate denominators. They are not chunked into RAG.
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS bronze_orders (
      order_id STRING,
      event_date DATE,
      product_id STRING,
      quantity INT,
      unit_price_usd DOUBLE,
      market STRING,
      channel STRING,
      customer_segment STRING,
      ingestion_ts TIMESTAMP
    )
    USING DELTA
    PARTITIONED BY (event_date)
    """
)

# Returns are structured facts only. The agent uses them through Gold metrics,
# while RAG explanations come from reviews, tickets, and PDF guidance.
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS bronze_returns (
      return_id STRING,
      event_date DATE,
      order_id STRING,
      product_id STRING,
      return_reason STRING,
      refund_amount_usd DOUBLE,
      is_exchange BOOLEAN,
      condition STRING,
      ingestion_ts TIMESTAMP
    )
    USING DELTA
    PARTITIONED BY (event_date)
    """
)

# Inventory snapshots are structured facts only. They support operational
# questions such as whether exchange-first guidance is realistic.
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS bronze_inventory_snapshots (
      snapshot_date DATE,
      product_id STRING,
      warehouse_id STRING,
      on_hand_units INT,
      reserved_units INT,
      reorder_point INT,
      stock_status STRING,
      ingestion_ts TIMESTAMP
    )
    USING DELTA
    PARTITIONED BY (snapshot_date)
    """
)

# Knowledge documents store extracted PDF text plus document metadata. The
# original PDF files remain in the raw Unity Catalog volume; this Bronze table
# is the queryable/extractable copy used by chunking.
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS bronze_knowledge_docs (
      doc_id STRING,
      source_type STRING,
      product_id STRING,
      title STRING,
      body STRING,
      effective_date DATE,
      file_name STRING,
      page_count INT,
      ingestion_ts TIMESTAMP
    )
    USING DELTA
    """
)

# COMMAND ----------

# Silver RAG chunk table. Every review, support ticket, and PDF section is
# normalized into this one schema so retrieval does not need to understand each
# source system separately.
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS silver_document_chunks (
      chunk_id STRING,
      source_id STRING,
      source_type STRING,
      product_id STRING,
      event_date DATE,
      chunk_position INT,
      title STRING,
      chunk_text STRING,
      metadata_json STRING,
      updated_at TIMESTAMP
    )
    USING DELTA
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """
)

# Silver embeddings table. This keeps a governed Delta copy of vectors before
# they are synced into Lakebase. It is useful for lineage, reprocessing, and
# debugging model output without querying Postgres directly.
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS silver_document_embeddings (
      chunk_id STRING,
      source_id STRING,
      source_type STRING,
      product_id STRING,
      event_date DATE,
      chunk_position INT,
      title STRING,
      chunk_text STRING,
      metadata_json STRING,
      embedding ARRAY<DOUBLE>,
      embedded_at TIMESTAMP
    )
    USING DELTA
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """
)

# Gold product table. This is the main table behind SQL-only and hybrid metric
# answers, such as "top products by ticket count" or "highest return rate".
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS gold_product_daily (
      event_date DATE,
      product_id STRING,
      product_name STRING,
      category STRING,
      review_count BIGINT,
      ticket_count BIGINT,
      order_count BIGINT,
      units_sold BIGINT,
      return_count BIGINT,
      return_rate DOUBLE,
      avg_rating DOUBLE,
      negative_sentiment_count BIGINT,
      return_related_count BIGINT,
      on_hand_units BIGINT,
      stock_status STRING,
      top_issue STRING,
      updated_at TIMESTAMP
    )
    USING DELTA
    """
)

# Gold issue table. This stores issue-level aggregates so learners can extend
# the SQL tool later for questions like "which issue is growing fastest".
spark.sql(
    """
    CREATE TABLE IF NOT EXISTS gold_issue_daily (
      event_date DATE,
      product_id STRING,
      issue_tag STRING,
      feedback_count BIGINT,
      negative_count BIGINT,
      high_priority_ticket_count BIGINT,
      updated_at TIMESTAMP
    )
    USING DELTA
    """
)

print(f"ProductPulse setup complete in {catalog}.{schema}")
