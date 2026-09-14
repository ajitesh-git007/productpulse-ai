# Databricks notebook source
# ProductPulse AI - 02 build silver RAG chunks
#
# Purpose:
# Standardize customer reviews, support tickets, and internal PDF documents into
# one chunk table. Reviews/tickets are short enough to be one chunk each. PDFs
# are split into several character windows so citations can point to smaller,
# more relevant pieces of evidence.

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "adidas_retail_ai", "Unity Catalog catalog")
dbutils.widgets.text("schema", "productpulse", "Unity Catalog schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# Reviews are usually short enough to keep as one RAG chunk per review. The
# deterministic chunk_id matches the pattern used by the Python text utility:
# source type + source id + chunk position.
reviews = spark.table("bronze_reviews").select(
    F.sha2(F.concat_ws("||", F.lit("review"), F.col("review_id"), F.lit("0")), 256).substr(1, 24).alias("chunk_id"),
    F.col("review_id").alias("source_id"),
    F.lit("review").alias("source_type"),
    "product_id",
    "event_date",
    F.lit(0).alias("chunk_position"),
    F.col("review_title").alias("title"),
    F.col("review_body").alias("chunk_text"),
    F.to_json(
        F.struct(
            "rating",
            "sentiment",
            "issue_tag",
            "channel",
            "locale",
            "market",
            "verified_purchase",
        )
    ).alias("metadata_json"),
    F.current_timestamp().alias("updated_at"),
)

# Support tickets are also kept as one chunk each. The ticket reason becomes
# part of the title so retrieval can match phrases such as "return",
# "exchange", or "warranty" even when the ticket body is short.
tickets = spark.table("bronze_support_tickets").select(
    F.sha2(F.concat_ws("||", F.lit("ticket"), F.col("ticket_id"), F.lit("0")), 256).substr(1, 24).alias("chunk_id"),
    F.col("ticket_id").alias("source_id"),
    F.lit("support_ticket").alias("source_type"),
    "product_id",
    "event_date",
    F.lit(0).alias("chunk_position"),
    F.concat(F.lit("Support ticket: "), F.col("reason")).alias("title"),
    F.col("ticket_body").alias("chunk_text"),
    F.to_json(F.struct("order_id", "reason", "status", "priority", "channel", "customer_segment")).alias("metadata_json"),
    F.current_timestamp().alias("updated_at"),
)

# PDF bodies are longer than reviews/tickets. This block creates chunk positions
# 0, 1, 2... and extracts 900-character windows from each PDF's text.
# Character windows are used here because this notebook stays entirely inside
# Spark SQL functions. The Python utility uses word windows for local code, but
# both approaches produce the same target schema.
pdf_chunk_size = 900
docs_base = (
    spark.table("bronze_knowledge_docs")
    .where(F.length(F.coalesce(F.col("body"), F.lit(""))) > 0)
    .withColumn(
        "chunk_position",
        F.explode(
            F.sequence(
                F.lit(0),
                F.floor((F.length("body") - F.lit(1)) / F.lit(pdf_chunk_size)).cast("int"),
            )
        ),
    )
)

# PDF chunk ids include source_type, doc_id, and chunk_position. This lets the
# Lakebase sync upsert the same PDF chunk when the pipeline is rerun.
docs = docs_base.select(
    F.sha2(F.concat_ws("||", F.col("source_type"), F.col("doc_id"), F.col("chunk_position")), 256).substr(1, 24).alias("chunk_id"),
    F.col("doc_id").alias("source_id"),
    "source_type",
    "product_id",
    F.col("effective_date").alias("event_date"),
    "chunk_position",
    F.concat(F.col("title"), F.lit(" - part "), (F.col("chunk_position") + F.lit(1)).cast("string")).alias("title"),
    F.substring(F.col("body"), (F.col("chunk_position") * F.lit(pdf_chunk_size)) + F.lit(1), pdf_chunk_size).alias("chunk_text"),
    F.to_json(F.struct("effective_date", "file_name", "page_count")).alias("metadata_json"),
    F.current_timestamp().alias("updated_at"),
)

# This union is the key Silver modeling step: different source systems become
# one common RAG corpus. Downstream retrieval does not care whether a row came
# from a review, support ticket, SOP, policy, product brief, or QA guide.
chunks = reviews.unionByName(tickets).unionByName(docs)

# COMMAND ----------

# Overwrite keeps the class demo deterministic. A production scheduled version
# could change this to incremental merge using chunk_id and Delta Change Data
# Feed.
chunks.write.mode("overwrite").saveAsTable("silver_document_chunks")

print(f"Silver chunk build complete: {chunks.count()} chunks")
