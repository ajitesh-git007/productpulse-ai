# Databricks notebook source
# ProductPulse AI - 05 manual agent demo
#
# Purpose:
# Run one end-to-end business question after the Lakebase sync has completed.
# This notebook is useful for classroom demonstration before opening the
# Streamlit app. It shows the final answer and the retrieved evidence.

import os
import sys
from pathlib import Path

dbutils.widgets.text("pg_host", "", "Lakebase host")
dbutils.widgets.text("pg_port", "5432", "Lakebase port")
dbutils.widgets.text("pg_database", "databricks_postgres", "Lakebase database")
dbutils.widgets.text("pg_user", "", "Lakebase user")
dbutils.widgets.text("pg_password", "", "Lakebase password or token")
dbutils.widgets.text("pg_sslmode", "require", "Lakebase SSL mode")
dbutils.widgets.text("embedding_model", "databricks-qwen3-embedding-0-6b", "Embedding serving endpoint")
dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct", "Chat serving endpoint")

# The reusable Python package reads environment variables. The notebook widgets
# are copied into environment variables here so manual notebook runs behave the
# same way as Databricks Jobs and Databricks Apps.
for env_name, widget_name in [
    ("PGHOST", "pg_host"),
    ("PGPORT", "pg_port"),
    ("PGDATABASE", "pg_database"),
    ("PGUSER", "pg_user"),
    ("PGPASSWORD", "pg_password"),
    ("PGSSLMODE", "pg_sslmode"),
    ("PRODUCTPULSE_EMBEDDING_MODEL", "embedding_model"),
    ("PRODUCTPULSE_LLM_ENDPOINT", "llm_endpoint"),
]:
    widget_value = dbutils.widgets.get(widget_name)
    if widget_value:
        os.environ[env_name] = widget_value

# The notebook can run from different working directories depending on whether
# it is launched from the bundle or cloned into the workspace. Add the project
# `src` folder dynamically so imports keep working.
for candidate in [Path.cwd(), Path.cwd().parent]:
    if (candidate / "src" / "productpulse").exists():
        sys.path.insert(0, str(candidate / "src"))
        break

from productpulse.agent import answer_question
from productpulse.lakebase import connect

# This question intentionally exercises the full hybrid path:
# - product id extraction: RUN-ULTRA-01
# - memory read/write through Lakebase conversation_memory
# - RAG search through Lakebase rag_chunks
# - metric SQL when configured
# - final answer synthesis through the Databricks chat endpoint
question = "Why are customers returning RUN-ULTRA-01, and what should support agents do next?"

with connect() as conn:
    result = answer_question(
        conn=conn,
        session_id="notebook-manual-demo",
        question=question,
    )

print(result["answer"])
display(result["evidence"])
