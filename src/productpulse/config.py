"""Configuration helpers for notebooks, scripts, and the Streamlit app.

Databricks projects often run in several places:
1. Locally from a laptop while developing.
2. Inside a Databricks notebook.
3. Inside a Databricks App with resources injected as environment variables.

This module keeps the access pattern consistent across those locations. The
code reads simple environment variables and never hard-codes workspace-specific
secrets into source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def get_env(name: str, default: str | None = None) -> str | None:
    """Read one environment variable.

    A tiny wrapper keeps the rest of the code readable and gives us one place
    to add validation later if the bootcamp wants stricter config checks.
    """

    # Empty strings are treated the same as missing values. This matters for
    # `.env` files where a student may leave `PGHOST=` blank while following the
    # runbook. Returning the default gives clearer downstream behavior.
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def require_env(name: str) -> str:
    """Read a required environment variable and raise a clear error if absent.

    Use this only for values that the code cannot safely continue without, such
    as Lakebase host/database/user. Optional values should use `get_env()`.
    """

    value = get_env(name)
    if value is None:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "See .env.example and TECHNICAL_RUNBOOK.md."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by retrieval, the agent, and the app.

    These are not secrets. They define where tables live and which model
    endpoints to call. Secrets such as tokens and Lakebase passwords are read
    separately from environment variables and are never stored in this dataclass.
    """

    catalog: str
    schema: str
    embedding_model: str
    llm_endpoint: str
    top_k: int
    rerank_k: int

    @property
    def silver_chunks_table(self) -> str:
        """Fully-qualified Delta table containing chunk text and metadata."""

        return f"{self.catalog}.{self.schema}.silver_document_chunks"

    @property
    def gold_product_daily_table(self) -> str:
        """Fully-qualified Delta table used by the agent's product metric tool."""

        return f"{self.catalog}.{self.schema}.gold_product_daily"

    @property
    def gold_issue_daily_table(self) -> str:
        """Fully-qualified Delta table for issue-level daily metrics."""

        return f"{self.catalog}.{self.schema}.gold_issue_daily"


def load_settings() -> Settings:
    """Build runtime settings from environment variables.

    Defaults are chosen so the project works naturally in Databricks Free
    Edition:
    - `adidas_retail_ai.productpulse` is the default Unity Catalog location.
    - `databricks-qwen3-embedding-0-6b` is the default embedding model.
    - `databricks-meta-llama-3-3-70b-instruct` is the default chat endpoint.

    Override these in `.env` for local runs or app environment variables for
    Databricks Apps.
    """

    return Settings(
        catalog=get_env("PRODUCTPULSE_CATALOG", "adidas_retail_ai"),
        schema=get_env("PRODUCTPULSE_SCHEMA", "productpulse"),
        embedding_model=get_env("PRODUCTPULSE_EMBEDDING_MODEL", "databricks-qwen3-embedding-0-6b"),
        llm_endpoint=get_env(
            "PRODUCTPULSE_LLM_ENDPOINT",
            "databricks-meta-llama-3-3-70b-instruct",
        ),
        top_k=int(get_env("PRODUCTPULSE_TOP_K", "8")),
        rerank_k=int(get_env("PRODUCTPULSE_RERANK_K", "4")),
    )


def lakebase_connection_info() -> dict[str, str]:
    """Return Lakebase/Postgres connection parameters.

    Databricks Apps inject PGHOST, PGPORT, PGDATABASE, PGUSER, and PGSSLMODE
    automatically when a Lakebase database resource is attached. For Lakebase
    Autoscaling, the app separately generates a short-lived password token using
    LAKEBASE_ENDPOINT in `lakebase.py`.

    Local development also works if the same values are placed in a `.env` file.
    """

    return {
        "host": require_env("PGHOST"),
        "port": get_env("PGPORT", "5432"),
        "database": require_env("PGDATABASE"),
        "user": require_env("PGUSER"),
        "sslmode": get_env("PGSSLMODE", "require"),
    }
