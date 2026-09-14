"""A compact conversational retail intelligence agent.

The agent has four tool families:
1. Memory: retrieve and save useful conversation facts in Lakebase.
2. RAG search: retrieve product reviews, tickets, manuals, FAQs, and policies.
3. Metrics SQL: query curated Delta gold tables for trend questions.
4. Final answer: synthesize an answer with citations and operational next steps.

The planner is LLM-driven but still guarded by readable Python validation. This
mirrors a production pattern:
- ask the model to choose tools
- require a strict JSON plan
- validate the plan before executing anything
- fall back to deterministic rules if the planner model fails
"""

from __future__ import annotations

import json
import re
from typing import Any

from databricks import sql

from productpulse.config import get_env, load_settings
from productpulse.databricks_models import chat_completion
from productpulse.lakebase import read_memory, write_memory
from productpulse.retrieval import retrieve_context


FINAL_ANSWER_SYSTEM_PROMPT = """You are ProductPulse AI, an adidas-inspired retail
product intelligence and support copilot. Answer like a senior retail analytics
partner. Use only the provided evidence and metrics. If evidence is thin, say
what is missing. Include concise citations like [1], [2]. End with practical
next actions for product, support, or operations teams."""


PLANNER_SYSTEM_PROMPT = """You are the tool planner for ProductPulse AI.
Return only one valid JSON object. Do not return markdown.

Available tools:
- read_memory: read compact conversation memory from Lakebase.
- rag_search: retrieve customer reviews, support tickets, and PDF chunks from Lakebase rag_chunks.
- query_metrics: query curated Gold Delta metrics through Databricks SQL Warehouse.
- write_memory: save a compact summary of the current normal agent turn.
- write_durable_memory: save an explicit user preference/instruction.

Allowed route values:
- sql_only: only structured aggregate metrics are needed.
- rag_memory: source evidence and conversation context are needed.
- hybrid: both source evidence and structured metrics are needed.
- memory_only: the user is only asking to save or recall memory.

Planning rules:
- Use sql_only for pure ranking/count/rate/total/trend questions.
- Use rag_memory when the user asks why, summarize, explain, evidence, policy, guidance, or what support should say.
- Use hybrid when the question needs both explanation/evidence and metrics.
- Use memory_only when the user asks to remember a preference or asks what you remember.
- If the user explicitly says SQL only, metrics only, skip RAG, or skip Lakebase, set read_memory=false, rag_search=false, write_memory=false, write_durable_memory=false, query_metrics=true.
- write_durable_memory should be true only when the user explicitly asks to remember, save, store, note, or keep a preference/instruction in mind.

JSON schema:
{
  "route": "sql_only | rag_memory | hybrid | memory_only",
  "read_memory": true,
  "rag_search": true,
  "query_metrics": true,
  "write_memory": true,
  "write_durable_memory": false,
  "reason": "short explanation"
}
"""


MEMORY_WRITE_PATTERNS = [
    r"^\s*remember\s+(that\s+)?",
    r"^\s*please\s+remember\s+(that\s+)?",
    r"^\s*keep\s+in\s+mind\s+(that\s+)?",
    r"^\s*note\s+that\s+",
    r"^\s*save\s+this\s*:?\s*",
    r"^\s*store\s+this\s*:?\s*",
    r"\bfrom now on\b",
    r"\bmy preference\s+is\b",
    r"\bi prefer\b",
    r"\bi care most about\b",
]


SQL_ONLY_PATTERNS = [
    r"^\s*sql\s+only\s*:?",
    r"^\s*metrics\s+only\s*:?",
    r"^\s*structured\s+metrics\s+only\s*:?",
    r"\buse\s+only\s+sql\b",
    r"\buse\s+only\s+structured\s+metrics\b",
    r"\bskip\s+rag\b",
    r"\bdo\s+not\s+use\s+rag\b",
    r"\bwithout\s+rag\b",
    r"\bskip\s+lakebase\b",
    r"\bdo\s+not\s+use\s+lakebase\b",
    r"\bwithout\s+lakebase\b",
]


METRIC_TERMS = [
    "highest",
    "lowest",
    "trend",
    "count",
    "rate",
    "compare",
    "last 7 days",
    "daily",
    "which product",
    "top",
    "rank",
    "total",
    "orders",
    "units sold",
    "tickets",
    "returns",
    "rating",
    "negative sentiment",
]


RAG_EXPLANATION_TERMS = [
    "why",
    "reason",
    "reasons",
    "explain",
    "summarize",
    "summary",
    "what should",
    "support agents",
    "say about",
    "source",
    "evidence",
    "policy",
    "manual",
    "faq",
    "complaining",
    "complaints about",
]


TOOL_PLAN_KEYS = [
    "read_memory",
    "rag_search",
    "query_metrics",
    "write_memory",
    "write_durable_memory",
]


def extract_product_id(question: str) -> str | None:
    """Find a product id inside the user question.

    The synthetic product ids use a clear retail SKU-like format, for example:
    - RUN-ULTRA-01
    - SOC-CLEAT-04

    If a product id is found, the RAG tool can filter Lakebase retrieval to that
    product. If no product id is found, retrieval searches across all products.
    """

    match = re.search(r"\b[A-Z]{2,6}-[A-Z0-9-]{2,}\b", question.upper())
    return match.group(0) if match else None


def should_write_memory(question: str) -> bool:
    """Return True when the user is explicitly asking for durable memory.

    This function is not the only memory write path. Normal turns are also
    saved as compact `conversation_turn` memory rows after the answer is
    generated. This function only detects stronger long-term preferences and
    instructions, such as "remember that I care about sizing issues" or
    "from now on, answer for support team leads".
    """

    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in MEMORY_WRITE_PATTERNS)


def asks_memory_recall(question: str) -> bool:
    """Return True when the user asks to inspect remembered context.

    This is separate from durable memory writes. Questions like "what do you
    remember about my preferences?" should read memory but should not run RAG or
    SQL metrics.
    """

    lowered = question.lower()
    return any(
        phrase in lowered
        for phrase in [
            "what do you remember",
            "what have you remembered",
            "my preferences",
            "remember about me",
            "saved preferences",
        ]
    )


def should_use_sql_only(question: str) -> bool:
    """Return True when a turn should use only structured metrics.

    Normal agent questions combine memory, RAG evidence, and optional metrics.
    For classroom demos it is useful to prove the SQL tool separately. Explicit
    phrases such as `SQL only:` always force that path. Pure aggregate questions
    such as "top products by ticket count" also use SQL-only automatically.
    """

    lowered = question.lower()
    if any(re.search(pattern, lowered) for pattern in SQL_ONLY_PATTERNS):
        return True

    has_metric_intent = any(term in lowered for term in METRIC_TERMS)
    needs_unstructured_evidence = any(term in lowered for term in RAG_EXPLANATION_TERMS)
    return has_metric_intent and not needs_unstructured_evidence


def deterministic_plan_tools(question: str) -> dict[str, Any]:
    """Create a safe fallback plan without calling an LLM.

    This is used when the LLM planner is unavailable, returns invalid JSON, or
    is explicitly disabled. It also documents the guardrail logic in plain
    Python so learners can compare LLM planning with deterministic behavior.
    """

    lowered = question.lower()
    sql_only = should_use_sql_only(question)
    wants_metrics = any(term in lowered for term in METRIC_TERMS)
    durable_memory = should_write_memory(question)
    memory_recall = asks_memory_recall(question)

    if sql_only:
        return {
            "route": "sql_only",
            "read_memory": False,
            "rag_search": False,
            "query_metrics": True,
            "write_memory": False,
            "write_durable_memory": False,
            "reason": "Fallback planner selected SQL-only for a pure metric question.",
            "planner_source": "deterministic_fallback",
        }

    if durable_memory:
        return {
            "route": "memory_only",
            "read_memory": True,
            "rag_search": False,
            "query_metrics": False,
            "write_memory": True,
            "write_durable_memory": True,
            "reason": "Fallback planner detected an explicit remember/save instruction.",
            "planner_source": "deterministic_fallback",
        }

    if memory_recall:
        return {
            "route": "memory_only",
            "read_memory": True,
            "rag_search": False,
            "query_metrics": False,
            "write_memory": True,
            "write_durable_memory": False,
            "reason": "Fallback planner detected a memory recall question.",
            "planner_source": "deterministic_fallback",
        }

    return {
        "route": "hybrid" if wants_metrics else "rag_memory",
        "read_memory": True,
        "rag_search": True,
        "query_metrics": wants_metrics,
        "write_memory": True,
        "write_durable_memory": False,
        "reason": "Fallback planner selected the normal RAG/memory path.",
        "planner_source": "deterministic_fallback",
    }


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from an LLM planner response.

    The planner prompt asks for JSON only, but some models may still wrap the
    response in explanatory text. This helper accepts either strict JSON or a
    response containing one JSON object.
    """

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def bool_value(value: Any) -> bool:
    """Normalize planner booleans from JSON-like values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def normalize_tool_plan(raw_plan: dict[str, Any], question: str) -> dict[str, Any]:
    """Validate and guardrail the LLM tool plan before execution.

    The LLM can suggest a route, but the code owns enforcement. This prevents a
    malformed or unsafe plan from accidentally touching Lakebase during an
    explicit SQL-only demo question, or from forgetting to write durable memory
    when the user says "remember".
    """

    route = str(raw_plan.get("route") or "rag_memory").strip().lower()
    allowed_routes = {"sql_only", "rag_memory", "hybrid", "memory_only"}
    if route not in allowed_routes:
        route = "rag_memory"

    plan: dict[str, Any] = {
        "route": route,
        "reason": str(raw_plan.get("reason") or "LLM planner selected tool route.").strip()[:500],
        "planner_source": str(raw_plan.get("planner_source") or "llm_planner"),
    }
    for key in TOOL_PLAN_KEYS:
        plan[key] = bool_value(raw_plan.get(key, False))

    explicit_sql_only = should_use_sql_only(question)
    durable_memory = should_write_memory(question)
    memory_recall = asks_memory_recall(question)

    if explicit_sql_only:
        plan.update(
            {
                "route": "sql_only",
                "read_memory": False,
                "rag_search": False,
                "query_metrics": True,
                "write_memory": False,
                "write_durable_memory": False,
                "reason": "Guardrail forced SQL-only because the user asked for metrics/SQL without RAG or Lakebase.",
            }
        )
        return plan

    if durable_memory:
        plan.update(
            {
                "route": "memory_only",
                "read_memory": True,
                "rag_search": False,
                "query_metrics": False,
                "write_memory": True,
                "write_durable_memory": True,
            }
        )
        return plan

    if memory_recall and not plan["rag_search"] and not plan["query_metrics"]:
        plan.update(
            {
                "route": "memory_only",
                "read_memory": True,
                "write_memory": True,
                "write_durable_memory": False,
            }
        )
        return plan

    # Route names and tool booleans should agree. The route is primarily used
    # for trace readability; the booleans are what actually execute tools.
    if plan["route"] == "sql_only":
        plan.update(
            {
                "read_memory": False,
                "rag_search": False,
                "query_metrics": True,
                "write_memory": False,
                "write_durable_memory": False,
            }
        )
    elif plan["route"] == "memory_only":
        plan.update(
            {
                "read_memory": True,
                "rag_search": False,
                "query_metrics": False,
                "write_memory": True,
            }
        )
    elif plan["route"] == "hybrid":
        plan.update(
            {
                "read_memory": True,
                "rag_search": True,
                "query_metrics": True,
                "write_memory": True,
            }
        )
    else:
        plan.update(
            {
                "read_memory": True,
                "rag_search": True,
                "query_metrics": False,
                "write_memory": True,
            }
        )

    return plan


def memory_text_from_question(question: str) -> str:
    """Convert a memory command into a compact memory fact.

    This keeps Lakebase memory readable when learners inspect the
    `conversation_memory` table. For example, "Remember that I prefer concise
    answers" becomes "I prefer concise answers".
    """

    cleaned = question.strip()
    cleaned = re.sub(r"(?i)^please\s+remember\s+that\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^remember\s+that\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^remember\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^keep\s+in\s+mind\s+that\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^note\s+that\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^save\s+this\s*:?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)^store\s+this\s*:?\s*", "", cleaned)
    return cleaned[:500]


def turn_summary_for_memory(question: str, answer: str) -> str:
    """Create a compact memory row for the completed chat turn.

    We intentionally do not store the full assistant answer here. A concise
    summary keeps the `conversation_memory` table readable for learners and
    prevents the next prompt from being filled with long historical answers.
    """

    compact_question = " ".join(question.strip().split())
    compact_answer = " ".join(answer.strip().split())
    return (
        f"User asked: {compact_question[:240]} | "
        f"Assistant answered: {compact_answer[:420]}"
    )


def plan_tools(question: str) -> dict[str, Any]:
    """Ask the LLM planner which tools should run for this turn.

    The function returns a validated tool plan, not raw model output. That is
    important because tool execution changes system behavior:
    - SQL-only questions should not touch Lakebase.
    - RAG questions should retrieve evidence before answering.
    - memory instructions should write durable memory.

    If the model call fails, returns invalid JSON, or is disabled with
    `PRODUCTPULSE_DISABLE_LLM_PLANNER=true`, the deterministic fallback planner
    is used so the app remains demoable.
    """

    if (get_env("PRODUCTPULSE_DISABLE_LLM_PLANNER") or "").lower() == "true":
        return deterministic_plan_tools(question)

    planner_prompt = f"""
User question:
{question}

Return the tool plan JSON now.
"""
    try:
        raw_response = chat_completion(
            PLANNER_SYSTEM_PROMPT,
            planner_prompt,
            temperature=0.0,
            max_tokens=260,
        )
        raw_plan = extract_json_object(raw_response)
        return normalize_tool_plan(raw_plan, question)
    except Exception as exc:
        plan = deterministic_plan_tools(question)
        plan["planner_source"] = "deterministic_fallback_after_llm_error"
        plan["planner_error"] = f"{type(exc).__name__}: {exc}"[:500]
        return plan


def execution_mode_from_plan(tool_plan: dict[str, Any]) -> str:
    """Name the execution mode for the app trace.

    This is mainly for teaching and troubleshooting. When learners ask a
    SQL-only question, the trace should plainly say `sql_only_metrics`, and
    retrieved evidence should be an empty list.
    """

    if tool_plan.get("route") == "sql_only":
        return "sql_only_metrics"
    if tool_plan.get("route") == "hybrid":
        return "hybrid_rag_sql_agent"
    if tool_plan.get("route") == "memory_only":
        return "memory_only_agent"
    return "memory_rag_agent"


def query_product_metrics(question: str, product_id: str | None = None) -> list[dict[str, Any]]:
    """Query curated gold tables through a Databricks SQL warehouse.

    The query is intentionally constrained. The LLM never writes arbitrary SQL;
    the agent picks one known-safe template and passes only product/date filters.
    """

    # The SQL Warehouse HTTP path is required because the Streamlit app and
    # notebooks query Delta tables through the Databricks SQL connector.
    http_path = get_env("DATABRICKS_SQL_HTTP_PATH")
    if not http_path:
        return [{"warning": "DATABRICKS_SQL_HTTP_PATH is not configured, metrics skipped."}]

    settings = load_settings()

    # Local execution can use DATABRICKS_HOST and DATABRICKS_TOKEN from `.env`.
    # Databricks Apps should use the Databricks SDK credential provider injected
    # into the app runtime, so a personal access token is not required there.
    host = (get_env("DATABRICKS_HOST") or "").replace("https://", "")
    access_token = get_env("DATABRICKS_TOKEN")
    credentials_provider = None
    try:
        from databricks.sdk.core import Config

        cfg = Config()
        if not host:
            host = cfg.host.replace("https://", "")
        if not access_token:
            # Databricks SQL Connector expects a callable. In Databricks Apps,
            # Config().authenticate resolves to OAuth headers for the app
            # service principal, so wrap it in a lambda instead of passing the
            # dictionary directly.
            credentials_provider = lambda: cfg.authenticate
    except Exception:
        if not host:
            return [{"warning": "DATABRICKS_HOST is not configured, metrics skipped."}]

    table = settings.gold_product_daily_table

    lowered = question.lower()
    if "ticket" in lowered:
        order_column = "total_ticket_count"
    elif "return rate" in lowered or ("return" in lowered and "rate" in lowered):
        order_column = "computed_return_rate"
    elif "return" in lowered:
        order_column = "total_return_count"
    elif "rating" in lowered:
        order_column = "avg_rating"
    elif "sold" in lowered or "sales" in lowered or "units" in lowered:
        order_column = "total_units_sold"
    elif "review" in lowered:
        order_column = "total_review_count"
    else:
        order_column = "total_negative_sentiment_count"

    # The LLM never gets to generate SQL. This is a core production practice:
    # the agent selects a safe, predefined query template and only supplies a
    # product_id parameter when available.
    where = "WHERE product_id = %(product_id)s" if product_id else ""
    params = {"product_id": product_id} if product_id else {}

    query = f"""
    /* ProductPulse metric tool: governed aggregate over gold_product_daily. */
    SELECT
      product_id,
      product_name,
      category,
      MIN(event_date) AS first_event_date,
      MAX(event_date) AS latest_event_date,
      SUM(review_count) AS total_review_count,
      SUM(ticket_count) AS total_ticket_count,
      SUM(order_count) AS total_order_count,
      SUM(units_sold) AS total_units_sold,
      SUM(return_count) AS total_return_count,
      ROUND(SUM(return_count) / NULLIF(SUM(order_count), 0), 4) AS computed_return_rate,
      ROUND(AVG(return_rate), 4) AS avg_daily_return_rate,
      ROUND(AVG(avg_rating), 2) AS avg_rating,
      SUM(negative_sentiment_count) AS total_negative_sentiment_count,
      SUM(return_related_count) AS total_return_related_count,
      MIN(on_hand_units) AS min_on_hand_units,
      MAX(stock_status) AS latest_stock_status,
      ARRAY_JOIN(ARRAY_SORT(ARRAY_DISTINCT(COLLECT_LIST(top_issue))), ', ') AS observed_top_issues
    FROM {table}
    {where}
    GROUP BY product_id, product_name, category
    ORDER BY {order_column} DESC, product_id
    LIMIT 20
    """

    connection_args = {"server_hostname": host, "http_path": http_path}
    if access_token:
        connection_args["access_token"] = access_token
    elif credentials_provider:
        connection_args["credentials_provider"] = credentials_provider
    else:
        return [{"warning": "Databricks SQL credentials are not configured, metrics skipped."}]

    try:
        with sql.connect(**connection_args) as conn:
            with conn.cursor() as cur:
                # The connector handles parameter substitution for `product_id`.
                # This keeps the query safe and repeatable.
                cur.execute(query, params)
                columns = [description[0] for description in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
    except Exception as exc:
        return [
            {
                "warning": (
                    "Metric SQL query was skipped. Confirm the Databricks App "
                    "service principal has CAN USE on the SQL warehouse, USE "
                    f"CATALOG on {settings.catalog}, USE SCHEMA on "
                    f"{settings.catalog}.{settings.schema}, and SELECT on the "
                    f"gold tables. Original error: {exc}"
                )
            }
        ]


def answer_question(
    conn,
    session_id: str,
    question: str,
    tool_plan: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Run one complete agent turn.

    Input:
    - `conn`: an open Lakebase/Postgres connection for normal turns. It can be
      `None` for SQL-only turns because those do not read Lakebase memory or
      Lakebase vectors.
    - `session_id`: a stable id for the user's chat session. This groups
      conversation memory by user session.
    - `question`: the business question typed in the UI or notebook.
    - `tool_plan`: optional precomputed plan. The Streamlit app passes this so
      the LLM planner runs once, then the same validated plan is executed.

    Output:
    - A dictionary containing the final answer plus an agent trace. The
      Streamlit app shows this trace in an expander so learners can see which
      tools ran and which evidence was retrieved.
    """

    # Step 1: decide which tools should run for this question. The app may have
    # already planned the turn so it knows whether it needs a Lakebase
    # connection. If so, reuse that exact validated plan.
    tool_plan = tool_plan or plan_tools(question)
    execution_mode = execution_mode_from_plan(tool_plan)

    # Step 2: extract product id, if present. This improves retrieval precision
    # for product-specific questions.
    product_id = extract_product_id(question)

    needs_lakebase = (
        tool_plan["read_memory"] or tool_plan["rag_search"] or tool_plan["write_memory"]
    )
    if needs_lakebase and conn is None:
        raise RuntimeError(
            "This question needs Lakebase for memory or RAG retrieval, but no "
            "Lakebase connection was provided."
        )

    # Step 3: pull recent conversation memory. Memory is not used as ground
    # truth; it only provides conversational context like user preferences.
    memory_warnings = []
    try:
        memories = read_memory(conn, session_id=session_id) if tool_plan["read_memory"] else []
    except Exception as exc:
        memories = []
        memory_warnings.append(
            "Memory read was skipped because the app identity cannot read "
            f"conversation_memory: {exc}"
        )

    # Step 3b: save explicit durable memory before generating the answer. This
    # makes the current turn able to acknowledge a newly stated preference. The
    # regular turn summary is saved later, after the final answer exists.
    memory_write_results = []
    if tool_plan["write_durable_memory"]:
        memory_text_to_save = memory_text_from_question(question)
        try:
            write_memory(
                conn,
                session_id=session_id,
                memory_type="user_preference",
                memory_text=memory_text_to_save,
                metadata={"source": "agent_turn", "raw_question": question[:500]},
            )
            memory_write_results.append("saved:user_preference")
            memories.insert(
                0,
                {
                    "memory_type": "user_preference",
                    "memory_text": memory_text_to_save,
                    "metadata": {"source": "current_turn"},
                },
            )
        except Exception as exc:
            memory_write_results.append("failed:user_preference")
            memory_warnings.append(
                "Memory write was skipped because the app identity cannot write "
                f"conversation_memory: {exc}"
            )

    # Step 4: run the RAG retrieval chain. This returns source evidence that the
    # final LLM is allowed to cite.
    if tool_plan["rag_search"]:
        retrieval = retrieve_context(
            conn,
            question=question,
            product_id=product_id,
        )
    else:
        retrieval = {
            "rewritten_query": "RAG skipped because this was a SQL-only turn.",
            "evidence": [],
            "context": "",
        }

    # Step 5: optionally query structured metrics from the gold Delta table.
    # This gives the final answer operational numbers in addition to text
    # evidence.
    metrics = query_product_metrics(question, product_id) if tool_plan["query_metrics"] else []

    # Step 6: format memory and metrics into compact text for the final prompt.
    # The evidence context is already formatted in retrieval.py.
    memory_text = "\n".join(f"- {row['memory_type']}: {row['memory_text']}" for row in memories)
    metrics_text = "\n".join(str(row) for row in metrics)
    user_prompt = f"""
User question:
{question}

Recent memory:
{memory_text or "No prior memory for this session."}

Structured metrics:
{metrics_text or "No metric tool was needed for this question."}

Retrieved evidence:
{retrieval["context"] or "No evidence was retrieved."}
"""

    answer = chat_completion(
        FINAL_ANSWER_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.1,
        max_tokens=900,
    )

    # Step 7: save a compact record of the full turn. This is the realistic
    # "the agent remembers the conversation" behavior: every normal RAG/agent
    # turn leaves a small memory row, while SQL-only demo turns intentionally do
    # not touch Lakebase.
    if tool_plan["write_memory"]:
        try:
            write_memory(
                conn,
                session_id=session_id,
                memory_type="conversation_turn",
                memory_text=turn_summary_for_memory(question, answer),
                metadata={
                    "source": "agent_turn",
                    "product_id": product_id,
                    "query_metrics": tool_plan["query_metrics"],
                    "rag_search": tool_plan["rag_search"],
                },
            )
            memory_write_results.append("saved:conversation_turn")
        except Exception as exc:
            memory_write_results.append("failed:conversation_turn")
            memory_warnings.append(
                "Turn memory write was skipped because the app identity cannot "
                f"write conversation_memory: {exc}"
            )

    return {
        "answer": answer,
        "execution_mode": execution_mode,
        "lakebase_used": needs_lakebase,
        "tool_plan": tool_plan,
        "product_id": product_id,
        "rewritten_query": retrieval["rewritten_query"],
        "evidence": retrieval["evidence"],
        "metrics": metrics,
        "memory_write_result": memory_write_results or ["not_requested"],
        "memory_seen": memories,
        "memory_warnings": memory_warnings,
    }
