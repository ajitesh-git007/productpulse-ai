"""Streamlit interface for ProductPulse AI.

This app is designed for Databricks Apps. It can also run locally after the
environment variables in .env.example are filled in.
"""

from __future__ import annotations

import sys
import uuid
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


def find_project_root() -> Path:
    """Find the deployed project root in local and Databricks App runtimes.

    Locally, this file lives at `<project>/app/app.py`. In Databricks Apps, the
    source is copied under a runtime path such as `/app/python/source_code`.

    The project intentionally keeps only one backend package:
    `<project>/src/productpulse`. Therefore the Databricks App must be deployed
    from the bundle/project root, not from the `app/` folder alone.
    """

    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "src" / "productpulse").exists():
            return candidate
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = find_project_root()

# Add the shared package location to Python's import path:
# - `<project>/src/productpulse` contains the only backend package copy.
# - `<project>` is also added because some Databricks runtime helpers resolve
#   files relative to the deployed source root.
for import_root in [PROJECT_ROOT / "src", PROJECT_ROOT]:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

# Local runs can use a `.env` file copied from `.env.example`. In Databricks
# Apps, environment variables are injected by app configuration and Lakebase
# resources, so this line simply does nothing if `.env` is absent.
load_dotenv(PROJECT_ROOT / ".env")

try:
    from productpulse.agent import answer_question, plan_tools
    from productpulse.config import get_env, load_settings
    from productpulse.lakebase import connect
except ModuleNotFoundError as exc:
    st.error(
        "ProductPulse package import failed. Confirm the Databricks App source "
        "folder is the bundle root that contains app.yml, app/, src/, and "
        "requirements.txt. Do not deploy from the app/ folder alone."
    )
    st.write("Project root detected", str(PROJECT_ROOT))
    st.write("src/productpulse exists", (PROJECT_ROOT / "src" / "productpulse").exists())
    st.write("Python path", sys.path)
    raise exc


st.set_page_config(
    page_title="ProductPulse AI",
    page_icon="PP",
    layout="wide",
)


def inject_styles() -> None:
    """Apply a polished internal-tool visual style to the Streamlit app.

    The design is closer to an internal analyst cockpit than a default chat
    page: strong command bar, dense panels, clear route cards, and a controlled
    prompt surface.
    """

    st.markdown(
        """
        <style>
        :root {
            --pp-ink: #191d29;
            --pp-muted: #667085;
            --pp-soft: #8a93a5;
            --pp-line: #d8dee8;
            --pp-bg: #eef2f6;
            --pp-panel: #ffffff;
            --pp-panel-2: #f8fafc;
            --pp-red: #ff4f5e;
            --pp-yellow: #f5b544;
            --pp-green: #159a74;
            --pp-blue: #2d60b7;
            --pp-cyan: #1f9bb4;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(45, 96, 183, 0.10), transparent 28rem),
                linear-gradient(180deg, #f7f9fc 0%, #edf2f7 100%);
            color: var(--pp-ink);
        }

        .block-container {
            max-width: 1760px;
            padding: 1.15rem 2rem 2rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0;
            position: fixed;
        }

        .pp-command {
            background:
                linear-gradient(135deg, #191d29 0%, #252b3d 58%, #31446b 100%);
            border: 1px solid rgba(25, 29, 41, 0.2);
            border-radius: 8px;
            box-shadow: 0 24px 70px rgba(25, 29, 41, 0.18);
            overflow: hidden;
            margin-bottom: 1.1rem;
        }

        .pp-command-strip {
            height: 5px;
            background: linear-gradient(
                90deg,
                var(--pp-red) 0 20%,
                var(--pp-yellow) 20% 40%,
                var(--pp-green) 40% 60%,
                var(--pp-cyan) 60% 80%,
                var(--pp-blue) 80% 100%
            );
        }

        .pp-command-inner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.25rem;
            padding: 1.15rem 1.35rem;
        }

        .pp-brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .pp-logo {
            width: 52px;
            height: 52px;
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.24);
            color: white;
            display: grid;
            place-items: center;
            font-weight: 800;
            letter-spacing: 0;
            color: var(--pp-ink);
            box-shadow: inset 0 -3px 0 rgba(255, 79, 94, 0.85);
        }

        .pp-title {
            font-size: 2.05rem;
            line-height: 1.05;
            font-weight: 800;
            margin: 0;
            color: #ffffff;
        }

        .pp-subtitle {
            margin-top: 0.35rem;
            color: rgba(255, 255, 255, 0.72);
            font-size: 0.9rem;
        }

        .pp-status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            justify-content: flex-end;
            padding-top: 0.15rem;
        }

        .pp-pill {
            border: 1px solid var(--pp-line);
            background: #ffffff;
            border-radius: 999px;
            padding: 0.42rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .pp-pill-ok {
            border-color: rgba(31, 157, 120, 0.28);
            color: #146b52;
            background: #effaf6;
        }

        .pp-pill-warn {
            border-color: rgba(255, 181, 71, 0.45);
            color: #8a5a00;
            background: #fff8e9;
        }

        .pp-panel {
            border: 1px solid var(--pp-line);
            background: var(--pp-panel);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .pp-panel-title {
            color: var(--pp-ink);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 800;
            margin-bottom: 0.75rem;
        }

        .pp-workspace {
            border: 1px solid var(--pp-line);
            background: rgba(255, 255, 255, 0.92);
            border-radius: 8px;
            box-shadow: 0 18px 60px rgba(25, 29, 41, 0.08);
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .pp-section-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.8rem;
        }

        .pp-section-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--pp-ink);
            margin: 0;
        }

        .pp-section-subtitle {
            color: var(--pp-muted);
            font-size: 0.82rem;
            margin-top: 0.16rem;
        }

        .pp-ask-card {
            border: 1px solid #dce2eb;
            border-radius: 8px;
            background:
                linear-gradient(180deg, #ffffff 0%, #f9fbfd 100%);
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .pp-route-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .pp-route-card {
            position: relative;
            border: 1px solid #dbe2ec;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.85rem;
            min-height: 98px;
            overflow: hidden;
        }

        .pp-route-card::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 4px;
            height: 100%;
            background: var(--route-color);
        }

        .pp-route-label {
            color: var(--pp-soft);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .pp-route-value {
            color: var(--pp-ink);
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .pp-route-copy {
            color: var(--pp-muted);
            font-size: 0.78rem;
            line-height: 1.35;
        }

        .pp-kpi-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
        }

        .pp-kpi {
            border: 1px solid #e4e8ef;
            border-radius: 8px;
            padding: 0.75rem;
            background: #fbfcfe;
            min-height: 74px;
        }

        .pp-kpi-label {
            color: var(--pp-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .pp-kpi-value {
            color: var(--pp-ink);
            font-size: 0.95rem;
            font-weight: 750;
            overflow-wrap: anywhere;
        }

        .pp-runtime {
            border-collapse: collapse;
            width: 100%;
            font-size: 0.82rem;
        }

        .pp-runtime td {
            padding: 0.46rem 0;
            border-bottom: 1px solid #edf0f5;
            vertical-align: top;
        }

        .pp-runtime td:first-child {
            color: var(--pp-muted);
            width: 38%;
            font-weight: 700;
        }

        .pp-runtime td:last-child {
            color: var(--pp-ink);
            overflow-wrap: anywhere;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.78rem;
        }

        div[data-testid="stButton"] > button {
            border-radius: 8px;
            border: 1px solid #d8dee8;
            background: #ffffff;
            color: var(--pp-ink);
            min-height: 42px;
            font-weight: 700;
            text-align: left;
            justify-content: flex-start;
            padding: 0.6rem 0.75rem;
            white-space: normal;
            box-shadow: 0 1px 0 rgba(31, 36, 48, 0.03);
        }

        div[data-testid="stButton"] > button:hover {
            border-color: var(--pp-blue);
            color: var(--pp-ink);
            background: #f5f8ff;
        }

        div[data-testid="stForm"] {
            border: 0;
            padding: 0;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 8px;
            border: 1px solid #cfd7e3;
            min-height: 48px;
            font-weight: 650;
            background: #ffffff;
        }

        div[data-testid="stFormSubmitButton"] button {
            min-height: 48px;
            justify-content: center;
            background: var(--pp-ink);
            border-color: var(--pp-ink);
            color: #ffffff;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: #2c3348;
            color: #ffffff;
            border-color: #2c3348;
        }

        .stChatMessage {
            border: 1px solid #e3e7ef;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            box-shadow: 0 1px 0 rgba(31, 36, 48, 0.03);
        }

        div[data-testid="stChatMessageContent"] {
            color: var(--pp-ink);
        }

        div[data-testid="stChatInput"] {
            border-top: 1px solid var(--pp-line);
            background: rgba(245, 247, 250, 0.92);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--pp-line);
            border-radius: 8px;
            background: #fbfcfe;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid #f0d690;
        }

        .pp-empty-state {
            border: 1px dashed #c8d0dc;
            border-radius: 8px;
            padding: 1.1rem;
            background:
                linear-gradient(135deg, #fbfcfe 0%, #f2f6fb 100%);
            color: var(--pp-muted);
            margin-bottom: 1rem;
        }

        .pp-empty-title {
            color: var(--pp-ink);
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .pp-mini-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin-bottom: 1rem;
        }

        .pp-mini {
            background: #ffffff;
            border: 1px solid #dbe2ec;
            border-radius: 8px;
            padding: 0.8rem;
            min-height: 78px;
        }

        .pp-mini-num {
            font-weight: 850;
            color: var(--pp-ink);
            font-size: 1.05rem;
        }

        .pp-mini-label {
            margin-top: 0.25rem;
            color: var(--pp-muted);
            font-size: 0.76rem;
            line-height: 1.3;
        }

        .pp-side-note {
            font-size: 0.78rem;
            color: var(--pp-muted);
            line-height: 1.45;
            padding: 0.65rem 0.75rem;
            border: 1px solid #e3e8f0;
            background: #fbfcfe;
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .pp-header {
                flex-direction: column;
            }
            .pp-status-row {
                justify-content: flex-start;
            }
            .pp-kpi-grid {
                grid-template-columns: 1fr;
            }
            .pp-command-inner {
                flex-direction: column;
                align-items: flex-start;
            }
            .pp-route-grid,
            .pp-mini-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(settings, missing: list[str]) -> None:
    """Render the top command bar."""

    config_state = "Ready" if not missing else "Needs config"
    config_class = "pp-pill-ok" if not missing else "pp-pill-warn"
    sql_state = "SQL ready" if get_env("DATABRICKS_SQL_HTTP_PATH") else "SQL path missing"
    s
    











    
    lakebase_state = "Lakebase attached" if get_env("PGHOST") else "Lakebase pending"
    lakebase_class = "pp-pill-ok" if get_env("PGHOST") else "pp-pill-warn"
    catalog_label = escape(f"{settings.catalog}.{settings.schema}")

    st.markdown(
        f"""
        <div class="pp-command">
          <div class="pp-command-strip"></div>
          <div class="pp-command-inner">
            <div class="pp-brand">
              <div class="pp-logo">PP</div>
              <div>
                <h1 class="pp-title">ProductPulse AI</h1>
                <div class="pp-subtitle">Retail intelligence cockpit for product, support, returns, and quality teams</div>
              </div>
            </div>
            <div class="pp-status-row">
              <span class="pp-pill {config_class}">{config_state}</span>
              <span class="pp-pill {lakebase_class}">{lakebase_state}</span>
              <span class="pp-pill {sql_class}">{sql_state}</span>
              <span class="pp-pill">{catalog_label}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_route_cards() -> None:
    """Show the three agent routes as first-class UI signals."""

    st.markdown(
        """
        <div class="pp-route-grid">
          <div class="pp-route-card" style="--route-color:#ff4f5e;">
            <div class="pp-route-label">RAG route</div>
            <div class="pp-route-value">Evidence answers</div>
            <div class="pp-route-copy">Reviews, tickets, manuals, SOPs, policies, citations.</div>
          </div>
          <div class="pp-route-card" style="--route-color:#2d60b7;">
            <div class="pp-route-label">SQL route</div>
            <div class="pp-route-value">Metric answers</div>
            <div class="pp-route-copy">Gold Delta aggregates through the SQL Warehouse.</div>
          </div>
          <div class="pp-route-card" style="--route-color:#159a74;">
            <div class="pp-route-label">Memory route</div>
            <div class="pp-route-value">Context carryover</div>
            <div class="pp-route-copy">Compact turn memory and durable user preferences.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_strip() -> None:
    """Render compact dataset signals for the empty/start state."""

    st.markdown(
        """
        <div class="pp-mini-grid">
          <div class="pp-mini">
            <div class="pp-mini-num">2,331</div>
            <div class="pp-mini-label">customer reviews</div>
          </div>
          <div class="pp-mini">
            <div class="pp-mini-num">720</div>
            <div class="pp-mini-label">support tickets</div>
          </div>
          <div class="pp-mini">
            <div class="pp-mini-num">925</div>
            <div class="pp-mini-label">return events</div>
          </div>
          <div class="pp-mini">
            <div class="pp-mini-num">100</div>
            <div class="pp-mini-label">PDF pages indexed</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prompt_form() -> str | None:
    """Render the primary prompt form and return submitted text."""

    st.markdown(
        """
        <div class="pp-section-head">
          <div>
            <div class="pp-section-title">Ask ProductPulse</div>
            <div class="pp-section-subtitle">Questions are routed to RAG, SQL metrics, memory, or a combined agent turn.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("ask-productpulse", clear_on_submit=True):
        prompt = st.text_input(
            "Question",
            label_visibility="collapsed",
            placeholder="Ask about returns, product issues, ticket counts, sentiment, support guidance, or memory...",
        )
        submitted = st.form_submit_button("Run analysis", use_container_width=True)
    return prompt.strip() if submitted and prompt.strip() else None


def render_runtime_panel(settings) -> None:
    """Show compact runtime values for debugging deployment issues."""

    rows = {
        "Catalog": settings.catalog,
        "Schema": settings.schema,
        "Embedding": settings.embedding_model,
        "LLM": settings.llm_endpoint,
        "Retrieval": f"top_k={settings.top_k}, rerank_k={settings.rerank_k}",
        "Lakebase user": get_env("PGUSER") or "not injected",
    }
    html_rows = "\n".join(
        f"<tr><td>{escape(label)}</td><td>{escape(str(value))}</td></tr>"
        for label, value in rows.items()
    )
    st.markdown(
        f"""
        <div class="pp-panel">
          <div class="pp-panel-title">Runtime</div>
          <table class="pp-runtime">{html_rows}</table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_health_panel(missing: list[str]) -> None:
    """Show setup health without dumping stack traces into the main area."""

    if missing:
        status = "Configuration needed"
        detail = escape(", ".join(missing))
        css_class = "pp-pill-warn"
    else:
        status = "Ready"
        detail = "RAG, memory, and metric routing are configured"
        css_class = "pp-pill-ok"

    st.markdown(
        f"""
        <div class="pp-panel">
          <div class="pp-panel-title">Status</div>
          <div class="pp-kpi-grid">
            <div class="pp-kpi">
              <div class="pp-kpi-label">App</div>
              <div class="pp-kpi-value"><span class="pp-pill {css_class}">{status}</span></div>
            </div>
            <div class="pp-kpi">
              <div class="pp-kpi-label">Mode</div>
              <div class="pp-kpi-value">RAG + SQL + Memory</div>
            </div>
          </div>
          <div style="margin-top:0.75rem;color:#667085;font-size:0.82rem;overflow-wrap:anywhere;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """Render a clean first-run state above the chat input."""

    st.markdown(
        """
        <div class="pp-empty-state">
          <div class="pp-empty-title">Conversation is ready.</div>
          <div>Start with a RAG investigation, a SQL metric ranking, or a memory check from the prompt cards.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_starter_questions() -> None:
    """Render starter prompts grouped by expected routing behavior."""

    st.markdown('<div class="pp-panel-title">Prompt Library</div>', unsafe_allow_html=True)
    tabs = st.tabs(["RAG", "Metrics", "Memory"])

    question_groups = {
        "RAG": [
            "Why are customers returning RUN-ULTRA-01?",
            "Summarize negative sentiment for SOC-CLEAT-04.",
            "What should support agents say about narrow toe box complaints?",
        ],
        "Metrics": [
            "What are the top products by ticket count?",
            "Which product has the highest return rate?",
            "Rank products by negative sentiment count.",
        ],
        "Memory": [
            "Remember that I care most about sizing and fit issues for running shoes.",
            "For RUN-ULTRA-01, what should I look at first?",
            "What do you remember about my preferences?",
        ],
    }

    for tab, label in zip(tabs, ["RAG", "Metrics", "Memory"]):
        with tab:
            for starter in question_groups[label]:
                if st.button(starter, use_container_width=True, key=f"starter-{label}-{starter}"):
                    # Button clicks and chat input are handled through the same
                    # prompt path so the rest of the app logic stays simple.
                    st.session_state.pending_question = starter


def render_trace(result: dict[str, object]) -> None:
    """Show the agent trace in compact tabs.

    This trace is intentionally visible in the demo app. It lets learners
    verify whether a question used SQL-only routing, RAG retrieval, memory, or
    a hybrid path without reading logs.
    """

    with st.expander("Agent trace", expanded=False):
        overview, metrics_tab, memory_tab, evidence_tab = st.tabs(
            ["Route", "Metrics", "Memory", "Evidence"]
        )
        with overview:
            st.write("Execution mode", result.get("execution_mode"))
            st.write("Lakebase used", result.get("lakebase_used"))
            st.write("Tool plan", result["tool_plan"])
            st.write("Rewritten query", result["rewritten_query"])
        with metrics_tab:
            st.write(result["metrics"])
        with memory_tab:
            st.write("Memory write result", result.get("memory_write_result"))
            st.write("Memory seen", result.get("memory_seen"))
            if result.get("memory_warnings"):
                st.write("Memory warnings", result["memory_warnings"])
        with evidence_tab:
            st.write(result["evidence"])


def missing_config() -> list[str]:
    """List configuration values required before the app can answer questions.

    Lakebase values are needed for both vector retrieval and memory. The SQL
    Warehouse HTTP path is needed only for metric questions, but showing it here
    makes setup issues visible immediately in the UI.
    """

    required = ["PGHOST", "PGDATABASE", "PGUSER"]
    missing = [name for name in required if not get_env(name)]
    if not get_env("PGPASSWORD") and not get_env("LAKEBASE_ENDPOINT"):
        missing.append("PGPASSWORD or LAKEBASE_ENDPOINT")
    if not get_env("DATABRICKS_SQL_HTTP_PATH"):
        missing.append("DATABRICKS_SQL_HTTP_PATH")
    return missing


@st.cache_resource(show_spinner=False)
def ensure_lakebase_ready() -> bool:
    """Verify Lakebase connectivity without requiring schema DDL permissions.

    The daily pipeline owns schema creation and data loading. The app should be
    least-privilege: it needs to read `rag_chunks` and read/write
    `conversation_memory`, but it does not need permission to create extensions
    or tables in the `public` schema.
    """

    with connect() as conn:
        cur = conn.cursor()
        try:
            cur.execute("SELECT 1;")
            cur.fetchone()
        finally:
            cur.close()
    return True


if "session_id" not in st.session_state:
    # A session id keeps memory scoped to one browser conversation. Without it,
    # multiple users could accidentally share memory rows.
    st.session_state.session_id = f"streamlit-{uuid.uuid4()}"

if "messages" not in st.session_state:
    # Streamlit reruns the script on every interaction. session_state preserves
    # chat history across those reruns.
    st.session_state.messages = []


settings = load_settings()
missing = missing_config()
inject_styles()
render_header(settings, missing)


# Main screen layout:
# - left column: analyst workspace, route cards, prompt form, chat transcript
# - right column: prompt library, health, runtime values, reset action
left, right = st.columns([0.72, 0.28], gap="large")

with right:
    with st.container(border=True):
        render_starter_questions()
    render_health_panel(missing)
    render_runtime_panel(settings)
    st.markdown(
        """
        <div class="pp-side-note">
          SQL-only aggregate questions should show <strong>Lakebase used = false</strong>.
          RAG questions should show retrieved evidence and citations.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start new chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = f"streamlit-{uuid.uuid4()}"
        st.rerun()


with left:
    with st.container(border=True):
        # Route cards teach the three possible agent paths before the user asks
        # anything. This makes the app useful as a bootcamp demonstration tool.
        render_route_cards()
        render_dataset_strip()
        submitted_prompt = render_prompt_form()

    if submitted_prompt:
        st.session_state.pending_question = submitted_prompt

    if not st.session_state.messages:
        render_empty_state()

    for message in st.session_state.messages:
        # Re-render previous turns after every Streamlit rerun. Without this,
        # clicking any button would clear the visible chat transcript.
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = None
    if "pending_question" in st.session_state:
        # Starter questions are stored briefly in session_state because
        # Streamlit buttons and forms use the same downstream prompt path.
        prompt = st.session_state.pop("pending_question")

    if prompt:
        # Store and render the user turn first so the UI feels like a normal
        # chat app while the agent work is running.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking through product evidence and metrics..."):
                try:
                    # Ask the LLM planner once in the app. We need the plan
                    # before opening Lakebase so SQL-only turns can skip
                    # Lakebase completely. The same plan is passed into
                    # answer_question() to avoid a second planner call.
                    turn_plan = plan_tools(prompt)
                    needs_lakebase = (
                        turn_plan["read_memory"]
                        or turn_plan["rag_search"]
                        or turn_plan["write_memory"]
                    )
                    if needs_lakebase:
                        # Normal RAG/hybrid turns use one Lakebase connection
                        # for memory read, vector search, and memory writes.
                        with connect() as conn:
                            result = answer_question(
                                conn=conn,
                                session_id=st.session_state.session_id,
                                question=prompt,
                                tool_plan=turn_plan,
                            )
                    else:
                        # SQL-only turns intentionally avoid Lakebase. This is
                        # important for demonstrating the pure SQL Warehouse
                        # path and for validating `Lakebase used = false`.
                        result = answer_question(
                            conn=None,
                            session_id=st.session_state.session_id,
                            question=prompt,
                            tool_plan=turn_plan,
                        )
                    st.markdown(result["answer"])

                    render_trace(result)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"]}
                    )
                except Exception as exc:
                    error = f"Unable to complete the agent turn: {exc}"
                    st.error(error)
                    st.session_state.messages.append({"role": "assistant", "content": error})
