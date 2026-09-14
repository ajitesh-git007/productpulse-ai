# ProductPulse AI Technical Runbook

This runbook explains how to set up, run, validate, and troubleshoot ProductPulse AI on Databricks Free Edition.

ProductPulse AI is an adidas-inspired retail product intelligence and support copilot. It combines Databricks Workflows, Unity Catalog, Delta tables, Foundation Model APIs, Lakebase Postgres with pgvector, and Databricks Apps.

## 1. Final Architecture

The final project uses these components:

| Component | Purpose |
| --- | --- |
| Unity Catalog | Governance boundary for catalog, schema, volume, and Delta tables |
| UC volume | Landing location for raw CSV, JSONL, and PDF files |
| Bronze Delta tables | Cleanly loaded raw structured feeds and PDF text |
| Silver Delta tables | RAG-ready chunks and governed embedding copy |
| Gold Delta tables | Product-level and issue-level business metrics |
| Databricks Foundation Model APIs | Embeddings and final LLM answers |
| Lakebase Postgres | pgvector retrieval store and agent memory store |
| Databricks SQL Warehouse | Structured metric queries from gold Delta tables |
| Databricks Apps | Streamlit chat interface for business users |
| Databricks Asset Bundle | Deploys notebooks and jobs to the workspace |

Important design decisions:

- Lakebase is the vector store and memory store.
- AI Search is not used in this project.
- Unity Catalog setup is separated into its own manual job.
- The daily pipeline is a separate workflow that can be run manually or scheduled.
- SQL-only metric questions can skip Lakebase completely.
- Normal agent questions use memory, RAG retrieval, optional SQL metrics, and final answer generation.

## 2. Project Structure And File Guide

This section explains the codebase file by file. Use it as the first place to
orient learners before they run the project.

### 2.1 Root Files

| File | Used by | Purpose |
| --- | --- | --- |
| `README.md` | Humans | Business-facing overview of the problem statement, architecture, dataset, and demo questions. Start here when explaining why the project exists. |
| `TECHNICAL_RUNBOOK.md` | Humans | End-to-end setup, execution, validation, troubleshooting, and codebase guide. This is the primary instructor guide. |
| `databricks.yml` | Databricks Asset Bundles | Defines the deployable Databricks resources: the manual Unity Catalog setup job and the daily product intelligence pipeline job. |
| `app.yml` | Databricks Apps | Defines how the Streamlit app starts, which environment variables are injected, and which Lakebase app resource is attached. |
| `.env.example` | Local development only | Template for local Streamlit runs. Databricks Jobs and Databricks Apps do not read this file automatically. Never put real secrets in this template. |
| `requirements.txt` | Databricks Apps and local Python | Runtime Python dependencies for the Streamlit app and shared backend modules. Databricks Apps installs these when deploying the app source folder. |
| `pyproject.toml` | Python packaging tools | Project metadata for local editable installs or future wheel packaging. The current bundle deployment does not require a wheel file. |
| `.gitignore` | Git | Prevents local-only files, virtual environments, caches, and generated clutter from being committed. |
| `.databricks/.gitignore` | Databricks CLI | Keeps local Databricks bundle state out of source control. |

Important note about app specs:

```text
Keep app.yml only.
```

Databricks Apps accepts the app spec from the deployed source folder. Keeping
one app spec avoids confusion about which command and environment values are
active.

### 2.2 Databricks Bundle Definition

| File | What it contains | Why it matters |
| --- | --- | --- |
| `databricks.yml` | Bundle name, catalog/schema variables, job definitions, serverless environment dependencies, notebook task order, job parameters, and a paused daily schedule. | Running `databricks bundle deploy --force-lock` reads this file and creates/updates the Databricks jobs in the workspace. |

The bundle creates two jobs:

| Job | Task | Purpose |
| --- | --- | --- |
| `ProductPulse AI UC Setup` | `setup_unity_catalog` | Manual bootstrap job that creates catalog, schema, volume, and Delta table shells. |
| `ProductPulse AI Daily Pipeline` | `ingest_raw_to_bronze` | Loads raw product, order, return, inventory, review, ticket, and PDF data. |
| `ProductPulse AI Daily Pipeline` | `build_silver_chunks` | Converts source records into RAG-ready text chunks. |
| `ProductPulse AI Daily Pipeline` | `build_gold_metrics` | Builds aggregate business metric tables for SQL tool questions. |
| `ProductPulse AI Daily Pipeline` | `embed_and_sync_lakebase` | Generates embeddings and syncs chunk vectors into Lakebase. |

The daily pipeline is paused by default so an instructor can run it manually
during class.

### 2.3 Databricks Notebooks

These files are Python notebooks exported as `.py` files so they can be managed
cleanly in source control and deployed by the bundle.

| File | Pipeline layer | What it does |
| --- | --- | --- |
| `notebooks/00_setup_unity_catalog.py` | Bootstrap/governance | Creates `adidas_retail_ai.productpulse`, the `raw_data` Unity Catalog volume, and empty Bronze/Silver/Gold Delta tables. This is intentionally separate from the daily pipeline. |
| `notebooks/01_ingest_raw_to_bronze.py` | Bronze | Reads raw CSV, JSONL, and PDF files from the UC volume. It extracts PDF text with `pypdf` and writes clean Bronze Delta tables. |
| `notebooks/02_build_silver_chunks.py` | Silver RAG preparation | Standardizes reviews, tickets, and PDF text into one `silver_document_chunks` table. Reviews and tickets become one chunk each; PDFs are split into smaller windows for citations. |
| `notebooks/03_embed_and_sync_lakebase.py` | Embedding and Lakebase sync | Uses Spark SQL `ai_query` to create embeddings in `silver_document_embeddings`, then initializes Lakebase schema and upserts chunks into `rag_chunks`. |
| `notebooks/04_build_gold_metrics.py` | Gold analytics | Aggregates daily product metrics such as orders, units sold, returns, exchanges, reviews, negative sentiment, ticket count, stock, and top issue. |
| `notebooks/05_manual_agent_demo.py` | Manual validation | Runs one end-to-end question from a notebook after Lakebase sync completes. Use this before deploying or debugging the app UI. |

Execution order:

```text
00 setup UC job
upload raw data
01 ingest raw to bronze
02 build silver chunks
04 build gold metrics
03 embed and sync Lakebase
05 optional manual demo
```

The daily workflow runs tasks `01`, `02`, `04`, and `03`. After `02` finishes,
`04` and `03` are independent branches: Gold metrics can build while Lakebase
embedding sync runs. Notebook `00` is a separate manual setup job, and notebook
`05` is a separate manual demo.

### 2.4 Shared Python Package

The reusable backend code lives under `src/productpulse`. Notebooks and the
Streamlit app both use this logic.

| File | Main responsibility | Details |
| --- | --- | --- |
| `src/productpulse/__init__.py` | Package marker | Makes `productpulse` importable as a Python package. |
| `src/productpulse/config.py` | Runtime configuration | Reads environment variables, applies defaults, and builds shared settings for catalog, schema, embedding model, LLM endpoint, retrieval depth, and Lakebase connection fields. |
| `src/productpulse/databricks_models.py` | Databricks model calls | Contains the helper functions that call Databricks-hosted embedding and chat endpoints through the Databricks SDK. The bulk embedding notebook uses SQL `ai_query`, while interactive RAG query embedding uses this module. |
| `src/productpulse/lakebase.py` | Lakebase/Postgres access | Opens Lakebase connections with `pg8000`, creates `vector` extension tables, upserts RAG chunks, searches vectors, reads memory, and writes memory. |
| `src/productpulse/retrieval.py` | RAG retrieval | Rewrites the user question, embeds the rewritten query, searches Lakebase, reranks candidates, and formats numbered citation blocks for the final LLM prompt. |
| `src/productpulse/agent.py` | Agent planner and answer flow | Uses an LLM planner to choose memory, RAG, SQL metrics, normal conversation memory, or durable memory; validates the JSON plan with Python guardrails; then builds the final answer prompt. |
| `src/productpulse/text_processing.py` | Text helpers | Provides whitespace cleanup, stable ids, word chunking, and a simple keyword score used by reranking. |

Why the package is simple:

- There is no heavy agent framework.
- The planner uses the LLM for tool selection, but the validation and fallback logic is readable Python so learners can inspect how tool execution is controlled.
- SQL, RAG, and memory are separate functions instead of hidden behind a large abstraction.

### 2.5 Databricks App Files

The app is deployed as a Databricks App and runs Streamlit.

| File or folder | Used by | Purpose |
| --- | --- | --- |
| `app/app.py` | Databricks Apps / Streamlit | Main user interface. It renders the chat UI, health/status panels, starter questions, runtime configuration, trace details, and calls the backend agent. |
| `app.yml` | Databricks Apps | Starts Streamlit with `command: ["streamlit", "run", "app/app.py"]` and sets `PYTHONPATH` so the app imports the shared package from `src/productpulse`. |

Important:

- There is only one backend package copy: `src/productpulse`.
- Do not create or maintain `app/productpulse`.
- The Databricks App must be deployed from the bundle source root that contains `app.yml`, `app/`, `src/`, and `requirements.txt`.
- If you deploy only the `app/` folder, the app will fail with `ModuleNotFoundError: No module named 'productpulse'` because `src/productpulse` will not be present in the app source.

### 2.6 Data Files

The class dataset lives under `data/raw`. This is the local copy that gets
uploaded into the Unity Catalog volume.

| File or folder | Feed type | Purpose |
| --- | --- | --- |
| `data/raw/products.csv` | Product master | One row per synthetic adidas-inspired product. Contains product id, name, category, gender, launch date, price, fit profile, and known risk theme. |
| `data/raw/orders/event_date=*/orders.jsonl` | Daily order fact feed | Seven days of order lines. Used for units sold, order counts, revenue context, and product demand signals. |
| `data/raw/returns/event_date=*/returns.jsonl` | Daily return fact feed | Seven days of return and exchange events. Used for return rates, refund amount, exchange count, and return reason analytics. |
| `data/raw/reviews/event_date=*/reviews.jsonl` | Daily review feed | Seven days of customer review text with ratings, sentiment, issue tags, market, locale, channel, and verified purchase flag. Used for RAG and sentiment analytics. |
| `data/raw/support_tickets/event_date=*/tickets.jsonl` | Daily support feed | Seven days of ticket text with reason, priority, status, customer segment, and channel. Used for RAG and support workload analytics. |
| `data/raw/inventory_snapshots/snapshot_date=*/inventory_snapshots.jsonl` | Daily inventory snapshot | Seven days of product inventory by fulfillment node. Used for stock status and exchange feasibility context. |
| `data/raw/knowledge_pdf_manifest.csv` | PDF metadata | Metadata for each PDF document: document id, source type, product id where relevant, title, effective date, file name, and page count. |
| `data/raw/knowledge_pdfs/*.pdf` | Internal unstructured documents | Detailed synthetic product briefs, SOPs, QA guides, support macros, return operations, and escalation documents. These are extracted into Bronze and chunked for RAG. |

The generated scale is:

| Feed | Volume |
| --- | ---: |
| Products | 24 |
| Order lines | 12,819 |
| Return events | 925 |
| Reviews | 2,331 |
| Support tickets | 720 |
| Inventory snapshots | 672 |
| Internal PDF documents | 10 documents, 100 total pages |

The PDF files are:

| PDF file | Business purpose |
| --- | --- |
| `retail_returns_operations_playbook_2026.pdf` | Return, exchange, refund, and triage handling guidance. |
| `run_ultra_01_launch_fit_brief.pdf` | Product launch and fit guidance for `RUN-ULTRA-01`. |
| `soc_cleat_04_fg_qa_warranty_guide.pdf` | QA and warranty field guide for `SOC-CLEAT-04`. |
| `product_content_operations_feedback_sop.pdf` | SOP for turning feedback patterns into product-page updates. |
| `training_apparel_colorfastness_quality_review.pdf` | Apparel quality review focused on colorfastness and care signals. |
| `inventory_aware_exchange_operations_guide.pdf` | Guidance for using inventory signals when recommending exchanges. |
| `voice_of_customer_launch_digest_week_33.pdf` | Weekly VOC readout across launch products and markets. |
| `marketplace_product_content_alignment_guide.pdf` | Marketplace copy and PDP alignment guidance. |
| `support_macro_library_product_issue_intake.pdf` | Support-agent intake macros for product issues. |
| `product_quality_escalation_thresholds.pdf` | Thresholds for escalating repeated quality or warranty signals. |

### 2.7 Data Generation Script

| File | Purpose |
| --- | --- |
| `scripts/generate_sample_data.py` | Regenerates the full synthetic dataset under `data/raw`, including structured feeds and PDF files. Use this when you want to reset or expand the demo data. |

The script intentionally creates correlated data. For example, products with a
known fit-risk theme receive more matching review language, support tickets, and
returns. This makes classroom analytics questions meaningful instead of random.

### 2.8 Lakebase SQL

| File | Purpose |
| --- | --- |
| `sql/01_lakebase_schema.sql` | Optional manual schema setup for Lakebase. It creates `vector`, `rag_chunks`, `conversation_memory`, and indexes. Notebook `03_embed_and_sync_lakebase.py` also creates these objects automatically if the Lakebase user has permission. |

Use this SQL file when you want to initialize Lakebase manually from the
Lakebase SQL editor, or when permissions need to be validated separately from
the Databricks workflow job.

### 2.9 Diagrams

| File | Purpose |
| --- | --- |
| `diagrams/productpulse_high_level_architecture_imagegen.png` | Presentation-grade raster architecture image generated for bootcamp slides, README, and project walkthroughs. |
| `diagrams/productpulse_lld_hand_sketch_flow.jpg` | Hand-sketched flow-perspective LLD diagram. Use this when explaining component interaction without overwhelming learners with file-level implementation details. |
| `diagrams/productpulse_data_model_two_lane_hand_sketch.jpg` | Preferred hand-sketched data model diagram. It separates the RAG/text evidence flow from the structured analytics flow so learners can clearly see that Lakebase `rag_chunks` comes from Silver embeddings, while Gold metrics feed SQL Warehouse. |
| `diagrams/productpulse_data_model_two_lane_hand_sketch.png` | Source PNG version of the preferred two-lane data model image. |
| `diagrams/productpulse_data_model_corrected_hand_sketch.jpg` | Corrected hand-sketched data model diagram. Shows `silver_document_embeddings` syncing to Lakebase `rag_chunks`, while Gold tables feed SQL Warehouse separately. |
| `diagrams/productpulse_data_model_corrected_hand_sketch.png` | Source PNG version of the corrected hand-sketched data model image. |
| `diagrams/productpulse_data_model_hand_sketch.jpg` | Earlier hand-sketched data model image kept as a generated artifact. Prefer the corrected version above for teaching. |
| `diagrams/productpulse_rag_flow_hand_sketch.jpg` | Hand-sketched RAG flow diagram showing offline indexing and online answer generation with citations. |
| `diagrams/productpulse_agent_flow_lane_based_hand_sketch.jpg` | Preferred hand-sketched agent flow diagram. It avoids confusing yes/no arrows and shows SQL-only, RAG + memory, and hybrid execution as separate lanes. |
| `diagrams/productpulse_agent_flow_lane_based_hand_sketch.png` | Source PNG version of the preferred lane-based agent flow image. |
| `diagrams/productpulse_agent_flow_hand_sketch.jpg` | Earlier hand-sketched agent flow image kept as a generated artifact. Prefer the lane-based version above for teaching. |
| `diagrams/productpulse_data_model_hand_sketch.png` | Source PNG version of the hand-sketched data model image. |
| `diagrams/productpulse_rag_flow_hand_sketch.png` | Source PNG version of the hand-sketched RAG flow image. |
| `diagrams/productpulse_agent_flow_hand_sketch.png` | Source PNG version of the hand-sketched agent flow image. |
| `diagrams/productpulse_lld_component_flow_imagegen.png` | Presentation-grade raster LLD image showing deployment, ingestion, Delta processing, embedding sync, Lakebase, SQL Warehouse, app runtime, agent routing, RAG, SQL-only path, memory, and final answer trace. |
| `diagrams/productpulse_lld_component_flow.svg` | Low-level component interaction diagram showing local CLI actions, bundle deployment, UC setup, raw upload, notebook workflow tasks, Delta tables, Lakebase sync, SQL Warehouse, app config, agent routing, RAG, SQL-only path, model calls, and memory writeback. |
| `diagrams/high_level_architecture.svg` | Polished executive-level view of the full project architecture: raw retail sources, Unity Catalog, Delta layers, Lakebase, SQL Warehouse, Foundation Models, and Databricks Apps. |
| `diagrams/architecture.svg` | End-to-end architecture from raw data landing to Databricks Apps. |
| `diagrams/data_model.svg` | Logical data model across Bronze, Silver, Gold, and Lakebase stores. |
| `diagrams/rag_flow.svg` | RAG sequence: chunking, embedding, retrieval, reranking, and citation construction. |
| `diagrams/agent_flow.svg` | Agent sequence: planner, memory, RAG, SQL metrics, final answer, and memory writeback. |

Use these diagrams in the bootcamp before opening code. They make the project
easier to understand at architecture level before learners inspect notebooks.

Important data-model clarification:

- Structured facts such as products, orders, returns, and inventory snapshots
  are not chunked for RAG. They remain in Bronze and are aggregated into Gold
  metric tables.
- Reviews and support tickets contain both structured columns and text. Their
  text fields are chunked for RAG, while their structured fields are preserved
  as metadata and also contribute to Gold metrics.
- PDF documents are unstructured evidence. Their extracted text is chunked for
  RAG.
- Lakebase `rag_chunks` is populated from `silver_document_embeddings`.
- Gold tables are queried through SQL Warehouse. They do not feed Lakebase
  `rag_chunks`.

Important agent-flow clarification:

- The agent planner does not execute a chain of yes/no decision arrows.
- It asks the configured Databricks chat model to return a strict JSON tool plan.
- The Python code validates that plan, applies guardrails, and falls back to deterministic routing if the planner model fails.
- SQL-only metric questions set `query_metrics = true`, `rag_search = false`,
  `read_memory = false`, and `write_memory = false`.
- Normal product/support questions set `read_memory = true`,
  `rag_search = true`, and `write_memory = true`.
- Hybrid questions also set `query_metrics = true` when the question asks for
  counts, rankings, rates, or trends in addition to evidence.
- The preferred agent diagram therefore shows separate execution lanes instead
  of decision diamonds.

## 3. Prerequisites

You need:

- Databricks Free Edition workspace.
- Databricks CLI installed locally.
- Databricks CLI authenticated to your workspace.
- A Databricks SQL Warehouse.
- Lakebase access in the same Databricks workspace.
- Permission to create or use the target Unity Catalog catalog/schema.

Recommended default names:

```text
Unity Catalog catalog: adidas_retail_ai
Unity Catalog schema:  productpulse
Lakebase project:      productpulse-ai-lakebase
Databricks App:        productpulse-ai
Lakebase database:     databricks_postgres
Lakebase branch:       production
```

Use these names for the bootcamp unless your workspace requires different values.

## 4. Understand `.env.example`

`.env.example` is only a local template. It is not used automatically by Databricks Workflows or Databricks Apps.

Use it only if you want to run Streamlit locally from your laptop:

```bash
cp .env.example .env
```

Then fill `.env` with real values.

Do not put real secrets in `.env.example`.

Do not commit `.env`.

Databricks execution uses different configuration mechanisms:

| Surface | How configuration is supplied |
| --- | --- |
| Databricks Jobs | Job parameters and notebook widgets |
| Databricks Apps | `app.yml` plus attached app resources |
| Local Streamlit | `.env` copied from `.env.example` |

## 5. Authenticate Databricks CLI

From the project folder:

```bash
databricks auth login --host https://<your-workspace-host>
databricks current-user me
```

If `databricks current-user me` fails, fix authentication before continuing.

Why this is needed:

- `databricks bundle deploy` uses the CLI identity.
- Uploading raw files to the Unity Catalog volume uses the CLI identity.

## 6. Create Lakebase Project

In Databricks, open **Lakebase Postgres** and create a new project.

Use:

```text
Display name: productpulse-ai-lakebase
Postgres version: 17
Region: keep the workspace-selected region
Serverless usage policy: choose the allowed policy, or None if that is the only option
```

The UI will show:

```text
Project branch: production
Database: databricks_postgres
```

What this means:

- The Lakebase **project** is the managed Postgres project.
- The **production** branch is the default database branch created by Lakebase.
- The database name `databricks_postgres` is created by Databricks and may not be editable in Free Edition.
- You give a custom name to the Lakebase project, not to the default database.

For this project, keep:

```text
PGDATABASE=databricks_postgres
PGSSLMODE=require
```

## 7. Deploy Bundle Resources

From the project root:

```bash
databricks bundle validate
databricks bundle deploy
```

This deploys two jobs:

```text
ProductPulse AI UC Setup
ProductPulse AI Daily Pipeline
```

The UC setup job is separate by design. It is a bootstrap/governance task and should be run manually.

The daily pipeline is paused by default. Run it manually during the bootcamp unless you intentionally unpause the schedule.

If a previous deployment lock exists:

```bash
databricks bundle deploy --force-lock
```

## 8. Run Unity Catalog Setup Job

Open **Workflows** and run:

```text
ProductPulse AI UC Setup
```

Use job parameters:

```text
catalog = adidas_retail_ai
schema  = productpulse
```

This runs:

```text
notebooks/00_setup_unity_catalog.py
```

It creates:

- Catalog, if needed.
- Schema, if needed.
- Unity Catalog volume: `adidas_retail_ai.productpulse.raw_data`
- Bronze tables.
- Silver tables.
- Gold tables.

Expected tables:

```text
bronze_products
bronze_reviews
bronze_support_tickets
bronze_orders
bronze_returns
bronze_inventory_snapshots
bronze_knowledge_docs
silver_document_chunks
silver_document_embeddings
gold_product_daily
gold_issue_daily
```

## 9. Upload Raw Data

After the UC setup job succeeds, upload raw data to the UC volume:

```bash
databricks fs rm -r dbfs:/Volumes/adidas_retail_ai/productpulse/raw_data/raw
databricks fs cp -r data/raw dbfs:/Volumes/adidas_retail_ai/productpulse/raw_data/raw --overwrite
```

If you changed catalog/schema:

```bash
databricks fs rm -r dbfs:/Volumes/<catalog>/<schema>/raw_data/raw
databricks fs cp -r data/raw dbfs:/Volumes/<catalog>/<schema>/raw_data/raw --overwrite
```

The remove step is intentional when refreshing the class dataset. The new
dataset has different PDF file names and more daily partitions, so deleting the
old raw folder prevents stale files from mixing with the refreshed data.

Expected destination:

```text
/Volumes/adidas_retail_ai/productpulse/raw_data/raw
```

The raw dataset includes:

| Feed | Layout |
| --- | --- |
| Product master | `products.csv` |
| Orders | Daily JSONL folders |
| Returns | Daily JSONL folders |
| Inventory snapshots | Daily JSONL folders |
| Reviews | Daily JSONL folders |
| Support tickets | Daily JSONL folders |
| Internal knowledge documents | PDF files plus `knowledge_pdf_manifest.csv` |

Current generated scale:

| Feed | Volume |
| --- | ---: |
| Products | 24 |
| Order lines | 12,819 |
| Return events | 925 |
| Reviews | 2,331 |
| Support tickets | 720 |
| Inventory snapshots | 672 |
| Internal PDF documents | 10 documents, 100 total pages |

The daily feeds cover:

```text
2026-08-14 through 2026-08-20
```

The PDFs are real files in the project and are extracted during bronze ingestion.

## 10. Confirm Model Endpoints

Default endpoints:

```text
Embedding model: databricks-qwen3-embedding-0-6b
LLM endpoint:    databricks-meta-llama-3-3-70b-instruct
```

The embedding notebook uses Spark SQL `ai_query`:

```sql
ai_query('databricks-qwen3-embedding-0-6b', chunk_text)
```

This avoids unstable Python-side serving endpoint loops for the bulk embedding job.

If your workspace uses a different embedding endpoint, update:

- Job parameter `embedding_model`
- `PRODUCTPULSE_EMBEDDING_MODEL` in `app.yml`

If your workspace uses a different chat endpoint, update:

- `PRODUCTPULSE_LLM_ENDPOINT` in `app.yml`
- Notebook widget `llm_endpoint` for manual demo runs

## 11. Run Daily Pipeline

Open **Workflows** and run:

```text
ProductPulse AI Daily Pipeline
```

Before running it, fill these job parameters:

```text
pg_host      = <Lakebase host>
pg_port      = 5432
pg_database  = databricks_postgres
pg_user      = <Lakebase database user>
pg_password  = <Lakebase password or short-lived database token>
pg_sslmode   = require
embedding_model = databricks-qwen3-embedding-0-6b
```

The pipeline tasks are:

| Task | Notebook | Purpose |
| --- | --- | --- |
| `ingest_raw_to_bronze` | `01_ingest_raw_to_bronze.py` | Load structured feeds and extract PDF text |
| `build_silver_chunks` | `02_build_silver_chunks.py` | Create RAG-ready chunks |
| `build_gold_metrics` | `04_build_gold_metrics.py` | Build metric tables for SQL Warehouse queries |
| `embed_and_sync_lakebase` | `03_embed_and_sync_lakebase.py` | Generate embeddings and upsert chunks into Lakebase |

Expected results:

- Bronze tables are populated.
- `silver_document_chunks` is populated.
- `silver_document_embeddings` is populated.
- `gold_product_daily` and `gold_issue_daily` are populated.
- Lakebase table `rag_chunks` contains the same chunk population as the silver chunk table.
- Lakebase table `conversation_memory` exists and is ready for the app.

## 12. Lakebase Schema

The embedding sync notebook initializes Lakebase automatically by calling:

```python
init_lakebase_schema(...)
```

It creates:

```text
rag_chunks
conversation_memory
```

It also enables standard pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The project does not use `lakebase_vector` or Lakebase Search.

If you want to initialize Lakebase manually, use:

```text
sql/01_lakebase_schema.sql
```

## 13. Configure SQL Warehouse HTTP Path

The app needs a SQL Warehouse HTTP path for structured metric questions.

Find it in Databricks:

1. Open **SQL Warehouses**.
2. Select your warehouse.
3. Open **Connection details**.
4. Copy **HTTP path**.

Put it in:

```text
app.yml
```

Example:

```yaml
- name: "DATABRICKS_SQL_HTTP_PATH"
  value: "/sql/1.0/warehouses/<warehouse-id>"
```

Why this is needed:

- RAG questions use Lakebase.
- Metric questions use the SQL Warehouse.
- SQL-only questions skip Lakebase and require this SQL path.

## 14. Configure Databricks App

Open:

```text
app.yml
```

Confirm:

```yaml
command: ["streamlit", "run", "app/app.py"]
env:
  - name: "PRODUCTPULSE_CATALOG"
    value: "adidas_retail_ai"
  - name: "PRODUCTPULSE_SCHEMA"
    value: "productpulse"
  - name: "PRODUCTPULSE_EMBEDDING_MODEL"
    value: "databricks-qwen3-embedding-0-6b"
  - name: "PRODUCTPULSE_LLM_ENDPOINT"
    value: "databricks-meta-llama-3-3-70b-instruct"
  - name: "DATABRICKS_SQL_HTTP_PATH"
    value: "/sql/1.0/warehouses/<warehouse-id>"
  - name: "LAKEBASE_ENDPOINT"
    valueFrom: "postgres"
```

Important:

- The Databricks App **Environment** page is usually read-only.
- Edit `app.yml` locally.
- Run `databricks bundle deploy --force-lock`.
- Redeploy the Databricks App.

## 15. Create Databricks App

In Databricks, open **Apps** and create an app.

Recommended app name:

```text
productpulse-ai
```

When the app compute is ready, attach Lakebase:

1. Open the app.
2. Go to **Settings** or the app resource configuration area.
3. Add a Lakebase database resource.
4. Select the ProductPulse Lakebase project.
5. Use resource key:

```text
postgres
```

This key must match:

```yaml
valueFrom: "postgres"
```

Databricks then injects Lakebase-related values into the app runtime, such as:

```text
PGHOST
PGPORT
PGDATABASE
PGUSER
PGSSLMODE
DATABRICKS_CLIENT_ID
DATABRICKS_CLIENT_SECRET
```

For Lakebase Autoscaling, the app code uses `LAKEBASE_ENDPOINT` plus Databricks app credentials to generate short-lived database credentials. You do not manually type `PGPASSWORD` into the Databricks App UI.

## 16. Deploy Databricks App Source

First deploy the bundle:

```bash
databricks bundle deploy --force-lock
```

Then in Databricks Apps, click **Deploy**.

Select the bundle source folder:

```text
/Workspace/Users/<your-email>/.bundle/productpulse_ai/default/files
```

Do not select only `app.py`.

The folder must contain:

```text
app.yml
app/
src/
requirements.txt
```

`src/productpulse` is required because the app uses the same shared backend
package as the notebooks and jobs. This project intentionally keeps one copy of
the backend code.

The app starts Streamlit with:

```text
streamlit run app/app.py
```

## 17. Grant Unity Catalog And SQL Warehouse Permissions To The App

The Databricks App runs as an app service principal. It needs access to the Unity Catalog objects and SQL Warehouse.

Find the app service principal:

1. Open the app.
2. Open the Runtime panel in the app UI.
3. Copy the value shown as `lakebase_user`, or use the app ID/service principal shown in Databricks Apps.

Grant Unity Catalog permissions from a SQL editor or notebook using an identity that can grant privileges:

```sql
GRANT USE CATALOG ON CATALOG adidas_retail_ai TO `001f20f2-d766-4d5b-838f-34324409b0b0`;
GRANT USE SCHEMA ON SCHEMA adidas_retail_ai.productpulse TO `001f20f2-d766-4d5b-838f-34324409b0b0`;
GRANT SELECT ON TABLE adidas_retail_ai.productpulse.gold_product_daily TO `001f20f2-d766-4d5b-838f-34324409b0b0`;
GRANT SELECT ON TABLE adidas_retail_ai.productpulse.gold_issue_daily TO `001f20f2-d766-4d5b-838f-34324409b0b0`;
GRANT SELECT ON TABLE adidas_retail_ai.productpulse.silver_document_chunks TO `001f20f2-d766-4d5b-838f-34324409b0b0`;
```

Also grant **CAN USE** on the SQL Warehouse from the SQL Warehouse permissions UI.

Why this is needed:

- SQL metric questions query `gold_product_daily`.
- The app uses Databricks SQL connector through the app service principal.
- Without warehouse permission, SQL-only questions will fail or return a warning.

## 18. Grant Lakebase Table Permissions To The App

The app also needs Lakebase table privileges.

After attaching the Lakebase resource, open the app and copy:

```text
lakebase_user
```

Connect to Lakebase as your admin/OAuth user and run:

```sql
GRANT USAGE ON SCHEMA public TO "<lakebase_user_from_app_runtime>";
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE rag_chunks TO "<lakebase_user_from_app_runtime>";
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE conversation_memory TO "<lakebase_user_from_app_runtime>";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "<lakebase_user_from_app_runtime>";
```

Why this is needed:

- `rag_chunks` powers vector retrieval.
- `conversation_memory` stores chat memory.
- The app role is separate from the user or job identity that created the Lakebase tables.

## 19. App Memory Behavior

The app uses one Lakebase table:

```text
conversation_memory
```

It stores two memory types:

| Memory type | When written | Purpose |
| --- | --- | --- |
| `conversation_turn` | Normal non-SQL chat turns | Compact record of what was asked and answered |
| `user_preference` | Explicit durable instruction | Reusable preference or instruction |

Example normal question:

```text
For RUN-ULTRA-01, what should I look at first?
```

Expected trace:

```text
write_memory = true
write_durable_memory = false
Memory write result contains saved:conversation_turn
```

Example durable memory question:

```text
Remember that I care most about sizing and fit issues for running shoes.
```

Expected trace:

```text
write_memory = true
write_durable_memory = true
Memory write result contains saved:user_preference
Memory write result contains saved:conversation_turn
```

SQL-only questions do not touch Lakebase:

```text
What are the top products by ticket count?
```

Expected trace:

```text
Execution mode = sql_only_metrics
Lakebase used = false
write_memory = false
write_durable_memory = false
Memory seen = []
Retrieved evidence = []
```

## 20. Agent Routing Behavior

The agent uses an LLM planner for routing.

For every app question, the flow is:

```text
user question
    -> LLM planner
    -> JSON tool plan
    -> Python validation and guardrails
    -> selected tools execute
    -> final LLM answer
```

The planner returns fields like:

```json
{
  "route": "hybrid",
  "read_memory": true,
  "rag_search": true,
  "query_metrics": true,
  "write_memory": true,
  "write_durable_memory": false,
  "reason": "The user asks for explanation and issue size."
}
```

Why there are still Python guardrails:

- The LLM chooses the intended route.
- Python validates the JSON before executing tools.
- Explicit SQL-only questions are forced to skip Lakebase.
- Explicit remember instructions are forced to write durable memory.
- If the LLM planner fails, the deterministic fallback keeps the demo running.

To temporarily disable LLM planning and use fallback routing only, set:

```text
PRODUCTPULSE_DISABLE_LLM_PLANNER=true
```

The agent has these practical execution modes.

### 20.1 Memory + RAG Agent Mode

This mode uses Lakebase memory, Lakebase vector retrieval, optional SQL metrics, and the final LLM.

Examples:

```text
Why are customers returning RUN-ULTRA-01?
```

```text
What should support agents say about narrow toe box complaints?
```

```text
Summarize negative sentiment for SOC-CLEAT-04.
```

Expected trace:

```text
Execution mode = memory_rag_agent or hybrid_rag_sql_agent
Lakebase used = true
read_memory = true
rag_search = true
Retrieved evidence = populated
planner_source = llm_planner
```

### 20.2 SQL-Only Metrics Mode

This mode uses only the Databricks SQL Warehouse and final LLM response formatting. It does not read or write Lakebase.

Pure aggregate questions automatically use SQL-only mode:

```text
What are the top products by ticket count?
```

```text
Which product has the highest return rate?
```

```text
Rank products by negative sentiment count.
```

You can also force SQL-only mode:

```text
SQL only: Show total orders, units sold, returns, and tickets for RUN-ULTRA-01.
```

Expected trace:

```text
Execution mode = sql_only_metrics
Lakebase used = false
read_memory = false
rag_search = false
query_metrics = true
write_memory = false
write_durable_memory = false
Structured metrics = populated
Retrieved evidence = []
Memory seen = []
```

Where to confirm SQL ran:

- Open the configured SQL Warehouse.
- Check **Query history**.
- The query may appear under the Databricks App service principal, not your personal user.

## 21. Recommended Validation Sequence

Run these after pipeline and app deployment.

### 21.1 RAG Retrieval

Ask:

```text
Why are customers returning RUN-ULTRA-01?
```

Expected:

- `Execution mode = memory_rag_agent`
- `Lakebase used = true`
- Retrieved evidence is populated.
- Answer cites evidence.

### 21.2 SQL-Only Metrics

Ask:

```text
What are the top products by ticket count?
```

Expected:

- `Execution mode = sql_only_metrics`
- `Lakebase used = false`
- Structured metrics is populated.
- Retrieved evidence is empty.

### 21.3 Memory Write

Ask:

```text
Remember that I care most about sizing and fit issues for running shoes.
```

Expected:

- `write_memory = true`
- `write_durable_memory = true`
- `saved:user_preference`
- `saved:conversation_turn`

### 21.4 Memory Read

Ask in the same chat session:

```text
For RUN-ULTRA-01, what should I look at first?
```

Expected:

- Memory seen includes the sizing/fit preference.
- Answer reflects that preference when relevant.

### 21.5 Direct Memory Recall

Ask:

```text
What do you remember about my preferences?
```

Expected:

- Memory seen includes the saved preference.
- The answer summarizes the preference.

## 22. Optional Manual Agent Notebook

After the daily pipeline succeeds, you can run:

```text
notebooks/05_manual_agent_demo.py
```

Set widgets:

```text
pg_host      = <Lakebase host>
pg_port      = 5432
pg_database  = databricks_postgres
pg_user      = <Lakebase user>
pg_password  = <Lakebase password or short-lived token>
pg_sslmode   = require
embedding_model = databricks-qwen3-embedding-0-6b
llm_endpoint    = databricks-meta-llama-3-3-70b-instruct
```

This notebook asks:

```text
Why are customers returning RUN-ULTRA-01, and what should support agents do next?
```

It prints the answer and displays retrieved evidence.

## 23. Optional Local Streamlit Run

Local Streamlit is optional. Use it only if you want to test from your laptop.

Create `.env`:

```bash
cp .env.example .env
```

Fill:

```text
DATABRICKS_HOST=https://<workspace-host>
DATABRICKS_TOKEN=<personal-access-token>
DATABRICKS_SQL_HTTP_PATH=/sql/1.0/warehouses/<warehouse-id>
PRODUCTPULSE_CATALOG=adidas_retail_ai
PRODUCTPULSE_SCHEMA=productpulse
PRODUCTPULSE_EMBEDDING_MODEL=databricks-qwen3-embedding-0-6b
PRODUCTPULSE_LLM_ENDPOINT=databricks-meta-llama-3-3-70b-instruct
PGHOST=<lakebase-host>
PGPORT=5432
PGDATABASE=databricks_postgres
PGUSER=<lakebase-user>
PGPASSWORD=<password-or-token>
PGSSLMODE=require
```

Install packages:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app/app.py
```

## 24. Troubleshooting

| Symptom | Meaning | Fix |
| --- | --- | --- |
| `Libraries field is not supported for serverless task` | Libraries were configured directly on a serverless task | Use bundle `environments` dependencies, already configured in `databricks.yml` |
| Embedding endpoint 404 | Endpoint name is wrong for the workspace | Use `databricks-qwen3-embedding-0-6b` or another visible embedding endpoint |
| Python kernel unresponsive during embedding | Python-side endpoint loop or binary dependency instability | Use current notebook 03 with Spark SQL `ai_query`; redeploy latest code |
| `lakebase_vector must be loaded via shared_preload_libraries` | Using Lakebase Search extension accidentally | Use standard pgvector only: `CREATE EXTENSION IF NOT EXISTS vector` |
| `password authentication failed` | Wrong Lakebase database password/token/user | Use the exact Lakebase user and a valid short-lived token or configured database password |
| `permission denied for schema public` | App Lakebase role lacks schema usage | Grant `USAGE ON SCHEMA public` to the app Lakebase user |
| `permission denied for table conversation_memory` | App can connect to Lakebase but cannot read/write memory | Grant table privileges on `conversation_memory` |
| `User does not have USE CATALOG` | App service principal lacks Unity Catalog permission | Run UC grants for the app service principal |
| SQL metric warning about credentials | SQL Warehouse path or permission is missing | Set `DATABRICKS_SQL_HTTP_PATH` and grant SQL Warehouse CAN USE |
| SQL-only question still shows `memory_rag_agent` | Old app code is deployed or question needs unstructured explanation | Redeploy bundle and app; pure aggregate questions should show `sql_only_metrics` |
| App Environment values are not editable | Databricks App environment tab is read-only | Edit `app.yml`, deploy bundle, redeploy app |
| Deploy dialog shows `app.py` only | You are browsing inside the app folder | Select the parent bundle source folder containing `app.yml`, `app/`, `src/`, and `requirements.txt` |

## 25. Redeployment Checklist

Use this checklist whenever code or `app.yml` changes:

```bash
cd /Users/shashankmishra/Desktop/AI_DE_With_AWS/Databricks-Class-2/productpulse-ai
databricks bundle deploy --force-lock
```

Then:

1. Open Databricks Apps.
2. Open `productpulse-ai`.
3. Click **Deploy**.
4. Select:

```text
/Workspace/Users/<your-email>/.bundle/productpulse_ai/default/files
```

5. Wait for deployment to finish.
6. Open the app.
7. Validate with:

```text
What are the top products by ticket count?
```

Expected:

```text
Execution mode = sql_only_metrics
Lakebase used = false
Retrieved evidence = []
```

8. Validate RAG with:

```text
Why are customers returning RUN-ULTRA-01?
```

Expected:

```text
Execution mode = memory_rag_agent
Lakebase used = true
Retrieved evidence = populated
```

## 26. Teaching Flow For The Bootcamp

Recommended class sequence:

1. Explain the business problem and dataset.
2. Show the architecture diagram.
3. Run UC setup job.
4. Upload raw data to UC volume.
5. Run daily pipeline.
6. Inspect bronze tables.
7. Inspect silver chunks and PDF text extraction.
8. Inspect gold metrics.
9. Inspect Lakebase `rag_chunks`.
10. Open the Databricks App.
11. Demo SQL-only metric question.
12. Demo RAG question with citations.
13. Demo memory write and follow-up memory read.
14. Discuss permissions, service principals, and production hardening.

This gives learners a clean end-to-end path from raw retail data to a governed conversational AI application.
