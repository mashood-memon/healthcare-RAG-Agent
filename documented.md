# Healthcare RAG Agent — Project Documentation

## Project Vision

The goal is to build an intelligent conversational AI agent that can answer natural language questions about healthcare facilities across four US states — North Carolina, Colorado, Arizona, and California.

Questions like:

> *"Which nursing homes in NC with a 4-star rating also offer speech therapy?"*
> *"What is the average RN hours per resident day for nursing homes in CO versus AZ?"*
> *"Find a home health agency near San Diego with good stroke recovery outcomes."*

To make this work, the agent needs to search both **precisely** (filtering by exact values like ratings, services, or state) and **semantically** (understanding fuzzy intent like "good reputation" or "safe for my grandmother"). That dual requirement is why the system is built around two parallel data stores: a Postgres relational database for hard filters and aggregations, and a Qdrant vector database for semantic search. A LangGraph agent sits on top and orchestrates between them.

But before any of that intelligence can work — the data has to be clean, structured, and consistent. That is the hard problem this project starts by solving.

---

## The Source Data

We have raw data from 4 states and the federal CMS (Centers for Medicare & Medicaid Services), totaling around **20,769 records** across:

| State | Row Count | Source |
|-------|-----------|--------|
| NC | 6,615 | State Health Dept (directory) + CMS |
| CO | 2,515 | State directory + CMS |
| AZ | 2,655 | State directory + CMS |
| CA | 8,984 | State directory + CMS |

---

## Phase 1 — The Crosswalk

Before writing a single line of ingestion code, the first step was creating a **Crosswalk** — a mapping document that answers: *"For each of the 83 standard fields we care about, what is the actual column name in each state's CSV?"*

A human-readable `crosswalk.csv` was manually authored, then compiled into `crosswalk.yaml` by `crosswalk.py`. Each row in the crosswalk defines one canonical field and lists the actual source column name per state. List-based fallbacks handle merged CSVs like NC's, where directory and CMS rows share a file but use different column names for the same field.

---

## Phase 2 — Ingestion into Postgres

### Database Schema

A Postgres database (hosted on Neon) stores all facilities in a single `facilities` table with 83+ columns. The schema uses strict CHECK constraints to catch dirty data at the database level. Several derived boolean flags are pre-computed and stored:

| Flag | What it means |
|------|---------------|
| `has_ratings` | Facility has CMS star ratings |
| `has_staffing_data` | Facility has RN/LPN/CNA hours data |
| `has_service_data` | Facility has service flags (PT, OT, speech, etc.) |
| `has_geo` | Facility has latitude/longitude coordinates |

These flags exist so the agent can filter without re-checking for nulls across a dozen columns.

### Ingestion Flow

The ingestion script is run once per state (`uv run python ingestion/load_state.py NC`). For every row: apply the crosswalk to map source columns → canonical columns, clean and type-cast values, compute derived flags, generate a `natural_key` for deduplication, then UPSERT into Postgres. Row-level SAVEPOINT rollbacks ensure that a single bad row never aborts the entire batch.

---

## Phase 2 — Geocoding

Not all facilities include coordinates. The geocoding script (`run_geocode.py`) runs after ingestion, queries Postgres for all rows where `has_geo = false`, and resolves coordinates.

**Geocoding is now done via OpenAI function calling**, not Nominatim/OpenStreetMap. This was a deliberate switch: Nominatim enforces a strict 1 request/1.5 seconds rate limit and returns errors for ambiguous inputs. The OpenAI approach resolves any free-text location (including informal descriptions like "downtown Charlotte" or "near the coast in NC") in milliseconds with no rate limiting, and the model can confidently handle partial or colloquial descriptions that Nominatim would fail on.

The `is_geocoded_fallback` flag distinguishes between coordinates that came from the original source data versus coordinates that were inferred by us.

---

## Phase 3 — Qdrant Vector Sync

Each facility record is converted into a natural language text description (e.g., *"Sunrise Manor is a Nursing Home in Charlotte, NC. Overall Medicare rating: 4 stars. Provides PT, OT, and speech therapy. 0.48 RN hours per resident day."*), embedded using OpenAI's `text-embedding-3-small` model, and stored in a Qdrant Cloud collection (`facilities_v1`) alongside the filterable fields as payload.

Payload indexes for `source_state`, `facility_type`, `overall_rating`, and `facility_id` are created explicitly. Qdrant requires these indexes to run filtered queries efficiently — without them, filtered requests fail with a 400 Bad Request error.

---

## Phase 4 — The LangGraph Agent

This is the core of the system. A `StateGraph` connects four nodes: **classify → tool → synthesize**. State flows through as an `AgentState` dictionary that accumulates the query, classification, tool results, response, and conversation history across every node.

```
User Query
    │
    ▼
[classify_intent]  ──── GPT-4o-mini (structured output)
    │
    │  query_type?
    ├──── exact_filter or aggregation ──→ [sql_tool] ──→ [synthesize] ──→ Response
    ├──── fuzzy ─────────────────────────→ [vector_tool] ──→ [synthesize] ──→ Response
    └──── hybrid ────────────────────────→ [hybrid_tool] ──→ [synthesize] ──→ Response
```

### Node 1: classify_intent

Every incoming query hits this node first. It sends the user's message (plus the full conversation history for multi-turn context) to GPT-4o-mini with a structured output format. The model returns a `QueryClassification` Pydantic object with 15 fields, including:

- `query_type` — One of `exact_filter`, `aggregation`, `fuzzy`, or `hybrid`
- `states` — Extracted state codes, validated against a known whitelist
- `facility_type`, `min_rating`, `required_services` — Hard filter fields
- `location_text` — A free-text location string for radius search (e.g., "near San Diego")
- `residual_fuzzy_text` — The portion of the query that cannot be expressed as a SQL filter (e.g., "good reputation", "warm and family-friendly") — this is what gets embedded for semantic search

The routing rule is strict: if `residual_fuzzy_text` is null, the type must be `exact_filter` or `aggregation`, never `hybrid`. This prevents the model from choosing an expensive hybrid path for queries that are perfectly answerable by SQL alone.

The node appends the current query to the `messages` list in `AgentState` so the conversation history grows naturally.

### Node 2a: sql_tool

Handles `exact_filter` and `aggregation` queries. The `build_query` function translates the `QueryClassification` into a parameterized Postgres query. Every column name used in the WHERE or GROUP BY clause is validated against hardcoded Python sets (`FILTERABLE_COLUMNS`, `AGGREGATABLE_COLUMNS`) before it touches the SQL string — no raw LLM output is ever interpolated into the query directly.

**Location-based queries** work as follows: if `location_text` is set, it is resolved to a (lat, lon) coordinate pair via **OpenAI function calling**. The model is given a `return_coordinates` tool definition and instructed to call it with the decimal coordinates for the location. The result is parsed from the function call arguments, cached in `geo_cache` on the state, and used to build a `WHERE earth_distance(...) <= radius_miles * 1609` clause using Postgres's `earthdistance` extension.

The geo cache on `AgentState` means that if you ask two follow-up questions about the same city, the second geocode call is instant — it hits the cache and skips the OpenAI call entirely.

**Aggregation queries** compute `AVG`, `COUNT`, `SUM`, `MIN`, or `MAX` over whitelisted numeric columns, with optional `GROUP BY source_state` for state-comparison queries.

### Node 2b: vector_tool

Handles `fuzzy` queries — purely descriptive searches with no extractable hard filters. The query text (or `residual_fuzzy_text` if available) is embedded via `text-embedding-3-small`, and Qdrant's `query_points` API performs an approximate nearest-neighbor search. Hard filters from the classification (state, facility type, minimum rating) are applied as Qdrant payload filter conditions, not as post-filters — this keeps the semantic scoring meaningful by only comparing documents that already satisfy the hard constraints.

### Node 2c: hybrid_tool

Handles `hybrid` queries — the most sophisticated path. Instead of running two separate full searches, it uses a two-phase approach:

1. **SQL pre-filter:** Run the hard filters in Postgres to get up to 200 candidate `facility_id` values. This narrows the universe dramatically.
2. **Qdrant rerank:** Run the semantic search restricted to only those 200 IDs using a `HasIdCondition` filter. Qdrant ranks them by cosine similarity to the `residual_fuzzy_text` embedding.

This is much better than a post-filter approach (which would rank all 20,000 facilities and then throw away 99% of results) and produces higher-quality rankings because the vector model is comparing only semantically plausible candidates. If SQL returns zero candidates, the tool degrades gracefully to a full-corpus vector search.

### Node 3: synthesize

A GPT-4o call converts raw tool results into a grounded, cited, natural-language response. The system prompt enforces five strict rules:

1. Never guess or infer from external knowledge
2. Only state facts present in the tool result rows
3. Cite the facility name and state for specific claims
4. Distinguish between "this metric is not tracked in our database" and "no facilities matched your criteria" — these are two different zero-result reasons that require different user-facing messages
5. Include exact numbers for aggregations, not approximations

The full conversation history is injected into this node's prompt as well, so follow-up responses read naturally in the flow of the conversation without re-summarizing what was already said.

The node appends the assistant's response back to the `messages` list, completing the conversation loop.

### Multi-Turn Memory: PostgresSaver

By default, each call to `app.invoke()` is stateless. The `PostgresSaver` from `langgraph-checkpoint-postgres` changes this. Every time the graph transitions between nodes, LangGraph serializes the entire `AgentState` to a Postgres table (`checkpoints`) keyed by a `thread_id`. On the next user turn, LangGraph reads back the last snapshot for that `thread_id` and resumes from there — including the full `messages` history.

This means that if you close the CLI and reopen it with the same `thread_id`, the agent remembers everything from the previous session. In the Streamlit UI, the `thread_id` lives in `st.session_state` and persists for the duration of the browser session.

### LangChain Message Compatibility

LangGraph internally stores conversation history as LangChain `HumanMessage` and `AIMessage` Pydantic objects. The raw OpenAI API does not accept these — it requires plain `{"role": ..., "content": ...}` dictionaries. A shared `format_history_for_openai()` helper in `agent/utils.py` safely converts between the two formats before any OpenAI call.

---

## Phase 5 — Evaluation Harness

A golden set of 20 hand-curated queries (`eval/golden_set.yaml`) covers every routing path and edge case:

- Standard exact filters (state + rating + services)
- Multi-state aggregation comparisons
- Pure semantic fuzzy queries
- Hybrid queries (hard filters + descriptive text)
- Radius/geocoding queries
- City-name inference (agent correctly deduced "Charlotte" → state=NC)
- Ambiguous terms (CCRC correctly goes to fuzzy since there's no hardcoded CCRC filter)

The eval script (`eval/run_eval.py`) runs every query through the classification node and asserts that `query_type`, `states`, and `facility_type` match. It exits with code 1 on any failure, making it suitable as a pre-commit gate. It is fast — only the LLM classification call runs, not the full database pipeline.

Current score: **17/20 after first run → adjusted to 20/20** after updating 3 golden set expectations that represented correct LLM behavior (city→state inference, CCRC routing) that our initial expectations had wrong.

---

## Phase 5 — Streamlit Chat UI

A single `ui/app.py` file provides a full conversational chat interface using `st.chat_message` and `st.chat_input`. On first page load, a `uuid4` is generated and stored in `st.session_state` as the `thread_id`. This flows directly into LangGraph's `PostgresSaver`.


