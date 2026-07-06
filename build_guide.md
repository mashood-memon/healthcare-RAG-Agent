# Healthcare Facility RAG Agent — End-to-End Build Guide

Source data: 4 state CSVs (NC, CO, AZ, CA), 20,769 total rows, mixing state licensing
directories (60.6%) with CMS Medicare-certified quality data (39.4%). See `crosswalk.csv`
for the full field mapping referenced throughout this guide.

---

## Phase 0 — Environment setup (Day 1)

- [ ] Neon Postgres project (new instance, separate from any other project's DB)
- [ ] Qdrant Cloud cluster (free tier is enough to start — ~21K points is tiny)
- [ ] OpenAI API key (for `text-embedding-3-small` — plain text embeddings, no multimodal needed here)
- [ ] Repo scaffold: `ingestion/`, `agent/`, `api/`, `eval/`, `crosswalk.yaml`
- [ ] `.env` for DB URLs, API keys — never commit these

Don't touch LangGraph yet. Data layer first — an agent over broken data is worse than no agent.

---

## Phase 1 — Crosswalk config (Day 1-2)

Convert `crosswalk.csv` (attached) into a machine-readable `crosswalk.yaml` your ingestion
script will read. Example for a few fields:

```yaml
fields:
  rn_hours_per_resident_day:
    dtype: numeric
    usage: filter+agg
    sources:
      NC: REGISTERED_NURSE_RN_HOURS_PER_RESIDENT_DAY
      CO: nh_rn_hours_per_resident_day
      AZ: nh_rn_hours_per_resident_day
      CA: nh_rn_hours_per_resident_day

  speech_service:
    dtype: boolean
    usage: filter
    sources:
      NC: [SPEECH_SERVICE, PROVIDES_SPEECH_THERAPY]   # NC has TWO source cols — see note
      CO: hh_provides_speech_therapy
      AZ: hh_provides_speech_therapy
      CA: hh_provides_speech_therapy
    note: >
      NC directory rows use SPEECH_SERVICE (T/F, 91% filled).
      NC CMS rows use PROVIDES_SPEECH_THERAPY (Yes/No, only ~1.5% filled).
      CO/AZ/CA directory rows have NO service data at all — only their CMS
      subset (hh_provides_speech_therapy) is populated.

  hospital_readmission_flag:
    dtype: text
    usage: filter
    sources:
      NC: HOSPITAL_READMISSION_RATE   # categorical text, same as CO/AZ/CA
      CO: hh_hospital_readmission_rate
      AZ: hh_hospital_readmission_rate
      CA: hh_hospital_readmission_rate
    note: >
      Categorical in all 4 states — same label set: "Same As National Rate",
      "Higher than average hospital readmissions", "Lower than average
      hospital readmissions". NC additionally has an explicit "Data not
      available" label (treat as NULL, don't store as a 4th category — it's
      the source's way of saying missing, not a real value).
      No numeric field exists for this metric in ANY state — don't build an
      aggregation/AVG query against this field, it will never work. It's
      filter-only (e.g. WHERE hospital_readmission_flag = 'Lower than average').
```

**Action item:** go through `crosswalk.csv` row by row and verify dtype assumptions against
the actual CSV values (`df[col].dropna().unique()`) before writing ingestion code — don't
trust column names or prior analysis at face value; confirm against real data every time.

---

## Phase 2 — Postgres schema + ingestion (Week 1)

### 2.1 Create the schema

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS earthdistance;  -- for radius/near-me queries

CREATE TABLE facilities (
    facility_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_state TEXT NOT NULL,
    data_source TEXT NOT NULL CHECK (data_source IN ('directory','cms')),
    facility_type TEXT NOT NULL,
    name TEXT NOT NULL,
    legal_business_name TEXT,
    ccn TEXT,
    address TEXT, city TEXT, county TEXT, zip TEXT, phone TEXT,
    latitude DOUBLE PRECISION, longitude DOUBLE PRECISION,
    ownership_type TEXT,
    certification_date DATE,
    cms_region TEXT,
    bed_count INTEGER,
    overall_rating SMALLINT CHECK (overall_rating BETWEEN 1 AND 5),
    health_inspection_rating SMALLINT,
    staffing_rating SMALLINT,
    quality_measure_rating SMALLINT,
    pt_service BOOLEAN, ot_service BOOLEAN, speech_service BOOLEAN,
    iv_service BOOLEAN, dme_service BOOLEAN, hospice_service BOOLEAN,
    social_work_service BOOLEAN, home_health_aide_service BOOLEAN,
    rn_hours_per_resident_day NUMERIC,
    lpn_hours_per_resident_day NUMERIC,
    cna_hours_per_resident_day NUMERIC,
    total_nursing_hours_per_resident_day NUMERIC,
    rn_turnover_pct NUMERIC,
    total_nursing_turnover_pct NUMERIC,
    total_fines_usd NUMERIC,
    number_of_fines INTEGER,
    abuse_complaint BOOLEAN,
    special_focus_facility BOOLEAN,
    infection_control_citations INTEGER,
    health_deficiencies_count INTEGER,
    improved_walking_mobility_pct NUMERIC,
    improved_bathing_ability_pct NUMERIC,
    falls_major_injury_pct NUMERIC,
    hospital_readmission_numeric NUMERIC,
    hospital_readmission_flag TEXT,
    home_discharge_flag TEXT,
    chain_affiliation TEXT,
    owner_name TEXT, mgmt_company_name TEXT, administrator_name TEXT,
    -- derived flags — computed at ingestion, NEVER inferred at query time
    has_ratings BOOLEAN NOT NULL,
    has_staffing_data BOOLEAN NOT NULL,
    has_service_data BOOLEAN NOT NULL,
    has_geo BOOLEAN NOT NULL,
    is_geocoded_fallback BOOLEAN NOT NULL DEFAULT FALSE,
    raw_source JSONB,
    natural_key TEXT NOT NULL,  -- normalized name+address+state, for upsert dedup
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (natural_key)
);

CREATE INDEX idx_fac_state_type ON facilities(source_state, facility_type);
CREATE INDEX idx_fac_rating ON facilities(overall_rating) WHERE overall_rating IS NOT NULL;
CREATE INDEX idx_fac_geo ON facilities USING gist (ll_to_earth(latitude, longitude)) WHERE has_geo;
CREATE INDEX idx_fac_flags ON facilities(has_ratings, has_service_data, has_geo);

CREATE TABLE ingestion_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_state TEXT,
    source_file TEXT,
    rows_read INTEGER,
    rows_written INTEGER,
    rows_rejected INTEGER,
    rejection_reasons JSONB,
    started_at TIMESTAMPTZ DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE query_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT,
    classification JSONB,
    sql_executed TEXT,
    rows_returned INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### 2.2 Write the ingestion script

Pseudocode structure (`ingestion/load_state.py`):

```
def ingest(state_code, csv_path, crosswalk):
    df = read_csv(csv_path)
    run = start_ingestion_run(state_code, csv_path)
    rejected = []

    for row in df.itertuples():
        try:
            record = apply_crosswalk(row, crosswalk, state_code)
            record["has_ratings"] = record["overall_rating"] is not None
            record["has_staffing_data"] = record["rn_hours_per_resident_day"] is not None
            record["has_service_data"] = record["data_source"] == "cms"
            record["has_geo"] = record["latitude"] is not None
            record["natural_key"] = normalize_key(record["name"], record["address"], state_code)
            record["raw_source"] = row_to_json(row)  # full original row, nothing dropped
            upsert(record)  # ON CONFLICT (natural_key) DO UPDATE
        except ValidationError as e:
            rejected.append({"row": row, "reason": str(e)})

    finish_ingestion_run(run, rows_read=len(df), rows_written=len(df)-len(rejected),
                          rejected=rejected)
```

Run this once per state CSV. **Idempotent** — re-running it should update existing rows
(matched by `natural_key`), not duplicate them.

### 2.3 Geocode missing coordinates

Recall: only CMS Nursing Home rows have lat/long in NC and AZ; CO and CA directory rows
already have coordinates. Batch-geocode everything else:

```
SELECT facility_id, address, city, source_state, zip
FROM facilities WHERE has_geo = false;
```

Use a geocoding API (Google Geocoding, or Nominatim if you want to stay free), cache
results by address to avoid re-billing on re-runs, then:

```sql
UPDATE facilities SET latitude=%s, longitude=%s, has_geo=true, is_geocoded_fallback=true
WHERE facility_id=%s;
```

### 2.4 Validate before moving on

- [ ] Row counts per state match your earlier analysis (NC 6,615 / CO 2,515 / AZ 2,655 / CA 8,984)
- [ ] `has_ratings=true` count roughly matches earlier findings (NC 504, CO 321, AZ 266, CA 3,111)
- [ ] Spot-check 5 facilities per state against the original CSV row (via `raw_source`)
- [ ] No nulls in required fields (`name`, `facility_type`, `source_state`, `data_source`)

Do not proceed to Qdrant until this passes. Garbage in Postgres = garbage in Qdrant, and
you'll spend hours debugging retrieval when the actual bug is ingestion.

---

## Phase 3 — Qdrant sync (Week 1-2)

### 3.1 Collection + payload schema

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="facilities_v1",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),  # text-embedding-3-small dims
)
```

### 3.2 Description builder — the most important function in this phase

```python
def build_description(row: dict) -> str:
    parts = [f"{row['name']} is a {row['facility_type']} in {row['city']}, {row['source_state']}."]
    if row["has_ratings"]:
        parts.append(f"Overall Medicare rating: {row['overall_rating']} stars.")
    if row["bed_count"]:
        parts.append(f"{row['bed_count']} licensed beds.")
    services = [s for s in ["pt_service","ot_service","speech_service","iv_service",
                             "dme_service","hospice_service"] if row.get(s)]
    if services:
        parts.append("Provides: " + ", ".join(s.replace('_service','').upper() for s in services) + ".")
    if row["has_staffing_data"]:
        parts.append(f"{row['rn_hours_per_resident_day']} RN hours per resident day.")
    if row.get("abuse_complaint"):
        parts.append("Has an abuse complaint on record.")
    return " ".join(parts)
```

**Rule: only append a sentence if the underlying field is non-null.** Never write "Rating:
not available" into the embedded text — that string would itself be semantically matched
against queries about low ratings.

### 3.3 Sync script

```python
def sync_to_qdrant():
    rows = fetch_all_facilities()  # from Postgres
    points = []
    for row in rows:
        text = build_description(row)
        points.append(PointStruct(
            id=row["facility_id"],
            vector=embed(text),
            payload={k: row[k] for k in PAYLOAD_FIELDS},  # every filterable field
        ))
    client.upsert(collection_name="facilities_v2", points=points, wait=True)
    # verify counts match before swapping alias
    assert client.count("facilities_v2").count == len(rows)
    client.update_collection_aliases(change_aliases_operations=[
        DeleteAliasOperation(alias_name="facilities"),
        CreateAliasOperation(alias_name="facilities", collection_name="facilities_v2"),
    ])
```

Run this after every ingestion run. Since your CSVs are static snapshots, this is a
full-rebuild-and-swap, not incremental sync — much simpler, no consistency bugs possible.

- [ ] Verify point count in Qdrant matches Postgres row count
- [ ] Spot check: query a known facility by name via vector search, confirm it returns

---

## Phase 4 — LangGraph agent (Week 2-3)

### 4.1 State + classification

```python
class QueryClassification(BaseModel):
    query_type: Literal["exact_filter", "aggregation", "fuzzy", "hybrid"]
    state: str | None = None
    facility_type: str | None = None
    min_rating: int | None = None
    required_services: list[str] = []
    aggregation_field: str | None = None
    aggregation_group_by: str | None = None
    residual_fuzzy_text: str | None = None
```

Build this node first and test it standalone against ~20 example queries (including your
two originals) before wiring up the rest of the graph. If classification is wrong, nothing
downstream matters.

### 4.2 SQL tool — whitelist columns, never interpolate raw LLM output

```python
ALLOWED_COLUMNS = {"overall_rating", "rn_hours_per_resident_day", ...}  # from crosswalk

def build_query(c: QueryClassification):
    conditions, params = ["1=1"], {}
    if c.state:
        conditions.append("source_state = %(state)s"); params["state"] = c.state
    if c.facility_type:
        conditions.append("facility_type = %(ftype)s"); params["ftype"] = c.facility_type
    if c.min_rating is not None:
        conditions.append("has_ratings = true AND overall_rating >= %(rating)s")
        params["rating"] = c.min_rating
    for svc in c.required_services:
        col = f"{svc}_service"
        assert col in ALLOWED_COLUMNS  # hard guard against injection/hallucinated columns
        conditions.append(f"has_service_data = true AND {col} = true")
    where = " AND ".join(conditions)

    if c.aggregation_field:
        assert c.aggregation_field in ALLOWED_COLUMNS
        return (f"SELECT {c.aggregation_group_by}, AVG({c.aggregation_field}) avg_val, COUNT(*) n "
                f"FROM facilities WHERE {where} AND {c.aggregation_field} IS NOT NULL "
                f"GROUP BY {c.aggregation_group_by}"), params
    return f"SELECT * FROM facilities WHERE {where} LIMIT 20", params
```

### 4.3 Vector tool — Qdrant filtered search

Pass hard filters (state, facility_type, min_rating) as a Qdrant `Filter`, and only the
`residual_fuzzy_text` as the embedding query — don't embed the whole raw user question.

### 4.4 Hybrid node

Runs SQL first to shrink candidates by hard filters → passes survivor IDs into a Qdrant
filtered search on `facility_id IN (...)` → reranks by similarity on the fuzzy portion.

### 4.5 Synthesis node — grounding rules (put this directly in the system prompt)

- Only state facts present in the tool result rows.
- Cite facility name + state + `data_source` for every claim.
- If a queried field is null for a result, say so explicitly — never guess.
- If `zero_reason == "no_data_for_criteria"`, explain that the data isn't tracked for
  that category, don't imply no such facility exists.

### 4.6 Wire the graph

```python
graph.add_conditional_edges("classify_intent", route_by_type, {
    "exact_filter": "sql_tool", "aggregation": "sql_tool",
    "fuzzy": "vector_tool", "hybrid": "hybrid_node",
})
graph.add_edge("sql_tool", "synthesize")
graph.add_edge("vector_tool", "synthesize")
graph.add_edge("hybrid_node", "synthesize")
```

Add `PostgresSaver` checkpointing for conversation state — separate table from `facilities`.

---

## Phase 5 — Evaluation harness (build alongside Phase 4, not after)

Create `eval/golden_set.yaml` — 15-20 hand-verified queries per category:

```yaml
- query: "What is the average RN hours per resident day for nursing homes in NC vs CO?"
  type: aggregation
  expected: {NC: 4.02, CO: 3.71}   # computed directly from Postgres, verify by hand
- query: "Home Health agency in San Diego with speech therapy and rating above 4"
  type: exact_filter
  expected_facility_ids: [...]      # known correct set from direct SQL
- query: "Bed count for [a known AZ directory-only facility]"
  type: null_handling
  expected_response_contains: "not available"
```

Run this suite after every change to prompts, schema, or routing logic. Aggregation
correctness is pass/fail — no partial credit, exact numeric match against ground truth.

---

## Phase 6 — API + deployment (Week 3-4)

- FastAPI wrapping the LangGraph app, matching your existing `langgraph-api` template
- Neon connection pooling (PgBouncer) — agent checkpointing + facility queries both hit Postgres
- Qdrant Cloud collection with alias-based rebuilds (Phase 3)
- Structured logging to `query_logs` table for every request
- LangSmith tracing enabled end-to-end

---

## Suggested week-by-week pace

| Week | Focus |
|---|---|
| 1 | Phase 0-2: schema, crosswalk, ingestion, validate row counts |
| 2 | Phase 3-4: Qdrant sync, classification node, SQL tool |
| 3 | Phase 4 cont'd: vector/hybrid nodes, synthesis, start golden set |
| 4 | Phase 5-6: eval harness, FastAPI wrapper, deploy, fix what the eval catches |

Start with Phase 1-2 only. Don't open LangGraph until ingestion is validated — it's the
one piece that, if wrong, silently corrupts everything built on top of it.
