# Healthcare RAG Agent — Project Documentation

## Project Vision

The goal is to build an intelligent conversational AI agent that can answer natural language questions about healthcare facilities across four US states — North Carolina, Colorado, Arizona, and California.

Questions like:

> *"Which nursing homes in NC with a 4-star rating also offer speech therapy?"*
> *"What is the average RN hours per resident day for nursing homes in CO versus AZ?"*
> *"Find a home health agency near San Diego with a low hospital readmission rate."*

To make this work, the agent needs to be able to search both **semantically** (understanding intent and fuzzy phrasing) and **precisely** (filtering by exact values like ratings, services, or state). That dual requirement is why the system is designed around two parallel data stores: a Postgres relational database for hard filters and aggregations, and a Qdrant vector database for semantic/fuzzy search. A LangGraph AI agent sits on top and orchestrates between them.

But before any of that intelligence can work — the data has to actually be clean, structured, and consistent. That is the hard problem this project starts by solving.

---

## The Source Data Problem

We have raw data from 4 states and the federal CMS (Centers for Medicare & Medicaid Services), totaling around **20,769 records** across:

| State | Row Count | Source |
|-------|-----------|--------|
| NC | 6,615 | State Health Dept (directory) + CMS |
| CO | 2,515 | State directory + CMS |
| AZ | 2,655 | State directory + CMS |
| CA | 8,984 | State directory + CMS |

Each state's CSV was assembled independently. The columns have completely different names for the same concept. For example, a nursing home's physical address is stored as:

- `ADDRESS` in NC
- `address_line1` + `address_line2` (two separate columns) in CO, AZ, and CA

And it only gets worse. The same metric — say, whether a facility provides speech therapy — is tracked differently depending on *which kind* of record it is:

- NC **directory** rows use a column called `SPEECH_SERVICE` with values like `T` or `F`
- NC **CMS** rows use a different column called `PROVIDES_SPEECH_THERAPY` with values like `Yes` or `No`
- CO/AZ/CA **directory** rows have no speech therapy data at all
- CO/AZ/CA **CMS** rows use `hh_provides_speech_therapy`

If you tried to just join all these CSVs and query them without a unified structure, the AI agent would need to know each state's quirks to reason correctly. That is fragile, unscalable, and wrong. The solution is **normalization** — translating everything into a single standard language.

---

## Phase 1 — The Crosswalk: Building a Rosetta Stone

Before writing a single line of ingestion code, the first step was creating a **Crosswalk**.

The crosswalk is a mapping document that answers the question: *"For each of the 83 standard fields we care about, what is the actual column name in each state's CSV?"*

### How It Was Built

A human-readable `crosswalk.csv` was manually authored by going through every raw CSV file and cataloguing which column corresponds to which canonical field. Each row in the crosswalk defines one canonical field and contains:

- Its **standard name** (e.g., `rn_hours_per_resident_day`)
- Its **data type** (e.g., `numeric`, `boolean`, `text`, `date`)
- Its **usage** (whether it's used for filtering, aggregating, embedding, or just metadata)
- The **actual source column name** for each of the 4 states

A small build script (`crosswalk.py`) then compiles this CSV into a machine-readable `crosswalk.yaml` file that the ingestion scripts can consume programmatically.

**Why not just hardcode the column mappings into the ingestion script?**
Because that would bury the mapping logic deep in Python code. With a YAML file, the mapping is visible, editable, and a single source of truth. If a state publishes a new data file with renamed columns, you update one line in the YAML — not hunt through ingestion code.

### A Real Example

For the canonical field `speech_service`, the crosswalk YAML looks like this:

```yaml
speech_service:
  dtype: boolean
  usage: filter
  sources:
    NC:
      - SPEECH_SERVICE           # NC directory rows use this
      - PROVIDES_SPEECH_THERAPY  # NC CMS rows use this
    CO: hh_provides_speech_therapy
    AZ: hh_provides_speech_therapy
    CA: hh_provides_speech_therapy
```

When the ingestion script processes a North Carolina row, it checks `SPEECH_SERVICE` first. If that is empty or null, it falls back to `PROVIDES_SPEECH_THERAPY`. This list-based fallback handles the fact that the NC CSV is a *merged* file containing both directory and CMS records, each with different column structures.

---

## Phase 2 — Ingestion: Getting the Data into Postgres

### The Database Schema

A Postgres database (hosted on Neon) stores all facilities in a single `facilities` table with 83+ columns — one for every canonical field. The schema is deliberately strict:

- Star ratings have `CHECK (overall_rating BETWEEN 1 AND 5)` — the database will reject any row where a rating is out of that range, which caught real dirty data where CMS uses `9` as a sentinel for "not available"
- `data_source` can only be `'directory'` or `'cms'` — enforced by a CHECK constraint
- `name`, `facility_type`, and `source_state` are `NOT NULL` — these are required to make a record meaningful

Beyond the main columns, several **derived flags** are pre-computed and stored as simple booleans:

| Flag | What it means |
|------|---------------|
| `has_ratings` | This facility has CMS star ratings |
| `has_staffing_data` | This facility has RN/LPN/CNA hours data |
| `has_service_data` | This facility has service flags (PT, OT, speech, etc.) |
| `has_geo` | This facility has latitude/longitude coordinates |
| `is_geocoded_fallback` | Coordinates came from the geocoding API, not the source data |

These flags exist so the AI agent can immediately filter without re-checking for nulls across a dozen columns. When the agent needs "facilities where I can filter by services," it just asks `WHERE has_service_data = true`.

There are also two additional tables:
- `ingestion_runs` — an audit log of every time the ingestion script ran, recording how many rows were read, written, and rejected and why
- `query_logs` — for logging every AI query the agent makes in production

### The Ingestion Flow: `load_state.py`

The ingestion script is run once per state, with the state code as an argument:

```
uv run python ingestion/load_state.py NC
```

Here is the complete visual flow of what happens for every single row in the CSV:

```
Raw CSV Row (messy, state-specific column names)
         │
         ▼
    apply_crosswalk()
         │
         ├── Reads crosswalk.yaml to find the source column for this state
         │
         ├── [if list of sources] tries each column in order, takes first non-null
         │
         ├── Sends raw value to parse_value() for type cleaning:
         │       • Strips dollar signs, commas, percent signs from numbers
         │       • Converts "Yes/No/T/F/1/0/Checked" to Python True/False
         │       • Converts "Data not available"/"null"/"nan" to Python None
         │       • Parses date strings into actual date objects
         │       • Sanitizes out-of-range ratings (e.g., 9 → None)
         │
         ├── For NC: classifies data_source using exact filename lookup
         │   (Nursing_homes_NC.csv → "cms", health_care_north_carolina.csv → "directory")
         │   For other states: checks if the data_source column starts with "cms"
         │
         ├── Computes derived boolean flags:
         │       • has_ratings = (overall_rating is not None)
         │       • has_staffing_data = (rn_hours_per_resident_day is not None)
         │       • has_service_data = (data_source == "cms")
         │       • has_geo = (latitude and longitude are not None)
         │
         ├── Generates natural_key = "FACILITY_NAME|ADDRESS|STATE"
         │   (used for deduplication on re-runs)
         │
         └── Stores the entire original raw row as a JSON blob in raw_source
             (so no data is ever permanently lost)
         │
         ▼
    Postgres UPSERT
         │
         ├── INSERT INTO facilities (...) VALUES (...)
         │
         └── ON CONFLICT (natural_key) DO UPDATE SET ...
             (if this facility was already loaded, update it instead of duplicating)
         │
         ▼
    conn.transaction() SAVEPOINT
         │
         └── If this row throws a Postgres error (e.g., a bad value slips through),
             only THIS row rolls back. All previous rows in the session are safe.
             The failure is recorded in the rejected list and ingestion_runs table.
```

**Why do we need the `data_source` classification so carefully?**

The NC CSV is a merged file containing rows from both the state health directory AND the CMS federal database. The only reliable way to know which is which is the `SOURCE_FILE` column — it literally names which original file each row came from. We map those filenames to either `"directory"` or `"cms"` using an explicit lookup dictionary. This matters because `has_service_data` is derived from `data_source == "cms"`, and service fields (PT, OT, speech therapy, etc.) are only meaningfully populated for CMS rows. Getting this wrong would silently make every NC service query return no results.

---

## Phase 2 — Geocoding: Getting Coordinates on the Map

Not all facilities have latitude and longitude in the source data. Only CMS Nursing Home records reliably include coordinates. The NC and AZ directory rows have zero coordinates at all, and CO/CA directories have partial coverage.

**Why do we even need coordinates?**

Because a critical class of user query is location-aware: *"Find home health agencies near me"* or *"Nursing homes within 20 miles of downtown Charlotte."* Without coordinates, those queries fail. The Postgres schema even includes a GiST spatial index (`idx_fac_geo`) specifically optimized for radius-based queries using the `earthdistance` extension.

**Why is geocoding a separate script?**

Geocoding is slow. The free Nominatim API (OpenStreetMap) enforces a strict limit of 1 request per 1.5 seconds. If we ran geocoding inline during ingestion, loading NC alone would take over 2.5 hours — and a single network hiccup would crash the entire ingestion run.

Instead, `run_geocode.py` is run independently after ingestion is complete:

1. It queries Postgres for all rows where `has_geo = false` and an address exists
2. For each address, it calls Nominatim with a properly formatted query string (null zip codes are filtered out — not concatenated as the literal word "None")
3. It uses an in-memory address cache within the session, so if two facilities share the same physical building address, the second one gets its coordinates instantly from memory without a second API call
4. Each successful coordinate update is committed to Postgres immediately — the script is naturally resumable. If it crashes at row 3,000, re-running it picks up exactly where it left off because those rows now have `has_geo = true`
5. If 5 consecutive API calls fail, the script hard-stops rather than hammering a potentially rate-limited endpoint

The `is_geocoded_fallback` flag distinguishes between coordinates that came from the original source data versus coordinates that were inferred by us. This matters for downstream data quality audits.

---

## What Comes Next

With Phase 1 and Phase 2 complete, the Postgres database is populated, normalized, and validated. The architecture from here:

**Phase 3 — Qdrant Vector Sync**
Each facility record is converted into a natural language text description (e.g., *"Sunrise Manor is a Nursing Home in Charlotte, NC. Overall Medicare rating: 4 stars. Provides PT, OT, and speech therapy. 0.48 RN hours per resident day."*), embedded using OpenAI's `text-embedding-3-small` model, and stored in a Qdrant vector database alongside the filterable fields as payload. This enables semantic/fuzzy search.

**Phase 4 — LangGraph Agent**
A LangGraph agent classifies incoming user queries into types (exact filter, aggregation, fuzzy search, or hybrid), routes them to the correct tool (SQL query or Qdrant vector search), and synthesizes the results into a grounded natural language response. It never guesses — if a field is null, it says so.

**Phase 5 — Evaluation Harness**
A golden test set of 15-20 hand-verified queries is run after every change to ensure answer quality doesn't regress.

**Phase 6 — FastAPI + Deployment**
The agent is wrapped in a FastAPI app, deployed alongside Neon Postgres connection pooling and Qdrant Cloud, with full LangSmith tracing and structured request logging.
