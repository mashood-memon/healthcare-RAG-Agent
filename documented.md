# Healthcare Agent Project Overview

## Project Vision
The goal of this project is to build an intelligent, RAG-powered (Retrieval-Augmented Generation) Healthcare Agent. This agent will allow users to query healthcare facility data (Nursing Homes, Hospices, Home Health, etc.) across multiple states. 

Before the AI can answer natural language queries, we must solve a massive data engineering problem: harmonizing incredibly messy, disjointed, and structurally completely different CSV datasets from 4 states (North Carolina, Colorado, Arizona, California) and the federal CMS (Centers for Medicare & Medicaid Services).

## What Has Been Done So Far

We have successfully completed **Phase 1 (Setup & Mapping)** and **Phase 2 (Ingestion & Normalization)**.

### 1. The Crosswalk Architecture
Instead of writing 4 different scripts to parse 4 different states, we built a single "Rosetta Stone" called the **Crosswalk**.
* **`crosswalk.csv`**: A human-readable matrix that maps how 83 standardized "canonical" fields map to the unique column names in each state's raw CSV file.
* **`crosswalk.py`**: A build script that compiles the CSV into a clean, machine-readable **`crosswalk.yaml`**. This YAML acts as the source of truth for all ingestion logic.

### 2. The Database Schema (`schema.sql`)
We designed a highly strict Postgres schema (`facilities` table). We explicitly typed fields (UUIDs, BOOLEANs, SMALLINTs) and added strict integrity checks (e.g., `CHECK (overall_rating BETWEEN 1 AND 5)`). 

If dirty data attempts to enter the database, Postgres rejects it. We also added an `ingestion_runs` table to maintain a permanent audit trail of exactly how many rows successfully ingested vs failed.

---

## Deep Dive: The Ingestion Flow

The actual ingestion process is split across two robust scripts to ensure speed and fault tolerance.

### Script A: `load_state.py` (The Heavy Lifter)
This script is responsible for mapping and cleaning the messy CSV data and upserting it into Postgres. It processes thousands of rows per second.

1. **`main()` - The Orchestrator:**
   * Reads the target state from the command line (e.g., `NC`).
   * Opens a new audit log in the `ingestion_runs` table.
   * Loads the raw CSV into memory using `pandas`, treating all columns as strings to prevent automatic type-guessing errors.
   * Iterates through every row, passing it to `apply_crosswalk()`.
   * **Safe Upserts:** Executes the Postgres `UPSERT` using a `with conn.transaction():` context block. This creates a nested **Savepoint**. If row #157 is corrupted, only row #157 rolls back, preserving the previous 156 successful inserts.

2. **`apply_crosswalk()` - The Brain:**
   * Looks at the `crosswalk.yaml` and iterates over our 83 target fields.
   * Extracts the correct raw value from the row based on the state's mapping.
   * Dynamically calculates critical flags (`has_ratings`, `has_service_data`) so the AI doesn't have to guess later.
   * Serializes the *original, un-modified row* into a JSON string and stores it in the `raw_source` column so no historical data is ever permanently lost.

3. **`parse_value()` - The Data Janitor:**
   * Raw data is notoriously dirty. Ratings might be "9" (meaning N/A), booleans might be "Yes/No", and currency might contain "$". 
   * This function scrubs the string, handles edge cases (like "Data not available"), and strictly casts the value into the correct Python type (`int`, `float`, `bool`) that Postgres demands.

4. **`normalize_key()` - The Deduplicator:**
   * Creates a composite string: `FACILITY_NAME|ADDRESS|STATE_CODE`.
   * This `natural_key` prevents duplicate entries. If a state publishes an updated CSV tomorrow, the script will use this key to `UPDATE` the existing row rather than blindly inserting duplicates.

### Script B: `run_geocode.py` (The Map Maker)
Geocoding (converting an address into Latitude/Longitude) requires pinging the free `Nominatim` API. Because they enforce a strict limit of 1 request per 1.5 seconds, geocoding inline would stall the `load_state.py` script for hours. 

Instead, we decoupled it entirely:
1. `run_geocode.py` queries Postgres for all valid addresses `WHERE has_geo = false`.
2. It hits the Nominatim API to fetch coordinates.
3. It uses an **In-Memory Cache** (`address_cache`) to deduplicate identical addresses (like a hospital sharing a building with a hospice) to drastically reduce the number of API calls needed.
4. It updates the Postgres row directly (`UPDATE facilities SET latitude...`) and includes fail-safes to hard-stop the script if the API begins blocking us (e.g., 5 consecutive HTTP 403 errors).

---

## What Comes Next (Phase 3)
With the canonical SQL data securely loaded in Postgres, the next step is **Vector Search Synchronization**. We will generate rich, natural-language text summaries for each facility and embed them into our **Qdrant** vector database, allowing the AI agent to instantly perform semantic geographic searches.
