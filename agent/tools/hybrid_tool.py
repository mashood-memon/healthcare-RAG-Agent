from __future__ import annotations

import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

from agent.models import QueryClassification, AgentState
from agent.tools.vector_tool import (
    get_qdrant_client,
    embed_query,
    build_qdrant_filter,
    format_hits,
    COLLECTION_NAME,
)

load_dotenv()

# How many SQL candidates to pass into Qdrant for reranking.
# Large enough to give Qdrant a meaningful pool, small enough to stay fast.
SQL_CANDIDATE_LIMIT = 200


# ---------------------------------------------------------------------------
# SQL candidate fetch — returns just facility_id strings
# ---------------------------------------------------------------------------

def fetch_candidate_ids(c: QueryClassification) -> list[str]:
    """
    Run the hard-filter SQL to get a list of candidate facility_ids.
    These are passed into Qdrant as a payload filter so vector search
    only reranks within the SQL-filtered subset.
    """
    from agent.tools.sql_tool import build_query, RESULT_COLUMNS
    from agent.models import QueryClassification as QC

    # Build a modified classification for the SQL pass:
    # - Always exact_filter (aggregation makes no sense in hybrid)
    # - No fuzzy text (that's Qdrant's job)
    # - Higher limit to give Qdrant a good candidate pool
    sql_classification = QC(
        query_type="exact_filter",
        states=c.states,
        facility_type=c.facility_type,
        min_rating=c.min_rating,
        required_services=c.required_services,
        county=c.county,
        zip_code=c.zip_code,
        location_text=c.location_text,
        radius_miles=c.radius_miles,
        limit=SQL_CANDIDATE_LIMIT,
        # residual_fuzzy_text intentionally excluded — Qdrant handles that
    )

    geo_cache: dict = {}
    sql, params = build_query(sql_classification, geo_cache)

    # Override the SELECT to fetch only facility_id — saves bandwidth
    id_sql = sql.replace(
        f"SELECT {', '.join(RESULT_COLUMNS)}",
        "SELECT facility_id::text",
        1,  # only replace first occurrence
    )

    db_url = os.getenv("DATABASE_URL")
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(id_sql, params)
            rows = cur.fetchall()

    return [str(r["facility_id"]) for r in rows]


# ---------------------------------------------------------------------------
# LangGraph node: hybrid_tool
# Used for query_type == "hybrid"
# ---------------------------------------------------------------------------

def hybrid_tool(state: AgentState) -> dict:
    """
    SQL-first, then vector rerank.

    Flow:
      1. Run SQL with hard filters to shrink the candidate pool (up to 200 IDs)
      2. If zero candidates from SQL → fall back to pure vector search (no SQL restriction)
      3. Embed residual_fuzzy_text
      4. Run Qdrant search restricted to candidate IDs + hard filters
      5. Return top-K reranked by similarity score
    """
    c: QueryClassification = state["classification"]

    # Step 1: SQL pass — get candidate IDs from hard filters
    print(f"  [hybrid] Running SQL candidate fetch...")
    try:
        candidate_ids = fetch_candidate_ids(c)
        print(f"  [hybrid] SQL returned {len(candidate_ids)} candidates.")
    except Exception as e:
        print(f"  [hybrid] SQL candidate fetch failed: {e}. Falling back to pure vector search.")
        candidate_ids = []

    # Step 2: If SQL found nothing, skip ID restriction (still apply field-level filters)
    if len(candidate_ids) == 0:
        print(f"  [hybrid] No SQL candidates — running vector search without ID restriction.")
        qdrant_filter = build_qdrant_filter(c, candidate_ids=None)
        zero_sql = True
    else:
        qdrant_filter = build_qdrant_filter(c, candidate_ids=candidate_ids)
        zero_sql = False

    # Step 3: Embed the fuzzy portion only
    query_text = c.residual_fuzzy_text or state["query"]
    print(f"  [hybrid] Embedding: '{query_text[:80]}'" if len(query_text) > 80 else f"  [hybrid] Embedding: '{query_text}'")

    vector = embed_query(query_text)

    # Step 4: Qdrant rerank within candidates
    qdrant = get_qdrant_client()

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=qdrant_filter,
        limit=c.limit,
        with_payload=True,
    )
    hits = response.points

    rows = format_hits(hits)
    row_count = len(rows)

    zero_reason = None
    if row_count == 0:
        if zero_sql:
            zero_reason = (
                "no_facilities_match_criteria — The hard filters (state, rating, services) "
                "found no matching facilities, and the semantic search also returned nothing. "
                "Try broadening the filters."
            )
        else:
            zero_reason = (
                "no_facilities_match_criteria — SQL found candidates matching the hard filters, "
                "but none were semantically relevant to the descriptive part of the query."
            )

    return {
        "tool_result": {
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows,
            "row_count": row_count,
            "zero_reason": zero_reason,
            "error": None,
            "search_mode": "hybrid",
            "embedded_text": query_text,
            "sql_candidate_count": len(candidate_ids),
        }
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agent.models import QueryClassification

    TEST_CASES = [
        (
            "Home health agencies in San Diego with good walking mobility outcomes",
            QueryClassification(
                query_type="hybrid",
                facility_type="Home Health",
                location_text="San Diego, CA",
                radius_miles=25.0,
                residual_fuzzy_text="good walking mobility outcomes",
                limit=5,
            )
        ),
        (
            "PT and OT in Colorado somewhere with good staff",
            QueryClassification(
                query_type="hybrid",
                states=["CO"],
                required_services=["pt", "ot"],
                residual_fuzzy_text="somewhere with good compassionate staff",
                limit=5,
            )
        ),
        (
            "Non-profit nursing homes in NC with warm family-friendly atmosphere",
            QueryClassification(
                query_type="hybrid",
                states=["NC"],
                facility_type="Nursing Home",
                residual_fuzzy_text="warm family-friendly atmosphere non-profit",
                limit=5,
            )
        ),
        (
            "Nursing homes near downtown Charlotte with good ratings",
            QueryClassification(
                query_type="hybrid",
                states=["NC"],
                facility_type="Nursing Home",
                location_text="downtown Charlotte, NC",
                radius_miles=25.0,
                residual_fuzzy_text="good ratings excellent care",
                limit=5,
            )
        ),
    ]

    for label, classification in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"QUERY: {label}")
        state: AgentState = {
            "query": label,
            "classification": classification,
            "tool_result": None,
            "response": "",
            "messages": [],
            "geo_cache": {},
        }
        try:
            result = hybrid_tool(state)
            tr = result["tool_result"]
            print(f"SQL candidates: {tr['sql_candidate_count']}  |  Final results: {tr['row_count']}")
            if tr["zero_reason"]:
                print(f"ZERO REASON: {tr['zero_reason']}")
            for i, row in enumerate(tr["rows"], 1):
                name = row.get("name", "?")
                city = row.get("city", "?")
                state_code = row.get("source_state", "?")
                score = row.get("_similarity_score", 0)
                rating = row.get("overall_rating", "N/A")
                print(f"  [{i}] {name} — {city}, {state_code} | rating={rating} | score={score}")
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
