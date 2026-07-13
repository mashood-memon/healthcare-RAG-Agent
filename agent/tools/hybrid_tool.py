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
    enrich_from_postgres,
    COLLECTION_NAME,
)

load_dotenv()

# How many SQL candidates to pass into Qdrant for reranking.
# Large enough to give Qdrant a meaningful pool, small enough to stay fast.
SQL_CANDIDATE_LIMIT = 200


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
    rows = enrich_from_postgres(rows)
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


