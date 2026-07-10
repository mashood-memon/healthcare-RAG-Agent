from __future__ import annotations

import os
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range, HasIdCondition
from dotenv import load_dotenv

from agent.models import QueryClassification, AgentState
from agent.tools.sql_tool import RESULT_COLUMNS

load_dotenv()

COLLECTION_NAME = "facilities_v1"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 10  # default result cap for vector search


# ---------------------------------------------------------------------------
# Shared client helpers
# ---------------------------------------------------------------------------

def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )


def embed_query(text: str) -> list[float]:
    """Embed a single query string. Only called with residual_fuzzy_text — not the raw user query."""
    client = get_openai_client()
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Build Qdrant hard filters from a QueryClassification
# ---------------------------------------------------------------------------

def build_qdrant_filter(c: QueryClassification, candidate_ids: list[str] | None = None) -> Filter | None:
    """
    Convert classification hard filters into a Qdrant Filter object.
    candidate_ids — if provided, restrict search to these facility_id strings (for hybrid mode).
    """
    must_conditions = []

    # State filter
    if c.states:
        must_conditions.append(
            FieldCondition(key="source_state", match=MatchAny(any=c.states))
        )

    # Facility type filter
    if c.facility_type:
        must_conditions.append(
            FieldCondition(key="facility_type", match=MatchValue(value=c.facility_type))
        )

    # Minimum rating filter
    if c.min_rating is not None:
        must_conditions.append(
            FieldCondition(key="overall_rating", range=Range(gte=c.min_rating))
        )

    # Candidate ID restriction (hybrid mode only)
    if candidate_ids:
        must_conditions.append(
            HasIdCondition(has_id=candidate_ids)
        )

    if not must_conditions:
        return None

    return Filter(must=must_conditions)


# ---------------------------------------------------------------------------
# Format Qdrant hits into a standard result dict
# ---------------------------------------------------------------------------

def format_hits(hits) -> list[dict]:
    """Convert Qdrant ScoredPoint list into plain dicts with score included."""
    results = []
    for hit in hits:
        row = dict(hit.payload)
        row["facility_id"] = str(hit.id)
        row["_similarity_score"] = round(hit.score, 4)
        results.append(row)
    return results


def enrich_from_postgres(rows: list[dict]) -> list[dict]:
    """
    Take vector search results (which only have Qdrant payload fields) and
    enrich them with the full Postgres record (address, phone, etc.).
    This avoids duplicating data in Qdrant's payload.
    """
    if not rows:
        return rows

    facility_ids = [r["facility_id"] for r in rows if r.get("facility_id")]
    if not facility_ids:
        return rows

    db_url = os.getenv("DATABASE_URL")
    col_str = ", ".join(RESULT_COLUMNS)
    placeholders = ", ".join(f"%(id_{i})s" for i in range(len(facility_ids)))
    sql = f"SELECT {col_str} FROM facilities WHERE facility_id::text IN ({placeholders})"
    params = {f"id_{i}": fid for i, fid in enumerate(facility_ids)}

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            pg_rows = {str(r["facility_id"]): dict(r) for r in cur.fetchall()}

    # Merge: Postgres data is the base, Qdrant score is added on top
    enriched = []
    for row in rows:
        fid = row.get("facility_id")
        if fid and fid in pg_rows:
            merged = pg_rows[fid]
            merged["_similarity_score"] = row.get("_similarity_score", 0)
            enriched.append(merged)
        else:
            enriched.append(row)

    return enriched


# ---------------------------------------------------------------------------
# LangGraph node: vector_tool
# Used for query_type == "fuzzy"
# ---------------------------------------------------------------------------

def vector_tool(state: AgentState) -> dict:
    """
    Pure semantic search against Qdrant.

    Rules:
    - Only embeds residual_fuzzy_text, NOT the raw user query.
    - Hard filters (state, facility_type, min_rating) go into Qdrant Filter, not the embedding.
    - Falls back to embedding the raw query if residual_fuzzy_text is None (e.g., name lookup).
    """
    c: QueryClassification = state["classification"]

    # What to embed — prefer the focused fuzzy portion
    query_text = c.residual_fuzzy_text or state["query"]

    print(f"  [vector] Embedding: '{query_text[:80]}...' " if len(query_text) > 80 else f"  [vector] Embedding: '{query_text}'")

    vector = embed_query(query_text)
    qdrant = get_qdrant_client()
    qdrant_filter = build_qdrant_filter(c)

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
        zero_reason = "no_facilities_match_criteria — Vector search returned no results with the applied filters."

    return {
        "tool_result": {
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows,
            "row_count": row_count,
            "zero_reason": zero_reason,
            "error": None,
            "search_mode": "vector",
            "embedded_text": query_text,
        }
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agent.models import QueryClassification

    TEST_CASES = [
        (
            "Safe, clean nursing home for grandmother who had a stroke",
            QueryClassification(
                query_type="fuzzy",
                residual_fuzzy_text="safe clean place grandmother stroke recovery rehabilitation",
            )
        ),
        (
            "Places with low staff turnover and caring teams in NC",
            QueryClassification(
                query_type="fuzzy",
                states=["NC"],
                residual_fuzzy_text="low staff turnover, caring compassionate teams",
            )
        ),
        (
            "Tell me about Sunrise Manor",
            QueryClassification(
                query_type="fuzzy",
                residual_fuzzy_text="Tell me about Sunrise Manor",
            )
        ),
        (
            "Best home health in California",
            QueryClassification(
                query_type="fuzzy",
                states=["CA"],
                facility_type="Home Health",
                residual_fuzzy_text="best home health outstanding quality care",
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
            result = vector_tool(state)
            tr = result["tool_result"]
            print(f"ROWS RETURNED: {tr['row_count']}")
            if tr["zero_reason"]:
                print(f"ZERO REASON: {tr['zero_reason']}")
            for i, row in enumerate(tr["rows"][:3], 1):
                name = row.get("name", "?")
                city = row.get("city", "?")
                state_code = row.get("source_state", "?")
                score = row.get("_similarity_score", 0)
                rating = row.get("overall_rating", "N/A")
                print(f"  [{i}] {name} — {city}, {state_code} | rating={rating} | score={score}")
        except Exception as e:
            print(f"ERROR: {e}")
