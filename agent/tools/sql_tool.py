from __future__ import annotations

import json
import os
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI
from dotenv import load_dotenv

from agent.models import QueryClassification, AgentState, SERVICE_TO_COLUMN

load_dotenv()

# ---------------------------------------------------------------------------
# Column whitelists — validated in code, not just prompted.
# If the LLM hallucinates a column name, we raise immediately.
# ---------------------------------------------------------------------------

FILTERABLE_COLUMNS = {
    "source_state", "facility_type", "name", "city", "county", "zip",
    "overall_rating", "health_inspection_rating", "staffing_rating", "quality_measure_rating",
    "pt_service", "ot_service", "speech_service", "iv_service", "dme_service",
    "hospice_service", "social_work_service", "home_health_aide_service", "provides_nursing_care",
    "ownership_type", "bed_count", "abuse_complaint", "special_focus_facility",
    "ccrc_flag", "has_ratings", "has_staffing_data", "has_service_data", "has_geo",
    "staff_stability", "staffing_level_assessment",
    "hospital_readmission_flag", "home_discharge_flag",
}

AGGREGATABLE_COLUMNS = {
    "overall_rating", "health_inspection_rating", "staffing_rating", "quality_measure_rating",
    "rn_hours_per_resident_day", "lpn_hours_per_resident_day", "cna_hours_per_resident_day",
    "total_nursing_hours_per_resident_day", "rn_turnover_pct", "total_nursing_turnover_pct",
    "total_fines_usd", "number_of_fines", "total_penalties",
    "improved_walking_mobility_pct", "improved_bathing_ability_pct", "falls_major_injury_pct",
    "improved_breathing_pct", "improved_getting_out_of_bed_pct",
    "bed_count", "health_deficiencies_count", "infection_control_citations",
    "pt_hours_per_resident_day", "administrators_left_12mo",
    "weighted_health_inspection_score", "health_deficiency_severity_score",
    "medicare_cost_vs_national_avg", "avoidable_hospitalizations_pct",
    "started_care_on_time_pct", "medication_issues_fixed_on_time_pct",
    "improved_taking_medications_pct", "functional_ability_discharge_score",
    "info_shared_with_doctor_pct", "info_shared_with_family_pct",
}

ALLOWED_AGG_OPS = {"avg", "count", "sum", "min", "max"}

# Columns returned for exact_filter queries — omit heavy fields like raw_source
RESULT_COLUMNS = (
    "facility_id", "name", "facility_type", "data_source", "source_state",
    "city", "county", "zip", "address", "phone",
    "overall_rating", "health_inspection_rating", "staffing_rating", "quality_measure_rating",
    "has_ratings", "has_staffing_data", "has_service_data", "has_geo",
    "pt_service", "ot_service", "speech_service", "iv_service", "dme_service",
    "hospice_service", "social_work_service", "home_health_aide_service", "provides_nursing_care",
    "rn_hours_per_resident_day", "total_nursing_hours_per_resident_day",
    "total_fines_usd", "number_of_fines", "abuse_complaint", "special_focus_facility",
    "ownership_type", "chain_affiliation", "bed_count",
    "improved_walking_mobility_pct", "improved_bathing_ability_pct", "falls_major_injury_pct",
    "hospital_readmission_flag", "home_discharge_flag",
    "latitude", "longitude",
)


# ---------------------------------------------------------------------------
# Geocoding via OpenAI function calling
# ---------------------------------------------------------------------------

_GEOCODE_TOOL = {
    "type": "function",
    "function": {
        "name": "return_coordinates",
        "description": "Return the latitude and longitude for a given location description.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Decimal latitude of the location."
                },
                "longitude": {
                    "type": "number",
                    "description": "Decimal longitude of the location."
                },
                "resolved_name": {
                    "type": "string",
                    "description": "The canonical place name that was resolved, e.g. 'San Diego, CA'."
                }
            },
            "required": ["latitude", "longitude", "resolved_name"]
        }
    }
}


def geocode_location(location_text: str, geo_cache: dict) -> tuple[float, float] | None:
    """
    Convert a free-text location to (lat, lon) using OpenAI function calling.
    Results are cached in geo_cache to avoid duplicate API calls within a session.
    Returns None if geocoding fails.
    """
    cache_key = location_text.lower().strip()
    if cache_key in geo_cache:
        cached = geo_cache[cache_key]
        return cached["lat"], cached["lon"]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a geocoding assistant. Given a location description, "
                        "return its latitude and longitude using the return_coordinates function. "
                        "Focus on the US. If you cannot confidently determine coordinates, "
                        "return the closest major city in the described area."
                    )
                },
                {
                    "role": "user",
                    "content": f"What are the coordinates for: {location_text}"
                }
            ],
            tools=[_GEOCODE_TOOL],
            tool_choice={"type": "function", "function": {"name": "return_coordinates"}},
            temperature=0,
        )

        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        lat = float(args["latitude"])
        lon = float(args["longitude"])
        resolved = args.get("resolved_name", location_text)

        geo_cache[cache_key] = {"lat": lat, "lon": lon}
        print(f"  [geocode] '{location_text}' → {resolved} ({lat:.4f}, {lon:.4f})")
        return lat, lon

    except Exception as e:
        print(f"  [geocode] Warning: failed to geocode '{location_text}': {e}")
        return None


# ---------------------------------------------------------------------------
# Query builder — never interpolates raw LLM strings into SQL
# ---------------------------------------------------------------------------

def build_query(c: QueryClassification, geo_cache: dict) -> tuple[str, dict]:
    """
    Build a parameterized SQL query from a QueryClassification.

    Returns (sql_string, params_dict).
    All column names are validated against the whitelist before use.
    Raw LLM string outputs are NEVER interpolated into SQL — only used as param values.
    """
    conditions = ["1=1"]
    params: dict = {}

    # --- State filter ---
    if c.states:
        placeholders = ", ".join(f"%(state_{i})s" for i in range(len(c.states)))
        conditions.append(f"source_state IN ({placeholders})")
        for i, s in enumerate(c.states):
            params[f"state_{i}"] = s

    # --- Facility type filter ---
    if c.facility_type:
        conditions.append("facility_type = %(facility_type)s")
        params["facility_type"] = c.facility_type

    # --- Rating filter ---
    if c.min_rating is not None:
        conditions.append("has_ratings = true AND overall_rating >= %(min_rating)s")
        params["min_rating"] = c.min_rating

    # --- Service filters ---
    if c.required_services:
        # Guard emitted ONCE — not once per service (prevents duplicate conditions)
        conditions.append("has_service_data = true")
        for svc in c.required_services:
            col = SERVICE_TO_COLUMN.get(svc)
            if col is None:
                raise ValueError(f"Unknown service short name: '{svc}'. Valid: {list(SERVICE_TO_COLUMN)}")
            # col comes from a hardcoded dict, never from LLM output — injection-safe
            assert col in FILTERABLE_COLUMNS, f"Service column '{col}' not in whitelist"
            conditions.append(f"{col} = true")


    # --- County filter ---
    if c.county:
        conditions.append("LOWER(county) = LOWER(%(county)s)")
        params["county"] = c.county

    # --- ZIP filter ---
    if c.zip_code:
        conditions.append("zip = %(zip_code)s")
        params["zip_code"] = c.zip_code

    # --- Location / radius filter (earthdistance) ---
    geo_center = None
    if c.location_text:
        coords = geocode_location(c.location_text, geo_cache)
        if coords:
            lat, lon = coords
            geo_center = (lat, lon)
            radius_meters = c.radius_miles * 1609.34
            conditions.append(
                "has_geo = true AND "
                "earth_distance(ll_to_earth(latitude, longitude), "
                "ll_to_earth(%(geo_lat)s, %(geo_lon)s)) <= %(geo_radius_m)s"
            )
            params["geo_lat"] = lat
            params["geo_lon"] = lon
            params["geo_radius_m"] = radius_meters

    where = " AND ".join(conditions)

    # --- Aggregation query ---
    if c.query_type == "aggregation" and (c.aggregation_field or c.aggregation_op == "count"):
        agg_field = c.aggregation_field
        agg_op = (c.aggregation_op or "avg").lower()

        # Hard guard — never trust LLM-provided field/op names without validation
        if agg_field and agg_field not in AGGREGATABLE_COLUMNS:
            raise ValueError(
                f"aggregation_field '{agg_field}' is not in the allowed aggregatable columns. "
                f"Valid columns: {sorted(AGGREGATABLE_COLUMNS)}"
            )
        if agg_op not in ALLOWED_AGG_OPS:
            raise ValueError(
                f"aggregation_op '{agg_op}' is not allowed. Valid: {ALLOWED_AGG_OPS}"
            )

        if agg_op == "count":
            # COUNT(*) — no need for a specific column, always valid
            select_agg = "COUNT(*) AS result"
        else:
            # agg_field is validated against whitelist — safe to interpolate
            select_agg = f"{agg_op.upper()}({agg_field}) AS result"

        if c.aggregation_group_by:
            group_col = c.aggregation_group_by
            if group_col not in FILTERABLE_COLUMNS and group_col not in AGGREGATABLE_COLUMNS:
                raise ValueError(f"aggregation_group_by '{group_col}' not in allowed columns")
            # group_col is validated — safe to interpolate
            sql = (
                f"SELECT {group_col}, {select_agg}, COUNT(*) AS n "
                f"FROM facilities "
                f"WHERE {where}"
            )
            if agg_op != "count":
                sql += f" AND {agg_field} IS NOT NULL"
            sql += f" GROUP BY {group_col} ORDER BY result DESC"
        else:
            sql = (
                f"SELECT {select_agg}, COUNT(*) AS n "
                f"FROM facilities WHERE {where}"
            )
            if agg_op != "count":
                sql += f" AND {agg_field} IS NOT NULL"

        return sql, params

    # --- Exact filter / hybrid query (returns full rows) ---
    col_str = ", ".join(RESULT_COLUMNS)
    limit = min(c.limit, 50)  # hard cap — never return more than 50 rows

    order_by = "overall_rating DESC NULLS LAST"
    if geo_center:
        # Sort by distance when a geo filter is active
        lat, lon = geo_center
        order_by = (
            f"earth_distance(ll_to_earth(latitude, longitude), "
            f"ll_to_earth({lat}, {lon})) ASC"
        )

    sql = (
        f"SELECT {col_str} FROM facilities "
        f"WHERE {where} "
        f"ORDER BY {order_by} "
        f"LIMIT {limit}"
    )
    return sql, params


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def run_query(sql: str, params: dict) -> list[dict]:
    """Execute the query and return rows as a list of dicts."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL not set in .env")

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def determine_zero_reason(c: QueryClassification) -> str:
    """
    When a query returns zero rows, produce a helpful explanation of why.
    Distinguishes between 'filters too strict' vs 'data not tracked for this type'.
    """
    # Service data is only tracked for CMS-sourced facilities
    if c.required_services and c.facility_type in ("Nursing Home", "Home Health", "Hospice"):
        state_coverage = {
            "NC": "partial (CMS rows only)",
            "CO": "partial (CMS rows only)",
            "AZ": "partial (CMS rows only)",
            "CA": "partial (CMS rows only)",
        }
        state_note = ", ".join(
            f"{s}: {state_coverage.get(s, 'unknown')}" for s in c.states
        ) if c.states else "all states: CMS rows only"
        return (
            f"no_data_for_field — Service data (PT, OT, speech, etc.) is only tracked "
            f"for CMS-certified facilities. Coverage: {state_note}. "
            f"Try removing the service filter or search without it."
        )

    if c.min_rating and c.states:
        return (
            f"no_facilities_match_criteria — No facilities in {c.states} "
            f"match the combined filters. Try lowering the minimum rating or broadening the search."
        )

    return "no_facilities_match_criteria — No facilities match the specified filters."


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def sql_tool(state: AgentState) -> dict:
    """
    LangGraph node: sql_tool.
    Reads state["classification"] and state["geo_cache"].
    Writes state["tool_result"].
    """
    c: QueryClassification = state["classification"]
    geo_cache: dict = state.get("geo_cache", {})

    try:
        sql, params = build_query(c, geo_cache)
    except ValueError as e:
        return {
            "tool_result": {
                "columns": [],
                "rows": [],
                "sql": "",
                "row_count": 0,
                "error": str(e),
                "zero_reason": "invalid_query",
            },
            "geo_cache": geo_cache,
        }

    rows = run_query(sql, params)
    row_count = len(rows)

    zero_reason = None
    if row_count == 0:
        zero_reason = determine_zero_reason(c)

    columns = list(rows[0].keys()) if rows else []

    return {
        "tool_result": {
            "columns": columns,
            "rows": rows,
            "sql": sql,
            "row_count": row_count,
            "zero_reason": zero_reason,
            "error": None,
        },
        "geo_cache": geo_cache,
    }


# ---------------------------------------------------------------------------
# Standalone test helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agent.models import QueryClassification

    TEST_CASES = [
        (
            "Nursing homes in NC with 4+ star rating and speech therapy",
            QueryClassification(
                query_type="exact_filter",
                states=["NC"],
                facility_type="Nursing Home",
                min_rating=4,
                required_services=["speech"],
            )
        ),
        (
            "Average RN hours per resident day for nursing homes in NC vs CO",
            QueryClassification(
                query_type="aggregation",
                states=["NC", "CO"],
                facility_type="Nursing Home",
                aggregation_field="rn_hours_per_resident_day",
                aggregation_op="avg",
                aggregation_group_by="source_state",
            )
        ),
        (
            "How many home health agencies are in California?",
            QueryClassification(
                query_type="aggregation",
                states=["CA"],
                facility_type="Home Health",
                aggregation_field="rn_hours_per_resident_day",
                aggregation_op="count",
            )
        ),
        (
            "Home health near San Diego with PT and OT",
            QueryClassification(
                query_type="exact_filter",
                facility_type="Home Health",
                required_services=["pt", "ot"],
                location_text="San Diego, CA",
                radius_miles=25.0,
            )
        ),
    ]

    geo_cache: dict = {}

    for label, classification in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"QUERY: {label}")
        try:
            sql, params = build_query(classification, geo_cache)
            print(f"SQL:\n  {sql}")
            print(f"PARAMS: {params}")
            rows = run_query(sql, params)
            print(f"ROWS RETURNED: {len(rows)}")
            if rows:
                first = rows[0]
                print(f"FIRST ROW: {first.get('name')} — {first.get('city')}, {first.get('source_state')}")
                if "result" in first:
                    print(f"AGG RESULT: {first}")
        except Exception as e:
            print(f"ERROR: {e}")
