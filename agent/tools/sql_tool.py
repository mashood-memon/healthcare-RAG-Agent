from __future__ import annotations

import json
import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

from agent.models import QueryClassification, AgentState, SERVICE_TO_COLUMN
from agent.utils import geocode_location

load_dotenv()

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
    "rn_hours_per_resident_day", "lpn_hours_per_resident_day", "cna_hours_per_resident_day", "total_nursing_hours_per_resident_day",
    "rn_turnover_pct", "total_nursing_turnover_pct", "staff_stability", "staffing_level_assessment",
    "total_fines_usd", "number_of_fines", "total_penalties", "abuse_complaint", "special_focus_facility",
    "ownership_type", "chain_affiliation", "bed_count",
    "improved_walking_mobility_pct", "improved_bathing_ability_pct", "falls_major_injury_pct",
    "improved_breathing_pct", "improved_getting_out_of_bed_pct", "health_deficiencies_count", "infection_control_citations",
    "pt_hours_per_resident_day", "administrators_left_12mo", "weighted_health_inspection_score", "health_deficiency_severity_score",
    "medicare_cost_vs_national_avg", "avoidable_hospitalizations_pct", "started_care_on_time_pct",
    "medication_issues_fixed_on_time_pct", "improved_taking_medications_pct", "functional_ability_discharge_score",
    "info_shared_with_doctor_pct", "info_shared_with_family_pct",
    "hospital_readmission_flag", "home_discharge_flag",
    "latitude", "longitude",
)

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
        geo = geocode_location(c.location_text, geo_cache)
        if geo:
            lat, lon = geo["lat"], geo["lon"]
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
    """
    # Build a list of filters that were active, for a useful error message
    active_filters = []
    if c.facility_type:
        active_filters.append(f"facility_type={c.facility_type}")
    if c.min_rating is not None:
        active_filters.append(f"min_rating={c.min_rating}")
    if c.required_services:
        active_filters.append(f"services={c.required_services}")
    if c.states:
        active_filters.append(f"states={c.states}")
    if c.location_text:
        active_filters.append(f"near={c.location_text}")

    filter_str = ", ".join(active_filters) if active_filters else "current filters"
    return (
        f"no_facilities_match_criteria — No facilities match the {filter_str}. "
        f"Try removing one or more filters (e.g. drop the rating requirement, "
        f"widen the radius, or remove the service requirement)."
    )


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


