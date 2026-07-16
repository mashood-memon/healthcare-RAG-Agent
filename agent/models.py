from __future__ import annotations

from typing import Literal, Annotated, Any
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from langgraph.graph import add_messages


# Classification output — the LLM must return this exact shape

VALID_STATES = {"NC", "CO", "AZ", "CA"}

VALID_FACILITY_TYPES = {
    "Nursing Home",
    "Home Health",
    "Hospice",
    "Inpatient Rehabilitation",
    "Healthcare Facility",   # generic catch-all from directory rows
}

VALID_SERVICES = {
    "pt",             # Physical Therapy         → pt_service
    "ot",             # Occupational Therapy     → ot_service
    "speech",         # Speech Therapy           → speech_service
    "iv",             # IV Therapy               → iv_service
    "dme",            # Durable Medical Equip.   → dme_service
    "hospice",        # Hospice services         → hospice_service
    "social_work",    # Social Work              → social_work_service
    "home_health_aide",  # HHA                  → home_health_aide_service
    "nursing_care",   # Skilled Nursing Care     → provides_nursing_care
}

# Maps short service name → DB column name
SERVICE_TO_COLUMN: dict[str, str] = {
    "pt": "pt_service",
    "ot": "ot_service",
    "speech": "speech_service",
    "iv": "iv_service",
    "dme": "dme_service",
    "hospice": "hospice_service",
    "social_work": "social_work_service",
    "home_health_aide": "home_health_aide_service",
    "nursing_care": "provides_nursing_care",
}


class QueryClassification(BaseModel):
    """
    Structured output of the classify_intent node.
    Every downstream tool consumes this — get it right here.
    """
    query_type: Literal["exact_filter", "aggregation", "fuzzy", "hybrid", "clarification", "web_search"] = Field(
        description=(
            "exact_filter: hard filter on known fields (state, rating, services, etc.). "
            "aggregation: compute avg/count/sum/min/max across groups. "
            "fuzzy: purely semantic/descriptive, no hard filters extractable. "
            "hybrid: has BOTH hard filters AND fuzzy/descriptive language that can't be expressed as SQL. "
            "web_search: purely out-of-scope queries (general medical knowledge, or states not in our DB)."
        )
    )
    requires_web_search: bool = Field(
        default=False,
        description=(
            "Set to True if the query asks for facility information not tracked in our database "
            "(e.g. visiting hours, pet policies, prices, reviews) alongside an exact, fuzzy, or hybrid lookup."
        )
    )

    @field_validator("requires_web_search")
    @classmethod
    def validate_augmentation(cls, v: bool, info: ValidationInfo) -> bool:
        if v and info.data.get("query_type") not in {"exact_filter", "fuzzy", "hybrid"}:
            return False  # silently correct rather than reject
        return v
    states: list[str] = Field(
        default_factory=list,
        description=(
            "List of US state codes the query applies to. "
            f"Valid values: {sorted(VALID_STATES)}. "
            "Empty list means 'all states'. "
            "Use multiple states for comparison queries like 'NC vs CO'."
        )
    )
    facility_type: str | None = Field(
        default=None,
        description=(
            "Canonical facility type. "
            f"Valid values: {sorted(VALID_FACILITY_TYPES)}. "
            "Null if not specified or ambiguous."
        )
    )
    min_rating: int | None = Field(
        default=None,
        description=(
            "Minimum overall Medicare star rating (1-5). "
            "Set this for queries like '4+ star', 'highly rated', 'good rating'. "
            "Null if no rating filter is requested."
        )
    )
    required_services: list[str] = Field(
        default_factory=list,
        description=(
            "Services that must be offered. "
            f"Valid values: {sorted(VALID_SERVICES)}. "
            "Use short names: 'speech', 'pt', 'ot', 'iv', 'dme', 'hospice', 'social_work', "
            "'home_health_aide', 'nursing_care'."
        )
    )
    location_text: str | None = Field(
        default=None,
        description=(
            "Raw location description from the user for radius-based search. "
            "Examples: 'near San Diego', 'within 10 miles of Charlotte', 'downtown Phoenix'. "
            "Leave null if the user only mentions a state or county (use states/county fields instead). "
            "Only set this when a specific city or address is mentioned for proximity search."
        )
    )
    radius_miles: float = Field(
        default=25.0,
        description=(
            "Radius in miles for location-based search. "
            "Default 25 miles if not specified. Extract from query if mentioned."
        )
    )
    county: str | None = Field(
        default=None,
        description="County name if the user specifically asks for a county. Null otherwise."
    )
    zip_code: str | None = Field(
        default=None,
        description="ZIP code if the user specifically provides one. Null otherwise."
    )
    aggregation_field: str | None = Field(
        default=None,
        description=(
            "The database column to aggregate. Only set for aggregation queries. "
            "Must be one of the aggregatable columns: rn_hours_per_resident_day, overall_rating, "
            "total_fines_usd, staffing_rating, health_deficiencies_count, bed_count, etc."
        )
    )
    aggregation_op: Literal["avg", "count", "sum", "min", "max"] | None = Field(
        default=None,
        description="Aggregation operation. Required when aggregation_field is set."
    )
    aggregation_group_by: str | None = Field(
        default=None,
        description=(
            "Group-by column for aggregation. "
            "Common values: 'source_state', 'facility_type', 'ownership_type'. "
            "Null if no grouping requested."
        )
    )
    residual_fuzzy_text: str | None = Field(
        default=None,
        description=(
            "The portion of the user's query that CANNOT be expressed as a hard filter — "
            "descriptive, qualitative, or conceptual language. "
            "Examples: 'safe and clean', 'good reputation', 'helped patients recover from stroke', "
            "'family-friendly atmosphere', 'excellent care'. "
            "This text will be embedded and used for semantic vector search. "
            "Null for pure exact_filter and aggregation queries."
        )
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to return. Default 10."
    )


class AgentState(dict):
    query: str                                     # current raw user input
    classification: QueryClassification | None     # output of classify_intent
    tool_result: dict | None                       # output of sql/vector/hybrid tool
    response: str                                  # final synthesized answer
    messages: Annotated[list, add_messages]        # full conversation history (multi-turn)
    geo_cache: dict                                # {location_text: {"lat": float, "lon": float}}
    unsupported_states: list[str]                  # states the user asked for that aren't in our DB
    needs_clarification: bool = False              # whether we need to ask user for more info
    clarification_stage: Literal["location", "facility_type", "services"] | None = None  # what info we need
    pending_classification: QueryClassification | None = None  # partial classification stored during clarification
    web_results: str | None = None
    web_search_source: Literal["tavily", None] = None
    web_search_failed: bool = False
