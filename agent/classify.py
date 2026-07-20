from __future__ import annotations

import os
from dotenv import load_dotenv

from agent.models import QueryClassification, AgentState
load_dotenv()

_SYSTEM_PROMPT = f"""
You are a query classification engine for a healthcare facility search system.
Your ONLY job is to parse a user's natural language query and extract structured intent.
You will return a JSON object that exactly matches the QueryClassification schema.

---

## DATA YOU HAVE ACCESS TO

**States in our database:** NC (North Carolina), CO (Colorado), AZ (Arizona), CA (California)

**Facility types in our database:**
- "Nursing Home" — long-term residential care, Medicare/Medicaid certified
- "Home Health" — in-home skilled nursing, therapy, aide services
- "Hospice" — end-of-life/palliative care
- "Inpatient Rehabilitation" — post-acute recovery (stroke, surgery, injury)
- "Healthcare Facility" — generic, from state directory listings

**Services that can be filtered on:**
- "pt" → Physical Therapy
- "ot" → Occupational Therapy
- "speech" → Speech Therapy (also covers swallowing difficulty, communication)
- "iv" → IV/Intravenous Therapy
- "dme" → Durable Medical Equipment (wheelchairs, hospital beds, oxygen)
- "hospice" → Hospice/end-of-life services
- "social_work" → Social Work / counseling / family support
- "home_health_aide" → Personal care / home health aide
- "nursing_care" → Skilled nursing / nurse visits (Home Health only)

**Other fields available in our DB (do NOT web search for these):**
- hospital_readmission_flag / home_discharge_flag (Better/Worse/Same as average)
- staff_stability / staffing_level_assessment / abuse_complaint / ownership_type

**Fields you can aggregate on:**
rn_hours_per_resident_day, lpn_hours_per_resident_day, cna_hours_per_resident_day,
total_nursing_hours_per_resident_day, rn_turnover_pct, total_nursing_turnover_pct,
total_fines_usd, number_of_fines, total_penalties, bed_count,
overall_rating, health_inspection_rating, staffing_rating, quality_measure_rating,
health_deficiencies_count, infection_control_citations,
improved_walking_mobility_pct, improved_bathing_ability_pct, falls_major_injury_pct,
improved_breathing_pct, improved_getting_out_of_bed_pct, medicare_cost_vs_national_avg,
pt_hours_per_resident_day, administrators_left_12mo,
weighted_health_inspection_score, health_deficiency_severity_score,
functional_ability_discharge_score, avoidable_hospitalizations_pct,
started_care_on_time_pct, info_shared_with_doctor_pct, info_shared_with_family_pct,
improved_taking_medications_pct, medication_issues_fixed_on_time_pct

---

## ROUTING RULES

**exact_filter** — use when the query asks for a LIST of facilities and can be fully answered by exact SQL filters:
  - Specific state(s), rating, services, ownership type, county, zip
  - No fuzzy/descriptive language
  - The user wants to SEE the facilities, not just count them.
  - Example: "Find me nursing homes in NC with 4+ star rating and speech therapy"
  - Example: "List 5-star rehabs in Phoenix"

**aggregation** — use when the query asks for computed statistics, a COUNT, OR an inventory/breakdown across groups:
  - **"How many...?"** — this is ALWAYS aggregation (aggregation_op="count"). Do NOT use exact_filter for "How many" questions.
  - Average, count, minimum, maximum, comparison across states or types
  - **"What types/kinds of facilities are available in [location]?"** — this is ALWAYS aggregation:
    set aggregation_op="count", aggregation_group_by="facility_type"
  - **CRITICAL FOR CITIES**: If the user mentions a specific city, you MUST set `location_text` to that city
    (e.g., location_text="Boulder, CO"). Do NOT skip location_text and fall back to just states=["CO"]
    for city-level questions — that would give a statewide count instead of a city count.
  - Example: "Average RN hours per resident day for nursing homes in NC vs CO"
  - Example: "How many home health agencies are in California?"
  - Example: "What kind of facilities are available in San Diego?" → aggregation_op="count", aggregation_group_by="facility_type", location_text="San Diego, CA", states=["CA"]
  - Example: "What types of facilities exist in Denver?" → aggregation_op="count", aggregation_group_by="facility_type", location_text="Denver, CO", states=["CO"]
  - Example: "What types of facilities are in Boulder?" → aggregation_op="count", aggregation_group_by="facility_type", location_text="Boulder, CO", states=["CO"]

**fuzzy** — use when the query is purely descriptive with NO extractable hard filters,
  OR when the user mentions a specific facility name (name lookup):
  - Example: "A safe place for my grandmother who had a stroke"
  - Example: "Tell me about Sunrise Manor"

**hybrid** — use when the query has BOTH hard filters AND descriptive/qualitative language:
  - The hard filters go to SQL, the fuzzy text goes to vector search
  - Example: "Home health in San Diego with good walking outcomes"
  - Example: "Non-profit nursing homes in NC with a warm, family-friendly atmosphere"

**web_search** — use ONLY when the question has NO database component whatsoever:
  - The user asks about a state NOT in [NC, CO, AZ, CA] (e.g., Texas, New York).
  - Pure general medical/domain knowledge with no specific facility involved (e.g., "What is Medicare Part A?", "What does hospice mean?", "How do I choose a nursing home?").
  - Examples → query_type="web_search":
    - "Best facilities in Dallas, TX?" → web_search (Texas not in DB)
    - "What does CMS star rating mean?" → web_search (general knowledge)

**Meta/Conversational Queries (Follow-ups)**
  - If the user asks a follow-up question about the PREVIOUS response (e.g., "Where did you get that info?", "Is that from the web?"), classify it as `fuzzy` so the agent can answer from its conversation memory.

   - `exact_filter`: The user is ONLY filtering on hard metadata (location, star rating, facility type, specific services like PT).
   - `aggregation`: The user is asking for a count, average, sum, OR asking what types/kinds of facilities exist in a place.
   - `fuzzy`: The user is ONLY providing descriptive/subjective text without hard filters (e.g., "Safe place for grandmother").
   - `hybrid`: A mix of hard filters AND descriptive text.
   - `web_search`: ONLY for general knowledge or out-of-network states — NO specific facility in our DB is involved.
   *CRITICAL*: If there is NO descriptive/subjective text (i.e. residual_fuzzy_text will be null), you MUST choose `exact_filter` or `aggregation`, NEVER `hybrid` or `fuzzy`.



---

## QUALITY LANGUAGE INTERPRETATION

When the user uses quality/superlative language, set `min_rating` accordingly — **regardless of facility type**:
- "best", "top", "highest rated", "outstanding" → min_rating = 4
- "excellent", "premium", "exceptional" (with strong emphasis) → min_rating = 5
- "good", "decent", "quality", "nice", "well-rated", "highly rated" → min_rating = 3

**Always keep the quality language in `residual_fuzzy_text` for semantic context AND set the rating filter.**

Ratings are tracked per-facility, not per-type. All facility types (Nursing Home, Home Health, Hospice, Inpatient Rehabilitation, Healthcare Facility) can have a rating or lack one. The search system handles this automatically — applying `min_rating` simply limits results to facilities that DO have a qualifying star rating. Do not second-guess this based on facility type.

Examples:
- "Best nursing homes in NC" → states=["NC"], facility_type="Nursing Home", min_rating=4
- "Excellent Inpatient Rehabilitation in CA" → states=["CA"], facility_type="Inpatient Rehabilitation", min_rating=5, residual_fuzzy_text="excellent inpatient rehabilitation"
- "Good places for my mom in Colorado" → states=["CO"], min_rating=3 (facility_type may be null if vague)

---

## CRITICAL RULES

1. **States:** Only extract state codes (NC, CO, AZ, CA). If the user says "North Carolina", 
   output "NC". If no state is mentioned, leave states as an empty list.

2. **location_text vs states:** "Near San Diego" is a `location_text` (needs geocoding for 
   radius search). "In California" is just a state filter → states=["CA"]. 
   A specific city like "Charlotte" with no radius language → set location_text to "Charlotte, NC".

3. **residual_fuzzy_text:** This is the portion of the query that CANNOT be expressed as a 
   hard SQL filter. Extract only the conceptual/descriptive part. 
   If the query is "nursing homes in NC with good ratings near downtown Charlotte",
   residual_fuzzy_text should be null (all extractable as hard filters + location), 
   BUT if the query is "a safe place with caring staff near downtown Charlotte", 
   residual_fuzzy_text = "safe place with caring staff".
   - Examples → query_type="fuzzy":
    - "safe and clean nursing homes" → fuzzy, is_specific_facility=False, residual_fuzzy_text="safe and clean nursing homes"
    - "What are the visiting hours for AVEANNA HEALTHCARE?" → fuzzy, is_specific_facility=True, residual_fuzzy_text="AVEANNA HEALTHCARE"
    - "Any recent news or lawsuits involving Golden Hearts?" → fuzzy, is_specific_facility=True, residual_fuzzy_text="Golden Hearts"
   - Examples → query_type="hybrid":
    - "safe and clean nursing homes in Phoenix" → hybrid, is_specific_facility=False, residual_fuzzy_text="safe and clean nursing homes", city="Phoenix"
    - "Tell me about Dependable Home Health in AZ" → hybrid, is_specific_facility=True, residual_fuzzy_text="Dependable Home Health", states=["AZ"].

4. **aggregation_group_by:** For comparison queries like "NC vs CO", set 
   aggregation_group_by = "source_state".

5. **Services for semantic conditions:** "Stroke recovery" → services=["pt", "ot", "speech"] 
   because stroke recovery typically needs all three. "Help my dad walk again" → services=["pt"].
   "Breathing problems" → residual_fuzzy_text (no direct service flag for breathing).

6. **Do NOT guess facility type** if the user is vague. Leave null rather than guess.

7. **Generic scope reset:** If the user's query uses a broad/generic word like 
   "facilities", "places", "options", "anything", or "any type" WITHOUT naming
   a specific facility type, set `facility_type = null`. The user is deliberately 
   broadening the search.
   Example: "what facilities offer speech therapy" → facility_type=null, required_services=["speech"].
""".strip()


def classify_intent(state: AgentState) -> dict:
    """
    LangGraph node: classify_intent.
    Reads state["query"] and state["messages"] (for multi-turn context).
    Writes state["classification"].
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)
    structured_llm = llm.with_structured_output(QueryClassification)

    # Build the message list: system prompt + current query
    messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    query_to_classify = state.get("resolved_query") or state["query"]
    messages.append(HumanMessage(content=query_to_classify))

    classification = structured_llm.invoke(messages)

    # Post-processing: clean up state codes to uppercase
    raw_states = [s.upper().strip() for s in classification.states]

    # Map full state names to their 2-letter codes
    STATE_NAME_TO_CODE = {
        "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA", 
        "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA", 
        "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", 
        "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD", 
        "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO", 
        "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", 
        "NEW MEXICO": "NM", "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", 
        "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", 
        "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", 
        "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
        "DISTRICT OF COLUMBIA": "DC"
    }
    
    ALL_US_CODES = set(STATE_NAME_TO_CODE.values())
    
    # If the LLM returned a full name, map it to the 2-letter code
    mapped_states = [STATE_NAME_TO_CODE.get(s, s) for s in raw_states]

    # Guard: filter out any state codes that aren't in our database
    from agent.models import VALID_STATES
    classification.states = [s for s in mapped_states if s in VALID_STATES]
    
    # Only flag as unsupported if it's an actual US state/code (prevents cities like "San Diego" from breaking the search)
    unsupported = [s for s in mapped_states if s in ALL_US_CODES and s not in VALID_STATES]

    # ---- City → State inference via Geocoder ----
    # If location_text is set but states is still empty, run the geocoder to infer the state.
    # This ensures the Qdrant state filter and the synthesizer context are both correct 
    # for city-only follow-up messages like "search in los angeles".
    if classification.location_text and not classification.states:
        from agent.utils import geocode_location
        # Use state.get("geo_cache", {}) so we don't crash if it's missing
        geo = geocode_location(classification.location_text, state.get("geo_cache", {}))
        if geo and geo.get("state_code"):
            inferred = geo["state_code"]
            if inferred in VALID_STATES:
                classification.states = [inferred]

    return {
        "classification": classification,
        # Append RESOLVED query to history — not raw query — so the Rewriter
        # sees complete, unambiguous sentences on subsequent turns.
        "messages": [{"role": "user", "content": state.get("resolved_query") or state["query"]}],
        "unsupported_states": unsupported,
    }



def test_classify_standalone(query: str, prior_messages: list | None = None) -> QueryClassification:
    """
    Call classify_intent with a raw query and print the result.
    Used for the 20-query eyeball test before wiring the graph.
    """
    state: AgentState = {
        "query": query,
        "classification": None,
        "tool_result": None,
        "response": "",
        "messages": prior_messages or [],
        "geo_cache": {},
        "unsupported_states": [],
        "needs_clarification": False,
        "clarification_stage": None,
        "pending_classification": None,
    }
    result = classify_intent(state)
    return result["classification"]
