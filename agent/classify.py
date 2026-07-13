from __future__ import annotations

import os
from openai import OpenAI
from dotenv import load_dotenv

from agent.models import QueryClassification, AgentState
from agent.utils import format_history_for_openai

load_dotenv()

# System prompt is built at module load time so it's visible during testing
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

**exact_filter** — use when the query can be fully answered by exact SQL filters:
  - Specific state(s), rating, services, ownership type, county, zip
  - No fuzzy/descriptive language
  - Example: "Nursing homes in NC with 4+ star rating and speech therapy"

**aggregation** — use when the query asks for computed statistics across groups:
  - Average, count, minimum, maximum, comparison across states or types
  - Example: "Average RN hours per resident day for nursing homes in NC vs CO"
  - Example: "How many home health agencies are in California?"

**fuzzy** — use when the query is purely descriptive with NO extractable hard filters,
  OR when the user mentions a specific facility name (name lookup):
  - Example: "A safe place for my grandmother who had a stroke"
  - Example: "Tell me about Sunrise Manor"

**hybrid** — use when the query has BOTH hard filters AND descriptive/qualitative language:
  - The hard filters go to SQL, the fuzzy text goes to vector search
  - Example: "Home health in San Diego with good walking outcomes"
  - Example: "Non-profit nursing homes in NC with a warm, family-friendly atmosphere"

   - `exact_filter`: The user is ONLY filtering on hard metadata (location, star rating, facility type, specific services like PT).
   - `aggregation`: The user is asking for a count, average, or sum (e.g., "How many...", "Average RN hours").
   - `fuzzy`: The user is ONLY providing descriptive/subjective text without hard filters (e.g., "Safe place for grandmother").
   - `hybrid`: A mix of hard filters AND descriptive text.
   *CRITICAL*: If there is NO descriptive/subjective text (i.e. residual_fuzzy_text will be null), you MUST choose `exact_filter` or `aggregation`, NEVER `hybrid` or `fuzzy`.

---

## QUALITY LANGUAGE INTERPRETATION

When the user uses quality/superlative language, set `min_rating` accordingly:
- "best", "top", "highest rated", "outstanding" → min_rating = 4
- "excellent", "premium", "exceptional" (with strong emphasis) → min_rating = 5
- "good", "decent", "quality", "nice", "well-rated", "highly rated" → min_rating = 3
- "poor", "bad", "worst" → typically sets min_rating=1 for comparison, but usually users want to avoid these

**Keep the quality language in `residual_fuzzy_text` for semantic context, BUT also set the rating filter.**

Examples:
- "Best nursing homes in NC" → states=["NC"], facility_type="Nursing Home", min_rating=4
- "Excellent home health agencies in CA" → states=["CA"], facility_type="Home Health", min_rating=5
- "Good places for my mom in Colorado" → states=["CO"], min_rating=3 (facility_type may be null if vague)

---

## MISSING INFORMATION DETECTION

After extracting all available information, check if critical information is missing:

**Location is required for meaningful search:**
- If no location info (states, location_text, zip_code, county) is provided → set needs_clarification=True and clarification_stage="location"

**Facility type helps narrow results:**
- If query_type is "fuzzy" and facility_type is null → set needs_clarification=True and clarification_stage="facility_type"
- Only set this IF we have location information (ask location first)

**Priority: Location first, then facility type.**
- If missing both, only set clarification_stage="location" (ask one at a time)

When needs_clarification=True, continue extracting all other available information into the classification fields (rating, services, etc.) so we can merge them after the user provides the missing piece.

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

4. **aggregation_group_by:** For comparison queries like "NC vs CO", set 
   aggregation_group_by = "source_state".

5. **Services for semantic conditions:** "Stroke recovery" → services=["pt", "ot", "speech"] 
   because stroke recovery typically needs all three. "Help my dad walk again" → services=["pt"].
   "Breathing problems" → residual_fuzzy_text (no direct service flag for breathing).

6. **Do NOT guess facility type** if the user is vague. Leave null rather than guess.

7. **Multi-turn awareness:** The conversation history is provided. If the user says 
   "What about in Arizona?" without context, it means they are asking the same question 
   but filtered to Arizona — carry over the previous classification with states=["AZ"].
""".strip()


def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=key)


def classify_intent(state: AgentState) -> dict:
    """
    LangGraph node: classify_intent.
    Reads state["query"] and state["messages"] (for multi-turn context).
    Writes state["classification"].
    """
    client = get_openai_client()

    # Build the message list: system prompt + full conversation history + current query
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Include prior conversation turns if they exist (multi-turn awareness)
    prior = state.get("messages", [])
    if prior:
        messages.extend(format_history_for_openai(prior))

    # Add the current user query
    messages.append({"role": "user", "content": state["query"]})

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=QueryClassification,
        temperature=0,          # deterministic — classification is not creative
        max_tokens=1024,
    )

    classification = response.choices[0].message.parsed

    # Post-processing: clean up state codes to uppercase
    classification.states = [s.upper().strip() for s in classification.states]

    # Guard: filter out any state codes that aren't in our database
    from agent.models import VALID_STATES
    all_extracted = classification.states[:]
    classification.states = [s for s in classification.states if s in VALID_STATES]
    unsupported = [s for s in all_extracted if s not in VALID_STATES]

    return {
        "classification": classification,
        # Append current user message to conversation history for multi-turn
        "messages": [{"role": "user", "content": state["query"]}],
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



