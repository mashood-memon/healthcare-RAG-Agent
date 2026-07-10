from __future__ import annotations

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from agent.models import AgentState, QueryClassification
from agent.utils import format_history_for_openai

load_dotenv()

# We use gpt-4o for synthesis because writing natural, fluent, and strictly
# grounded summaries requires more reasoning capability than classification.
SYNTHESIS_MODEL = "gpt-4o"

_SYSTEM_PROMPT_TEMPLATE = """
You are a healthcare facility assistant. Your job is to answer the user's query 
using ONLY the data provided in the Tool Results section below.

## STRICT GROUNDING RULES
1. **Never guess or infer.** If a field is null or missing for a facility, say so explicitly 
   (e.g., "Bed count data is not available for this facility").
2. **Only state facts present in the tool result rows.** Do not add external knowledge.
3. **Cite the source for specific claims** by mentioning the facility name and state.
   (e.g., "Sunrise Manor in NC has a 4-star rating").
4. **Be concise but helpful.** Present the information clearly. Use bullet points 
   for lists of facilities.
5. **Exact numbers.** For averages, counts, or sums, state the exact numbers provided 
   in the results. Do not round aggressively unless it makes the text flow better 
   (e.g., 4.02 is fine, 4.023847 is too much).

## ZERO RESULTS HANDLING
If the tool returned zero results, look at the `zero_reason` provided in the tool output.
- If it says "no_data_for_field", explain to the user that this specific metric (like 
  speech therapy or RN hours) is not tracked for that type of facility in our database.
  Do NOT imply that no such facility exists, only that the data isn't tracked.
- If it says "no_facilities_match_criteria", politely inform the user that no facilities 
  matched all of their strict filters, and suggest they broaden their search (e.g., lower 
  the star rating or remove a service requirement).

## CONVERSATION CONTEXT
You are in a multi-turn conversation. The user's query might be a follow-up (e.g., 
"What about in Arizona?"). Answer naturally in the flow of the conversation, but 
always base the actual data on the Tool Results provided below.
"""

def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def synthesize(state: AgentState) -> dict:
    """
    LangGraph node: synthesize.
    Reads state["query"], state["classification"], state["tool_result"], and state["messages"].
    Writes state["response"].
    """
    c: QueryClassification = state.get("classification")
    tool_result: dict = state.get("tool_result", {})
    
    # 1. Format the tool results into a compact string for the LLM context
    # We drop heavy fields or nulls to save tokens if we have many rows
    context_blocks = []
    
    if tool_result.get("error"):
        context_blocks.append(f"ERROR EXECUTING SEARCH: {tool_result['error']}")
    elif tool_result.get("row_count", 0) == 0:
        context_blocks.append(f"ZERO RESULTS FOUND.")
        context_blocks.append(f"Reason: {tool_result.get('zero_reason', 'Unknown')}")
    else:
        context_blocks.append(f"FOUND {tool_result['row_count']} RESULTS:")
        
        # Format rows compactly
        rows = tool_result.get("rows", [])
        for i, row in enumerate(rows, 1):
            # Clean out None values to save space
            clean_row = {k: v for k, v in row.items() if v is not None}
            # For aggregations, dict might just be {"source_state": "NC", "result": 4.5, "n": 100}
            context_blocks.append(f"Row {i}: {json.dumps(clean_row, default=str)}")

    tool_context_str = "\n".join(context_blocks)
    
    # 2. Build the LLM prompt
    system_prompt = _SYSTEM_PROMPT_TEMPLATE + f"\n\n### TOOL RESULTS ###\n{tool_context_str}"
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Include prior conversation history
    prior = state.get("messages", [])
    if prior:
        messages.extend(format_history_for_openai(prior))
    else:
        # Fallback if messages isn't populated
        messages.append({"role": "user", "content": state["query"]})

    client = get_openai_client()
    
    print(f"  [synthesize] Generating response...")
    response = client.chat.completions.create(
        model=SYNTHESIS_MODEL,
        messages=messages,
        temperature=0.3,  # Slight creativity for natural text, but strictly grounded
        max_tokens=1000,
    )
    
    final_text = response.choices[0].message.content
    
    return {
        "response": final_text,
        # Append the assistant's response to the conversation history
        "messages": [{"role": "assistant", "content": final_text}]
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agent.models import QueryClassification
    from decimal import Decimal

    # Mock state for an aggregation query
    mock_state_agg = AgentState(
        query="Average RN hours per resident day for nursing homes in NC vs CO",
        classification=QueryClassification(
            query_type="aggregation",
            states=["NC", "CO"],
            facility_type="Nursing Home",
            aggregation_field="rn_hours_per_resident_day",
            aggregation_op="avg",
            aggregation_group_by="source_state",
        ),
        tool_result={
            "columns": ["source_state", "result", "n"],
            "rows": [
                {"source_state": "NC", "result": Decimal("0.48"), "n": 405},
                {"source_state": "CO", "result": Decimal("0.81"), "n": 196},
            ],
            "row_count": 2,
            "zero_reason": None,
            "error": None,
        },
        messages=[{"role": "user", "content": "Average RN hours per resident day for nursing homes in NC vs CO"}],
        response="",
        geo_cache={}
    )
    
    # Mock state for a zero-results query
    mock_state_zero = AgentState(
        query="Nursing homes in NC with speech therapy",
        classification=QueryClassification(
            query_type="exact_filter",
            states=["NC"],
            facility_type="Nursing Home",
            required_services=["speech"],
        ),
        tool_result={
            "columns": [],
            "rows": [],
            "row_count": 0,
            "zero_reason": "no_data_for_field — Service data (PT, OT, speech, etc.) is only tracked for CMS-certified facilities. Coverage: NC: partial (CMS rows only). Try removing the service filter or search without it.",
            "error": None,
        },
        messages=[{"role": "user", "content": "Nursing homes in NC with speech therapy"}],
        response="",
        geo_cache={}
    )

    print("=== TESTING SYNTHESIS NODE ===\n")
    
    print("Test 1: Aggregation")
    res1 = synthesize(mock_state_agg)
    print("Response:\n" + res1["response"] + "\n")
    
    print("Test 2: Zero Results Handling")
    res2 = synthesize(mock_state_zero)
    print("Response:\n" + res2["response"] + "\n")
