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

## ABSOLUTE GROUNDING RULES — VIOLATION IS UNACCEPTABLE
1. **The Tool Results below are your ONLY source of truth.** You have NO other knowledge.
   Treat your own training data as if it does not exist. You know NOTHING about any 
   healthcare facility, state, address, phone number, or service except what appears 
   in the Tool Results below.
2. **Never guess, infer, or fill in blanks.** If a field (address, phone, services, state, 
   rating, etc.) is not present in the tool results, say: "This information is not available 
   in our database." Do NOT fill it in from memory. Do NOT say "typically" or "usually."
3. **Cite the source for specific claims** by mentioning the facility name and state 
   exactly as they appear in the tool results.
4. **Be concise but helpful.** Present the information clearly. Use bullet points 
   for lists of facilities.
5. **Exact numbers.** For averages, counts, or sums, state the exact numbers provided 
   in the results. Do not round aggressively.
6. **For follow-up questions:** ONLY use data from the CURRENT tool results. If the user 
   asks "what services does it provide?" and the tool results show service fields as null 
   or false, say those services are not available or not provided — do NOT make up services.

## UNSUPPORTED STATES
Our database ONLY covers: North Carolina (NC), Colorado (CO), Arizona (AZ), and California (CA).
If the UNSUPPORTED_STATES warning appears below, you MUST tell the user:
"We don't currently have data for [state name(s)]. Our database covers NC, CO, AZ, and CA."
Do NOT attempt to answer the query using your own knowledge. Do NOT invent facilities.

## ZERO RESULTS HANDLING
If the tool returned zero results, look at the `zero_reason` provided in the tool output.
- If it says "no_data_for_field", explain to the user that this specific metric (like 
  speech therapy or RN hours) is not tracked for that type of facility in our database.
  Do NOT imply that no such facility exists, only that the data isn't tracked.
- If it says "no_facilities_match_criteria", politely inform the user that no facilities 
  matched all of their strict filters, and suggest they broaden their search.

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
    
    # Inject unsupported state warning if the user asked for states we don't cover
    unsupported = state.get("unsupported_states", [])
    state_warning = ""
    if unsupported:
        state_names = ", ".join(unsupported)
        state_warning = (
            f"\n\n⚠️ UNSUPPORTED_STATES: The user asked about state(s): {state_names}. "
            f"We do NOT have data for these states. Our database only covers NC, CO, AZ, CA. "
            f"You MUST inform the user that we don't have data for {state_names}. "
            f"Do NOT invent or guess any facilities.\n"
        )
    
    # 2. Build the LLM prompt
    system_prompt = _SYSTEM_PROMPT_TEMPLATE + state_warning + f"\n\n### TOOL RESULTS ###\n{tool_context_str}"
    
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


