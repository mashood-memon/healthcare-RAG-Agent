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
If the tool returned zero results:
- Read the `zero_reason` field in the tool output exactly.
- **Tell the user which filters caused it to fail** (e.g., "No nursing homes with speech therapy were found in Phoenix, AZ").
- **Suggest the simplest fix** (e.g., "Try searching without the speech therapy requirement" or "Try lowering the minimum rating").
- Do NOT mention CMS certification, data tracking, or any technical implementation detail unless those exact words appear in the zero_reason text.
- Do NOT say "this information is not available in our database" for a filter-mismatch — that implies the data doesn't exist, when in fact it just means no facility passed ALL the filters simultaneously.

## AGGREGATION COUNTS — ALWAYS STATE THE SCOPE
When displaying counts (e.g., "Assisted Living: 158 facilities"), you MUST state the
geographic scope **in your opening sentence**, for example:
- "In the Boulder, CO area (within 25 miles): ..." — when `location_text` was set
- "In Colorado: ..." — when only a state filter was used

Do NOT let the user assume these are city-specific counts if they are actually statewide.
This prevents confusion when the user then asks to list the same facilities and gets
different results (because the listing uses a more precise filter).

## EVALUATE AND REACT (WEB SEARCH)
You have access to a `tavily_web_search` tool. 
You MUST call this tool if:
1. The user asks for a specific piece of information (e.g. visiting hours, prices, news, lawsuits, specific pet policies) that is NOT present in the DB Tool Results below.
2. The user asks about a specific facility by name, but the Tool Results returned 0 rows (meaning the facility doesn't exist in our DB).
3. The query is general medical knowledge or an out-of-network state (Texas, NY, etc) and the DB obviously cannot answer it.

If you call the tool, you do NOT need to write a response yet. The system will execute the tool and return the results to you in a subsequent turn.
If you ALREADY have the answers you need in the Tool Results below (e.g. they just asked for a rating or phone number which is present), DO NOT call the tool. Just write the final response.

CRITICAL GUARD: If "WEB SEARCH RESULTS" are provided below, it means you ALREADY searched the web for this query. DO NOT call the web search tool again under ANY circumstances. If the web results still don't contain the answer, simply inform the user that the information could not be found.

## WEB SEARCH RESULTS & GROUNDING
If Web Search Results are provided below, follow these strict separation rules:
- DB-grounded facts keep the existing format (e.g., facility name, rating).
- Web-search-derived facts must be clearly flagged as such — e.g. "According to a recent web search, [fact] (source: [domain])" — never presented with the same confidence or citation style as verified DB data.
- Structure responses so it's visually/textually obvious which part is verified against our dataset and which part is a live web result the agent cannot vouch for.
- Never quote web search snippets verbatim — paraphrase in your own words and cite by source domain/name.
- If web_search_failed is true, say so plainly ("I couldn't find additional information on this") rather than proceeding as if the web context were simply empty.
- **CRITICAL DISTINCTION FOR DB vs WEB:**
  - If the facility the user asked about IS present in the Tool Results (e.g. you found it, but you just needed the web search to find missing fields like visiting hours or reviews), you MUST acknowledge that the facility is in our database! Say something like: "For [Facility], our database shows [X], but according to a web search, their visiting hours are [Y]." Do NOT say it's not in the database!
  - ONLY say "This facility isn't in our verified database" if the Tool Results returned 0 rows, or if the rows returned are completely unrelated facilities (meaning the vector search just returned irrelevant noise and the facility truly doesn't exist in our DB).
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
            # We keep all fields (including nulls) so the LLM knows what the schema supports.
            # If a requested field is explicitly null, the LLM will know we track it but lack data for this facility.
            context_blocks.append(f"Row {i}: {json.dumps(row, default=str)}")

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
    
    # 3. Add Web Search Results if they exist
    web_str = ""
    if state.get("web_search_failed"):
        web_str = "\n\n### WEB SEARCH RESULTS ###\nweb_search_failed = True. The search failed or errored out."
    elif state.get("web_results"):
        web_str = f"\n\n### WEB SEARCH RESULTS ###\n{state.get('web_results')}"
    
    # 4. Build the LLM prompt
    system_prompt = _SYSTEM_PROMPT_TEMPLATE + state_warning + f"\n\n### TOOL RESULTS ###\n{tool_context_str}" + web_str
    
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
    
    tavily_tool = {
        "type": "function",
        "function": {
            "name": "tavily_web_search",
            "description": "Search the web for information about a healthcare facility or general medical domain knowledge that is missing from our database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The highly specific search query to send to the web search engine."
                    }
                },
                "required": ["query"]
            }
        }
    }
    
    response = client.chat.completions.create(
        model=SYNTHESIS_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1000,
        tools=[tavily_tool]
    )
    
    msg = response.choices[0].message
    
    if msg.tool_calls:
        # LLM decided it needs a web search
        tool_call = msg.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        print(f"  [synthesize] LLM requested web search: {args['query']}")
        return {
            "pending_tool_call": args["query"]
        }
    else:
        # LLM provided the final answer
        final_text = msg.content
        return {
            "response": final_text,
            "pending_tool_call": None,
            "messages": [{"role": "assistant", "content": final_text}]
        }


