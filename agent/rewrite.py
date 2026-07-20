from __future__ import annotations

import os
from openai import OpenAI
from dotenv import load_dotenv

from agent.models import AgentState

load_dotenv()

_REWRITE_SYSTEM_PROMPT = """
You are a conversation context manager for a healthcare facility search system.
Your ONLY job is to rewrite the user's latest message into a single, standalone search query
that captures their full intent, using the conversation history for context.

RULES:
1. **Filter carry-over:** If the user refers back to previous search filters (e.g., "What about in Arizona?",
   "Ones with speech therapy?", "What about 4-star ones?"), incorporate the previously active filters
   (facility type, rating, services, state) into the new standalone query.

2. **Implicit subject (most important rule):** If the user's message is a short, ambiguous follow-up whose
   subject is clearly the specific facility or topic discussed in the PREVIOUS assistant turn
   (e.g., "what are their visiting hours?", "tell me more", "so what about stroke recovery",
   "what services do they offer?", "how many staff do they have?"), resolve the implicit
   subject by inserting the specific facility name from the previous assistant turn.
   Example: history ends with the bot describing "NOVANT HEALTH PRESBYTERIAN MEDICAL CENTER-SNU",
   user says "what are visiting hours" → rewrite to:
   "What are the visiting hours for NOVANT HEALTH PRESBYTERIAN MEDICAL CENTER-SNU?"

3. **Topic pivot with specific name:** If the user explicitly names a DIFFERENT specific facility
   (e.g., "Tell me about Liberty Home Care"), do NOT carry over old location filters or services.
   Produce a clean query for that new facility.

4. **Meta-questions:** If the user asks a meta-question about the previous response (e.g., "Where did you get that?"),
   rewrite it clearly: "Where did you get the information in your previous response?"

5. Return ONLY the rewritten query text. No conversational filler, no quotes.
"""

def rewrite_query(state: AgentState) -> dict:
    """
    LangGraph node: rewrite_query.
    Reads state["query"] and state["messages"].
    Writes state["resolved_query"].
    """
    prior_messages = state.get("messages", [])
    current_query = state["query"]

    # First turn optimization: no history, no need to rewrite
    if not prior_messages:
        return {"resolved_query": current_query}

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=200)

    messages = [SystemMessage(content=_REWRITE_SYSTEM_PROMPT.strip())]
    messages.extend(prior_messages)
    messages.append(HumanMessage(content=f"Please rewrite this query to be standalone: {current_query}"))

    print(f"  [rewrite] Resolving context for: '{current_query}'")

    response = llm.invoke(messages)
    resolved = response.content.strip()
    
    print(f"  [rewrite] Resolved to: '{resolved}'")

    return {"resolved_query": resolved}
