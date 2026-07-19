import os
import os
from agent.models import AgentState
from exa_py import Exa

def get_exa_client():
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY not set in .env")
    return Exa(api_key=api_key)

def run_web_search(state: AgentState) -> dict:
    """Execute a web search using Exa."""
    query = state.get("pending_tool_call")
    if not query:
        query = state["query"]

    try:
        exa = get_exa_client()
        response = exa.search(
            query,
            type="fast",
            num_results=3,
            contents={"highlights": True}
        )
        
        formatted_results = []
        for r in response.results:
            highlights = "\n".join(r.highlights) if r.highlights else "No highlights found."
            formatted_results.append(f"Source: {r.url}\nTitle: {r.title}\n{highlights}")
            
        new_web_str = "\n\n".join(formatted_results) if formatted_results else "No relevant web results found."
        
        # Append to existing results so the LLM can do multi-step research
        existing_results = state.get("web_results") or ""
        if existing_results:
            existing_results += "\n\n---\n\n"
        
        final_web_str = existing_results + f"### SEARCH QUERY: {query} ###\n" + new_web_str
        
        return {
            "web_results": final_web_str,
            "web_search_source": "exa",
            "web_search_failed": False,
        }
    except Exception as e:
        print(f"  [web_search] Exa search failed: {e}")
        return {
            "web_results": None,
            "web_search_source": "exa",
            "web_search_failed": True,
        }
