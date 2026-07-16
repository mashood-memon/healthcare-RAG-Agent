import os
from agent.models import AgentState

def get_tavily_client():
    from tavily import TavilyClient
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set in .env")
    return TavilyClient(api_key=api_key)

def run_web_search(state: AgentState) -> dict:
    classification = state.get("classification")
    
    # Safety guard - if somehow classification is None, just search the query
    if not classification:
        query = state["query"]
    elif classification.query_type == "web_search":
        # Pure out-of-scope — search the raw query directly
        query = state["query"]
    else:
        # DB-augmentation case — build a targeted query using facility identity
        tool_res = state.get("tool_result") or {}
        rows = tool_res.get("rows", [])
        if not rows:
            # Zero-result fallback case (facility wasn't found in DB)
            query = state["query"]
        else:
            # Found at least one facility, anchor the search to the first one
            facility = rows[0]
            query = f"{facility.get('name', '')} {facility.get('city', '')} {facility.get('source_state', '')} {state['query']}"
            query = query.strip()

    try:
        client = get_tavily_client()
        response = client.search(query=query, max_results=5)
        
        # Format the response into a readable string for the synthesizer
        results_list = response.get("results", [])
        formatted_results = []
        for res in results_list:
            formatted_results.append(f"Source: {res.get('url')}\nContent: {res.get('content')}")
            
        web_str = "\n\n".join(formatted_results) if formatted_results else "No relevant web results found."
        
        return {
            "web_results": web_str,
            "web_search_source": "tavily",
            "web_search_failed": False,
        }
    except Exception as e:
        print(f"[Web Search Failed]: {e}")
        return {
            "web_results": None,
            "web_search_source": "tavily",
            "web_search_failed": True,
        }
