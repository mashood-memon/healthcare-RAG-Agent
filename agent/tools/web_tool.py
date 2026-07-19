import os
from agent.models import AgentState

def get_tavily_client():
    from tavily import TavilyClient
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set in .env")
    return TavilyClient(api_key=api_key)

def run_web_search(state: AgentState) -> dict:
    """Execute a web search using Tavily."""
    # In the new Evaluate-and-React architecture, synthesize decides EXACTLY
    # what to search for and places it in pending_tool_call.
    query = state.get("pending_tool_call")
    if not query:
        # Fallback if someone routes here directly without a tool call
        query = state["query"]


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
