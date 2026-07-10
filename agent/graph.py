from __future__ import annotations

import os
from contextlib import contextmanager

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from dotenv import load_dotenv

from agent.models import AgentState
from agent.classify import classify_intent
from agent.tools.sql_tool import sql_tool
from agent.tools.vector_tool import vector_tool
from agent.tools.hybrid_tool import hybrid_tool
from agent.synthesize import synthesize

load_dotenv()


def route_by_intent(state: AgentState) -> str:
    """Read the classification and route to the correct tool node."""
    c = state.get("classification")
    if not c:
        # Fallback if classification failed
        return "vector_tool"
    
    q_type = c.query_type
    
    if q_type == "exact_filter" or q_type == "aggregation":
        return "sql_tool"
    elif q_type == "fuzzy":
        return "vector_tool"
    elif q_type == "hybrid":
        return "hybrid_tool"
    else:
        return "vector_tool"


def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("sql_tool", sql_tool)
    workflow.add_node("vector_tool", vector_tool)
    workflow.add_node("hybrid_tool", hybrid_tool)
    workflow.add_node("synthesize", synthesize)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "sql_tool": "sql_tool",
            "vector_tool": "vector_tool",
            "hybrid_tool": "hybrid_tool",
        }
    )

    workflow.add_edge("sql_tool", "synthesize")
    workflow.add_edge("vector_tool", "synthesize")
    workflow.add_edge("hybrid_tool", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow


@contextmanager
def get_agent_executor():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL not set in .env")

    workflow = create_graph()

    with psycopg.connect(db_url, autocommit=True) as conn:
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        
        app = workflow.compile(checkpointer=checkpointer)
        
        yield app

if __name__ == "__main__":
    import uuid
    
    print("Initializing Healthcare RAG Agent...\n")
    
    # Generate a unique thread ID for this CLI session to test multi-turn
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    with get_agent_executor() as app:
        print("Agent ready! Type 'quit' to exit.\n")
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                if not user_input.strip():
                    continue

                # Run the graph
                # AgentState initial inputs: query, and an empty geo_cache
                result = app.invoke(
                    {"query": user_input, "geo_cache": {}, "unsupported_states": []}, 
                    config=config
                )
                
                print(f"\nAgent: {result['response']}\n")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                import traceback
                print(f"\n[ERROR]: {e}")
                traceback.print_exc()
                print("-" * 60)
