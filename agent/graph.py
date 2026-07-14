from __future__ import annotations

import os
from contextlib import contextmanager

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from dotenv import load_dotenv

from agent.models import AgentState
from agent.classify import classify_intent
from agent.clarification import check_clarification_needed, generate_clarification_response
from agent.tools.sql_tool import sql_tool
from agent.tools.vector_tool import vector_tool
from agent.tools.hybrid_tool import hybrid_tool
from agent.synthesize import synthesize

load_dotenv()


def route_after_clarification(state: AgentState) -> str:
    """Check if we need clarification, otherwise route to the correct tool."""
    if state.get("needs_clarification"):
        return "generate_clarification"

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
    workflow.add_node("check_clarification", check_clarification_needed)
    workflow.add_node("generate_clarification", generate_clarification_response)
    workflow.add_node("sql_tool", sql_tool)
    workflow.add_node("vector_tool", vector_tool)
    workflow.add_node("hybrid_tool", hybrid_tool)
    workflow.add_node("synthesize", synthesize)

    workflow.set_entry_point("classify_intent")

    workflow.add_edge("classify_intent", "check_clarification")

    workflow.add_conditional_edges(
        "check_clarification",
        route_after_clarification,
        {
            "sql_tool": "sql_tool",
            "vector_tool": "vector_tool",
            "hybrid_tool": "hybrid_tool",
            "generate_clarification": "generate_clarification",
        }
    )

    workflow.add_edge("sql_tool", "synthesize")
    workflow.add_edge("vector_tool", "synthesize")
    workflow.add_edge("hybrid_tool", "synthesize")
    workflow.add_edge("generate_clarification", END)
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


