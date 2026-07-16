import os
import sys
import uuid
import streamlit as st

# Ensure we can import from the agent package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agent.graph import get_agent_executor

st.set_page_config(page_title="Healthcare RAG Agent", page_icon="🏥", layout="centered")

st.title("🏥 Healthcare Facility Assistant")
st.markdown("Ask me to find nursing homes, home health agencies, and hospices!")



if "thread_id" not in st.session_state:
    # Unique thread ID for LangGraph PostgresSaver to maintain memory
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Streamlit chat history (separate from LangGraph's internal history)
    # We maintain this so the UI doesn't clear when the page re-renders
    st.session_state.messages = []



for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def get_spinner_label(query_type: str, requires_web_search: bool) -> str:
    """Return a status label that reflects the actual search path taken."""
    if query_type == "web_search":
        return "🌐 Searching the web..."
    elif requires_web_search:
        return "🔍 Searching database, then web..."
    elif query_type == "aggregation":
        return "📊 Running aggregation query..."
    elif query_type in ("fuzzy", "hybrid"):
        return "🔍 Searching database (semantic)..."
    else:
        return "🔍 Searching database..."


if prompt := st.chat_input("Ask about healthcare facilities... (e.g. '5-star nursing homes in NC')"):
    # 1. Add user message to UI state and render it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the LangGraph agent
    with st.chat_message("assistant"):
        # Start with a generic label; we'll update after classification if possible
        status_placeholder = st.empty()
        status_placeholder.markdown("⏳ Thinking...")

        try:
            # The config passes the thread_id to PostgresSaver
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            with get_agent_executor() as app:
                result = app.invoke(
                    {
                        "query": prompt,
                        "geo_cache": {},
                        "unsupported_states": [],
                        "needs_clarification": False,
                        "clarification_stage": None,
                        "pending_classification": None,
                        # Reset web state each turn so PostgresSaver doesn't bleed
                        # stale values from a previous query into the source footer.
                        "web_results": None,
                        "web_search_source": None,
                        "web_search_failed": False,
                    },
                    config=config
                )

            # Update spinner with the actual search type used
            classification = result.get("classification")
            if classification:
                query_type = getattr(classification, "query_type", "exact_filter")
                requires_web = getattr(classification, "requires_web_search", False)
                spinner_label = get_spinner_label(query_type, requires_web)
            else:
                spinner_label = "🔍 Searching..."

            status_placeholder.empty()

            agent_response = result["response"]

            # Append source attribution footer
            source_parts = []
            web_source = result.get("web_search_source")   # None unless web_tool ran THIS turn
            web_failed = result.get("web_search_failed", False)
            tool_result = result.get("tool_result") or {}
            db_rows = tool_result.get("row_count", 0)

            # Only show DB source if web was not the sole source
            if classification and query_type != "web_search" and db_rows and db_rows > 0:
                source_parts.append("📋 **Verified Database** (CMS / State Directory)")
            # Only show web source if the web_tool actually ran this turn
            if web_source and not web_failed:
                source_parts.append(f"🌐 **Web Search** via {web_source.capitalize()}")
            elif web_failed:
                source_parts.append("🌐 **Web Search** — *failed to retrieve results*")

            if source_parts:
                agent_response += "\n\n---\n**Sources used:** " + " · ".join(source_parts)

            st.markdown(agent_response)

            # 3. Save assistant message to UI state
            st.session_state.messages.append({"role": "assistant", "content": agent_response})

        except Exception as e:
            status_placeholder.empty()
            st.error(f"An error occurred: {e}")
