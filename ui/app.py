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
                full_state = {
                    "query": prompt,
                    "geo_cache": {},
                    "unsupported_states": [],
                    "needs_clarification": False,
                    "clarification_stage": None,
                    "pending_classification": None,
                    "web_results": None,
                    "web_search_source": None,
                    "web_search_failed": False,
                    "pending_tool_call": None,
                }
                
                # Use stream to update the UI dynamically as nodes execute
                # We use stream_mode=["updates", "messages"] to get both state changes and LLM tokens
                response_placeholder = st.empty()
                streamed_text = ""
                
                for event_mode, event_data in app.stream(full_state, config=config, stream_mode=["updates", "messages"]):
                    if event_mode == "messages":
                        chunk, metadata = event_data
                        # Stream tokens from the synthesize node to the UI
                        if metadata.get("langgraph_node") == "synthesize" and chunk.content:
                            streamed_text += chunk.content
                            response_placeholder.markdown(streamed_text + "▌")
                            
                    elif event_mode == "updates":
                        # Safely retrieve the officially merged state from PostgresSaver
                        # This fixes the UI State Mutation bug where lists were overwritten instead of appended
                        full_state = app.get_state(config).values
                        
                        for node_name, node_state in event_data.items():
                            # Update UI based on what node just finished
                            if node_name == "rewrite_query":
                                status_placeholder.markdown("🔄 Resolving query context...")
                            elif node_name == "classify_intent":
                                c = node_state.get("classification")
                                if c:
                                    if getattr(c, "query_type", "") == "web_search":
                                        status_placeholder.markdown("🌐 Searching the web...")
                                    elif getattr(c, "query_type", "") == "aggregation":
                                        status_placeholder.markdown("📊 Running aggregation query...")
                                    else:
                                        status_placeholder.markdown("🔍 Searching verified database...")
                            elif node_name == "synthesize" and node_state.get("pending_tool_call"):
                                status_placeholder.markdown("🌐 Information missing from DB, augmenting with web search...")
                                # Clear any streamed text since it was a tool call, not a final answer
                                response_placeholder.empty()
                                streamed_text = ""
                                
            status_placeholder.empty()
            # Clear the cursor from the streamed text
            if streamed_text:
                response_placeholder.markdown(streamed_text)

            classification = full_state.get("classification")
            query_type = getattr(classification, "query_type", "exact_filter") if classification else "exact_filter"

            agent_response = full_state.get("response", "")

            # Append source attribution footer
            source_parts = []
            web_source = full_state.get("web_search_source")   # None unless web_tool ran THIS turn
            web_failed = full_state.get("web_search_failed", False)
            tool_result = full_state.get("tool_result") or {}
            db_rows = tool_result.get("row_count", 0)

            # Only show DB source if web was not the sole source
            if classification and query_type != "web_search" and db_rows and db_rows > 0:
                # If the LLM realized the DB results were irrelevant (e.g. name mismatch) and explicitly
                # stated the facility isn't in the DB, we shouldn't list the DB as a source.
                if "isn't in our verified database" not in agent_response.lower() and "is not listed in our verified database" not in agent_response.lower():
                    source_parts.append("📋 **Verified Database** (CMS / State Directory)")
            # Only show web source if the web_tool actually ran this turn
            if web_source and not web_failed:
                source_parts.append(f"🌐 **Web Search** via {web_source.capitalize()}")
            elif web_failed:
                source_parts.append("🌐 **Web Search** — *failed to retrieve results*")

            if source_parts:
                agent_response += "\n\n---\n**Sources used:** " + " · ".join(source_parts)

            # Update the placeholder with the final response including sources
            if streamed_text:
                response_placeholder.markdown(agent_response)
            else:
                st.markdown(agent_response)

            # 3. Save assistant message to UI state
            st.session_state.messages.append({"role": "assistant", "content": agent_response})

        except Exception as e:
            status_placeholder.empty()
            st.error(f"An error occurred: {e}")
