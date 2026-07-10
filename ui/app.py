import os
import sys
import uuid
import streamlit as st

# Ensure we can import from the agent package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agent.graph import get_agent_executor

st.set_page_config(page_title="Healthcare RAG Agent", page_icon="🏥", layout="centered")

st.title("🏥 Healthcare Facility Assistant")
st.markdown("Ask me to find nursing homes, home health agencies, and hospices! I can handle strict filters (like star ratings) and fuzzy descriptions.")

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    # Unique thread ID for LangGraph PostgresSaver to maintain memory
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Streamlit chat history (separate from LangGraph's internal history)
    # We maintain this so the UI doesn't clear when the page re-renders
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Render Chat History
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about healthcare facilities... (e.g. '5-star nursing homes in NC')"):
    # 1. Add user message to UI state and render it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the LangGraph agent
    with st.chat_message("assistant"):
        with st.spinner("Searching database..."):
            try:
                # The config passes the thread_id to PostgresSaver
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                with get_agent_executor() as app:
                    result = app.invoke(
                        {"query": prompt, "geo_cache": {}}, 
                        config=config
                    )
                
                agent_response = result["response"]
                st.markdown(agent_response)
                
                # 3. Save assistant message to UI state
                st.session_state.messages.append({"role": "assistant", "content": agent_response})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
