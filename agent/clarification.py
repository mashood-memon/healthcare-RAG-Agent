from __future__ import annotations

import os
from openai import OpenAI
from dotenv import load_dotenv

from agent.models import AgentState, QueryClassification

load_dotenv()


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def check_clarification_needed(state: AgentState) -> dict:
    """
    LangGraph node: check_clarification_needed.
    Runs AFTER classification to detect missing critical information.

    Checks the QueryClassification for:
    - Missing location (state, location_text, zip_code, county)
    - Generic/vague facility type or none specified

    If critical info is missing, sets needs_clarification=True on the classification
    and returns early to generate clarification response.
    """
    c: QueryClassification = state.get("classification")

    if not c:
        # No classification - shouldn't happen, but be defensive
        return {
            "needs_clarification": False,
            "clarification_stage": None,
            "pending_classification": None,
        }

    has_location = bool(c.states or c.location_text or c.zip_code or c.county)

    clarification_stage = None

    # Require location for all non-aggregation queries
    if not has_location and c.query_type != "aggregation":
        clarification_stage = "location"

    if clarification_stage:
        # We need clarification
        return {
            "needs_clarification": True,
            "clarification_stage": clarification_stage,
        }

    # No clarification needed - proceed to search tools
    return {
        "needs_clarification": False,
        "clarification_stage": None,
        "pending_classification": None,
    }


def generate_clarification_response(state: AgentState) -> dict:
    """
    LangGraph node: generate_clarification_response.
    Generates a helpful question based on the clarification stage.
    """
    stage = state.get("clarification_stage") or (
        state.get("classification") and state["classification"].clarification_stage
    )

    if stage == "location":
        response = (
            "Where are you looking for facilities? We can search by:\n"
            "- State: NC, CO, AZ, CA\n"
            "- City or county\n"
            "- ZIP code\n\n"
            "Please let me know your preferred location."
        )
    elif stage == "facility_type":
        response = (
            "What type of facility do you need?\n"
            "- **Nursing Home** — long-term residential care\n"
            "- **Home Health** — in-home skilled nursing and therapy\n"
            "- **Hospice** — end-of-life/palliative care\n"
            "- **Inpatient Rehabilitation** — post-acute recovery (stroke, surgery, injury)\n\n"
            "Please let me know the type of care you're looking for."
        )
    elif stage == "services":
        response = (
            "What specific services do you need? Common options include:\n"
            "- Physical Therapy\n"
            "- Occupational Therapy\n"
            "- Speech Therapy\n"
            "- IV/Intravenous Therapy\n"
            "- Durable Medical Equipment (wheelchairs, oxygen)\n"
            "- Social Work / counseling\n\n"
            "Or tell me about the specific care situation (e.g., 'stroke recovery', 'help walking')."
        )
    else:
        response = "I need a bit more information to help you find what you're looking for."

    # Add the response to messages history
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": response})

    return {
        "response": response,
        "messages": messages,
    }

