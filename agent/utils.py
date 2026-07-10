from __future__ import annotations

def format_history_for_openai(history: list) -> list[dict]:
    """
    Safely parse LangChain message objects OR raw dictionaries into 
    OpenAI-compatible dictionaries (e.g. {"role": "user", "content": "..."}).
    """
    formatted = []
    for msg in history:
        # If it's already a dict, extract role/content safely
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("type", "user"))
            if role == "human": role = "user"
            if role == "ai": role = "assistant"
            formatted.append({"role": role, "content": msg.get("content", "")})
        else:
            # It's a LangChain message object
            # Pydantic v2 BaseModels don't have .get(), so use getattr safely
            msg_type = getattr(msg, "type", "user")
            role = "user" if msg_type == "human" else "assistant"
            content = getattr(msg, "content", "")
            formatted.append({"role": role, "content": content})
            
    return formatted
