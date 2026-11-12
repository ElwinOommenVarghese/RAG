"""
Stage 6 of Querying: LLM Answer Generation via Groq API.
Uses the official `groq` Python SDK. Never fabricates an LLM response
when the API key is missing or when the API fails.
"""

import time
from typing import Dict, Any, List
from groq import Groq
import config


def generate_answer(
    system_instructions: str,
    llm_user_message: str,
    has_context: bool,
    chat_history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """
    Send the augmented prompt to Groq and return the generated answer + metadata.

    Returns dict with:
      - status: "success" | "missing_api_key" | "no_context" | "api_error"
      - provider: "Groq"
      - model: str
      - answer: str
      - generation_time_sec: float
      - error_message: str | None
    """
    model_name = config.get_groq_model()
    api_key = config.get_groq_api_key()

    if not has_context:
        return {
            "status": "no_context",
            "provider": "Groq",
            "model": model_name,
            "answer": (
                "The uploaded documents do not contain enough information to answer this question "
                "(no relevant chunks were retrieved from the Knowledge Base)."
            ),
            "generation_time_sec": 0.0,
            "error_message": None,
        }

    if not api_key:
        return {
            "status": "missing_api_key",
            "provider": "Groq",
            "model": model_name,
            "answer": "",
            "generation_time_sec": 0.0,
            "error_message": (
                "Groq API key is not configured. Please add `GROQ_API_KEY=your_key` to a `.env` file "
                "in the `campus_companion/` folder (or enter it in Settings) to enable LLM answer generation."
            ),
        }

    messages = [{"role": "system", "content": system_instructions}]
    if chat_history:
        # Include recent conversation turns (up to last 4 messages) for follow-up context
        for msg in chat_history[-4:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": llm_user_message})

    start = time.perf_counter()
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=700,
        )
        elapsed = time.perf_counter() - start
        answer_text = completion.choices[0].message.content.strip()
        return {
            "status": "success",
            "provider": "Groq",
            "model": model_name,
            "answer": answer_text,
            "generation_time_sec": round(elapsed, 4),
            "error_message": None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "status": "api_error",
            "provider": "Groq",
            "model": model_name,
            "answer": "",
            "generation_time_sec": round(elapsed, 4),
            "error_message": f"Groq API request failed: {exc}",
        }

