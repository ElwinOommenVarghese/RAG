"""
Stage 5 of Querying: Prompt Augmentation.
Combines System Instructions, Retrieved Context chunks, and the User Question
into a transparent, inspectable prompt structure.
"""

from typing import List, Dict, Any


SYSTEM_INSTRUCTIONS = """You are Campus Companion, a helpful and transparent student handbook assistant.

RULES YOU MUST FOLLOW:
1. Answer ONLY using the Retrieved Context provided below.
2. Do NOT invent policies, numbers, dates, or facts that are not in the Retrieved Context.
3. If the Retrieved Context does not contain enough information to answer the question, clearly say:
   "The uploaded documents do not contain enough information to answer this question."
4. Cite the source document name(s) in your answer (for example: [Source: 01_student_attendance_policy.txt]).
5. Keep your explanation clear, concise, and easy for a student to understand."""


def build_augmented_prompt(
    user_question: str,
    selected_chunks: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Construct the augmented prompt separated into its 3 educational parts:
      - system_instructions
      - retrieved_context
      - user_question
      - full_prompt (user message sent to the LLM)
    """
    if not selected_chunks:
        context_block = "(No relevant document chunks were retrieved.)"
    else:
        formatted_parts = []
        for idx, chunk in enumerate(selected_chunks, start=1):
            source = chunk["metadata"].get("source", "Unknown Document")
            chunk_idx = chunk["metadata"].get("chunk_index", "?")
            sim = chunk.get("similarity", 0.0)
            formatted_parts.append(
                f"[Chunk {idx} | Source: {source} | Chunk #{chunk_idx} | Similarity: {sim:.4f}]\n"
                f"{chunk['text']}"
            )
        context_block = "\n\n---\n\n".join(formatted_parts)

    user_prompt = (
        f"### Retrieved Context\n"
        f"{context_block}\n\n"
        f"### User Question\n"
        f"{user_question}\n\n"
        f"### Answer (cite source document names):"
    )

    return {
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "retrieved_context": context_block,
        "user_question": user_question,
        "full_prompt": f"[SYSTEM]\n{SYSTEM_INSTRUCTIONS}\n\n[USER]\n{user_prompt}",
        "llm_user_message": user_prompt,
    }

