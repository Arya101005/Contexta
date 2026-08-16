def validate_answer(answer: str, context: str, sources: list | None = None) -> str:
    """
    Validate an LLM answer before returning it to the user.
    - Reject empty answers
    - Reject if there was no context
    - Reject if no sources were retrieved (indicates no evidence)
    - Otherwise return the answer as-is (including honest "I don't know" responses)
    """
    if not answer or not answer.strip():
        return "I could not generate an answer."  # LLM returned nothing

    if not context or not context.strip():
        return "I could not find this information in the provided documents."  # no context to answer from

    if sources is not None and len(sources) == 0:
        return "I could not find this information in the provided documents."  # retrieval found nothing

    return answer.strip()  # answer is valid — return it as-is
