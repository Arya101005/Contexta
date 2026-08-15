def validate_answer(answer: str, context: str) -> str:
    """
    Validate an LLM answer before returning it to the user.
    """

    if not answer or not answer.strip():
        return "I could not generate an answer."

    if not context or not context.strip():
        return "I could not find this information in the provided documents."

    # Basic protection against clearly invalid responses.
    invalid_phrases = [
        "I don't know",
        "I cannot answer",
        "I can't answer",
    ]

    answer_lower = answer.lower()

    if any(phrase.lower() in answer_lower for phrase in invalid_phrases):
        return "I could not find this information in the provided documents."

    return answer.strip()