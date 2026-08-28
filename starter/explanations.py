"""Deterministic customer-facing language.

The natural-language ``message`` must agree exactly with the structured
``ask_attribute``: the simulator answers the structured field, so wording a
question the field cannot express would mislead a human reader while producing
an unrelated reply. Templates are fixed strings - no model call, no sampling.
"""

from __future__ import annotations

QUESTION_TEMPLATES: dict[str, str] = {
    "category": "Which type of item are you looking for?",
    "material": "Do you have a preferred material or fabric?",
    "color": "Is there a color you prefer?",
    "size": "Do you have a size or fit requirement?",
    "style": "What style or fit are you looking for?",
    "brand": "Do you prefer a particular brand?",
    "budget": "What price range would you like to stay within?",
    "feature": "Is there a specific feature that matters most?",
    "use_case": "What will you mainly use it for?",
    "other": "What other requirement matters most to you?",
}

_NO_QUESTION = "Here are the closest matches I found based on everything you've told me."
_NO_RESULTS = "I couldn't find a close match yet - here are the nearest options I have."


def question_text(attribute: str | None) -> str:
    if not attribute:
        return _NO_QUESTION
    return QUESTION_TEMPLATES.get(attribute, QUESTION_TEMPLATES["other"])


def compose_message(attribute: str | None, count: int, known_constraints: int) -> str:
    """One short sentence of context plus the question, if any."""
    if count == 0:
        return _NO_RESULTS if not attribute else f"{_NO_RESULTS} {question_text(attribute)}"

    if known_constraints == 0:
        lead = f"Here are {count} options to start from."
    elif known_constraints == 1:
        lead = f"Here are {count} options matching what you've described."
    else:
        lead = f"Here are {count} options matching the {known_constraints} details you've shared."

    if not attribute:
        return f"{lead} {_NO_QUESTION}"
    return f"{lead} {question_text(attribute)}"
