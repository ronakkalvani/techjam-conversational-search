"""Question-selection policies.

Three policies are implemented so the default can be chosen from measured
evidence rather than intuition (see ``docs/EXPERIMENTS.md``):

``entropy``
    Pure score-aware utility. ``other`` competes on the same footing as any
    typed attribute and wins only when its expected value is genuinely higher.

``other_first``
    Open with the broad discovery question, capped, then fall back to entropy.

``hybrid``
    Use the broad question while the posterior is still diffuse and it keeps
    paying off, then switch to entropy-selected typed attributes. Stops
    immediately once a broad question returns nothing new.

All three share the same guard rails: never re-ask an exhausted attribute,
never exceed the configured ``other`` cap, and always degrade to ``None``
rather than looping.
"""

from __future__ import annotations

from .config import ALLOWED_ATTRIBUTES, PolicyConfig
from .questions import QuestionSelector
from .state import SessionState

# Attributes the customer model can never produce an answer for are still
# offered to the selector: answerability handles them, and keeping them in the
# candidate set means a differently behaved private evaluator can still surface
# them.
CANDIDATE_ATTRIBUTES: tuple[str, ...] = ALLOWED_ATTRIBUTES

# Static ordering for the ``fixed`` ablation: a sensible hand-written sequence
# with no adaptivity, used as the control against entropy-based selection.
FIXED_ORDER: tuple[str, ...] = (
    "material", "feature", "color", "style", "use_case", "size", "category",
    "brand", "budget", "other",
)


def _available(state: SessionState, config: PolicyConfig) -> tuple[str, ...]:
    blocked = state.blocked_attributes()
    attributes = [a for a in CANDIDATE_ATTRIBUTES if a not in blocked]
    if state.other_asks >= config.max_other_asks or not state.other_productive:
        attributes = [a for a in attributes if a != "other"]
    return tuple(attributes)


def _discovery_bonus(state: SessionState, config: PolicyConfig) -> dict[str, float]:
    """Extra credit for a broad question while the picture is still vague."""
    known = len(state.constraints)
    if known >= config.other_until_constraints:
        return {}
    if state.other_asks >= config.max_other_asks or not state.other_productive:
        return {}
    vagueness = 1.0 - (known / max(config.other_until_constraints, 1))
    return {"other": vagueness}


def select_attribute(
    state: SessionState,
    selector: QuestionSelector,
    scored: list[tuple[int, float]],
    config: PolicyConfig,
    intent_mode: str = "uncertain",
) -> tuple[str | None, dict[str, dict[str, float]]]:
    """Choose the next ``ask_attribute``. Returns (attribute, diagnostics)."""
    attributes = _available(state, config)
    if not attributes:
        return None, {}

    intent_enabled = config.use_intent_policy
    policy = config.question_policy

    if policy == "fixed":
        # No adaptivity: walk a static order, skipping anything already asked
        # or known to be exhausted.
        for attribute in FIXED_ORDER:
            if attribute in attributes and attribute not in state.asked:
                return attribute, {}
        return None, {}

    if policy == "other_first":
        if "other" in attributes and state.other_asks < config.max_other_asks and state.other_productive:
            return "other", {}
        report = selector.evaluate(
            scored, state.disclosed_keys, state.asked, state.blocked_attributes(),
            tuple(a for a in attributes if a != "other"),
        )
        return selector.best_attribute(report), report

    bonus = _discovery_bonus(state, config) if policy == "hybrid" else {}
    if intent_enabled and intent_mode == "buying":
        # Do not force broad discovery when the opening already contains a
        # concrete need, but retain ``other`` as a safe fallback for a private
        # simulator that discloses an untyped requirement.
        bonus = {name: value for name, value in bonus.items() if name != "other"}
    if intent_enabled and intent_mode == "browsing" and "other" in attributes:
        bonus = {**bonus, "other": bonus.get("other", 0.0) + 0.08}
    report = selector.evaluate(
        scored,
        state.disclosed_keys,
        state.asked,
        state.blocked_attributes(),
        attributes,
        discovery_bonus=bonus,
    )
    attribute = selector.best_attribute(report)

    if attribute is None:
        # Nothing scored positively. Prefer an unasked attribute with any
        # answerability at all before giving up entirely.
        fallback = [
            (metrics["answerability"], name)
            for name, metrics in report.items()
            if name not in state.asked and metrics["answerability"] > 0.05
        ]
        if fallback:
            fallback.sort(key=lambda item: (-item[0], item[1]))
            return fallback[0][1], report
    return attribute, report
