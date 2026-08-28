"""Per-session conversational state.

Sessions are fully isolated: nothing is shared between ``session_id`` values
except the immutable catalog indexes. State stays small - identifiers, short
strings and a bounded candidate list - so thousands of sessions cost little.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constraints import ConstraintSet
from .text import content_tokens


@dataclass
class SessionState:
    session_id: str
    user_profile: dict = field(default_factory=dict)

    # Observable conversation.
    messages: list[str] = field(default_factory=list)
    residual_tokens: list[str] = field(default_factory=list)

    # Structured belief.
    constraints: ConstraintSet = field(default_factory=ConstraintSet)
    category: str | None = None
    bucket_key: str | None = None
    bucket_confidence: float = 0.0

    # Question bookkeeping.
    asked: set[str] = field(default_factory=set)
    exhausted: set[str] = field(default_factory=set)
    unavailable: set[str] = field(default_factory=set)
    last_asked: str | None = None
    other_asks: int = 0
    other_productive: bool = True

    # Intent estimate: "buying", "browsing" or "uncertain".
    intent_mode: str = "uncertain"
    override_count: int = 0

    # Ranking memory.
    last_recommendations: list[str] = field(default_factory=list)
    previously_shown: list[str] = field(default_factory=list)
    turns_without_evidence: int = 0
    exploration_offset: int = 0

    @property
    def profile_tokens(self) -> list[str]:
        tags = self.user_profile.get("preference_tags") if isinstance(self.user_profile, dict) else None
        if not isinstance(tags, list):
            return []
        tokens: list[str] = []
        for tag in tags:
            tokens.extend(content_tokens(str(tag)))
        return list(dict.fromkeys(tokens))[:12]

    @property
    def disclosed_keys(self) -> set[str]:
        """Normalised text of every requirement the customer has stated."""
        return {constraint.key for constraint in self.constraints.items}

    def note_asked(self, attribute: str | None) -> None:
        if not attribute:
            self.last_asked = None
            return
        self.last_asked = attribute
        self.asked.add(attribute)
        if attribute == "other":
            self.other_asks += 1

    def blocked_attributes(self) -> set[str]:
        """Attributes that would waste a turn if asked again."""
        return self.exhausted | self.unavailable

    def lexical_tokens(self) -> list[str]:
        """Free-text tokens that did not become structured constraints."""
        return list(dict.fromkeys(self.residual_tokens))[:60]
