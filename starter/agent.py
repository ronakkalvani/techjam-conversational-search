"""EntropyShop - an information-theoretic conversational shopping agent.

This module holds the official ``Agent`` interface and the per-turn
orchestration only. The reasoning lives in focused modules:

    user message
      -> constraints.parse_message      observable constraint parsing
      -> state.SessionState             session belief update
      -> retrieval.Retriever            multi-route candidate pool
      -> ranking.Ranker                 deterministic fusion + rerank
      -> top-k recommendations
      -> questions.QuestionSelector     score-aware information gain
      -> policies.select_attribute      next ask_attribute

The runtime is fully offline and deterministic: no network, no model call, no
random source, and therefore zero token cost. Identical inputs always yield
identical output.
"""

from __future__ import annotations

from pathlib import Path

from .catalog import Catalog
from .config import DEFAULT_CONFIG, AgentConfig
from .constraints import parse_message
from .explanations import compose_message
from .policies import select_attribute
from .questions import QuestionSelector
from .ranking import Ranker
from .retrieval import Retriever
from .state import SessionState
from .text import content_tokens

_SAFE_RESPONSE = {
    "message": "Here are the closest matches I found.",
    "ask_attribute": None,
    "recommendations": [],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
}


class Agent:
    """Deterministic, offline conversational shopping agent."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
        catalog: Catalog | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        # A prebuilt catalog can be shared across agents (experiment sweeps).
        # It is immutable, so sharing cannot leak state between runs.
        self.catalog = catalog or Catalog(
            catalog_path, max_df_ratio=self.config.ranking.max_df_ratio
        )
        self.retriever = Retriever(self.catalog)
        self.ranker = Ranker(self.catalog, self.retriever, self.config.ranking)
        self.selector = QuestionSelector(self.catalog, self.config.questions)
        self._sessions: dict[str, SessionState] = {}

    # -- official interface ------------------------------------------------

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Start a fresh, isolated session."""
        self._sessions[str(session_id)] = SessionState(
            session_id=str(session_id),
            user_profile=user_profile if isinstance(user_profile, dict) else {},
        )

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        """Answer one customer turn. Never raises."""
        try:
            return self._respond(str(session_id), user_message, int(turn), int(top_k))
        except Exception:  # pragma: no cover - contract safety net
            return {**_SAFE_RESPONSE, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    # -- orchestration -----------------------------------------------------

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            # Defensive: a missing reset must not cost the whole session.
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state

        limit = top_k if isinstance(top_k, int) and top_k > 0 else 10
        limit = min(limit, self._turn_budget(turn))
        gained_evidence = self._update_state(state, user_message, turn)

        scored = self._rank(state)
        docs = [doc for doc, _score in scored]

        attribute, _report = select_attribute(state, self.selector, scored, self.config.policy)
        if attribute in state.blocked_attributes():
            attribute = None
        state.note_asked(attribute)

        recommendations = [{"parent_asin": self.catalog.ids[doc]} for doc in docs[:limit]]
        shown = [item["parent_asin"] for item in recommendations]
        state.last_recommendations = shown
        # Anything already shown and not converted is, by the protocol, not the
        # target: remember it so later turns present fresh candidates.
        for parent_asin in shown:
            if parent_asin not in state.previously_shown:
                state.previously_shown.append(parent_asin)

        state.turns_without_evidence = 0 if gained_evidence else state.turns_without_evidence + 1

        return {
            "message": compose_message(attribute, len(recommendations), len(state.constraints)),
            "ask_attribute": attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _turn_budget(self, turn: int) -> int:
        """How many recommendations this turn may return."""
        budget = self.config.policy.turn_budget
        if not budget:
            return 10
        index = min(max(turn, 1) - 1, len(budget) - 1)
        return max(1, int(budget[index]))

    # -- state update ------------------------------------------------------

    def _update_state(self, state: SessionState, user_message: str, turn: int) -> bool:
        """Fold one customer turn into the session belief.

        Returns whether the turn produced new evidence.
        """
        message = str(user_message or "")
        state.messages.append(message)
        parsed = parse_message(message, turn)
        gained = False

        if parsed.category and not state.category:
            state.category = parsed.category
            key, confidence = self.retriever.resolve_bucket(parsed.category)
            state.bucket_key, state.bucket_confidence = key, confidence
            gained = True

        if parsed.kind == "opening":
            state.intent_mode = "buying" if parsed.new_constraints else "browsing"

        if parsed.kind == "override":
            state.constraints.apply_override(parsed.override_value, turn)
            state.override_count += 1
            # Scores change materially and pre-override turns could not convert,
            # so suppression of earlier recommendations no longer applies.
            state.previously_shown.clear()
            state.exhausted.clear()
            state.other_productive = state.other_asks < self.config.policy.max_other_asks
            gained = True

        elif parsed.new_constraints:
            for text in parsed.new_constraints:
                hard = parsed.kind in ("opening", "disclosure")
                if state.constraints.add(text, turn, hard=hard) is not None:
                    gained = True

        if parsed.exhausted_attribute:
            state.exhausted.add(parsed.exhausted_attribute)
            if parsed.exhausted_attribute == "other" or state.last_asked == "other":
                state.other_productive = False
        elif parsed.kind == "exhausted" and state.last_asked:
            state.exhausted.add(state.last_asked)
            if state.last_asked == "other":
                state.other_productive = False

        if parsed.unavailable_attribute:
            state.unavailable.add(parsed.unavailable_attribute)
        elif parsed.kind == "no_preference" and state.last_asked:
            state.unavailable.add(state.last_asked)

        if parsed.residual:
            state.residual_tokens.extend(content_tokens(parsed.residual))

        # An unstructured turn still contributes lexical evidence.
        if parsed.kind == "statement" and message.strip():
            state.residual_tokens.extend(content_tokens(message))

        return gained

    # -- ranking -----------------------------------------------------------

    def _rank(self, state: SessionState) -> list[tuple[int, float]]:
        ranking_config = self.config.ranking
        policy = self.config.policy

        constraints = state.constraints.texts_for_ranking(ranking_config.soft_constraint_weight)
        lexical_tokens = state.lexical_tokens()

        query_tokens: list[str] = []
        for text, _weight, _attribute in constraints:
            query_tokens.extend(content_tokens(text))
        if state.category:
            query_tokens.extend(content_tokens(state.category))
        query_tokens.extend(lexical_tokens)
        query_tokens = list(dict.fromkeys(query_tokens))[:80]

        pool, bucket_members, _lexical = self.retriever.build_pool(
            state.bucket_key,
            query_tokens,
            ranking_config.bucket_pool_limit,
            ranking_config.lexical_pool_limit,
        )

        scored = self.ranker.rank(
            pool=pool,
            bucket_members=bucket_members,
            constraints=constraints,
            lexical_tokens=lexical_tokens,
            category=state.category,
            profile_tokens=state.profile_tokens,
            use_profile=policy.use_profile_prior,
            use_popularity=policy.use_popularity_prior,
        )

        if policy.enable_exploration and state.previously_shown:
            shown = set(state.previously_shown)
            penalty = ranking_config.w_shown_penalty
            adjusted = [
                (doc, score - penalty if self.catalog.ids[doc] in shown else score)
                for doc, score in scored
            ]
            adjusted.sort(key=lambda item: (-item[1], self.catalog.ids[item[0]]))
            return adjusted
        return scored
