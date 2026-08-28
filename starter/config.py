"""Tunable, deterministic configuration for the EntropyShop agent.

Every weight used by retrieval, ranking and question selection lives here so
that experiments in ``scripts/`` can sweep them without touching logic.

Nothing in this module reads the environment, the clock, or any random source:
identical configuration plus identical inputs must always produce identical
output.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


# Attributes the official contract allows in ``ask_attribute``.
ALLOWED_ATTRIBUTES: tuple[str, ...] = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)


@dataclass(frozen=True)
class RankingConfig:
    """Weights for the candidate fusion score.

    Route scores are each normalised to roughly [0, 1] before fusion so that no
    single component dominates by virtue of its numeric scale.
    """

    # Disclosed-constraint evidence. This is by far the strongest signal: the
    # simulated customer quotes phrases drawn from the target's own metadata.
    w_constraint: float = 1.0
    # Weight of an exact phrase containment on top of token coverage.
    w_phrase_bonus: float = 0.60
    # Weight applied to constraints that an override marked as superseded.
    soft_constraint_weight: float = 0.25

    # Free-text lexical relevance for anything the parser did not turn into a
    # structured constraint (keeps unrecognised paraphrases useful).
    w_lexical: float = 0.25

    # Category compatibility.
    w_category: float = 0.45
    # Extra additive boost for products inside the resolved category bucket.
    w_bucket: float = 0.35

    # Normalised facet agreement (material/color/size/... extracted from text).
    w_facet: float = 0.30
    # Penalty when the product explicitly contradicts a stated facet value.
    w_conflict: float = 0.45

    # Budget compatibility when a price constraint has been disclosed.
    w_budget: float = 0.35

    # Demotion for a product already shown in a scored top-10. If the session
    # is still running, that product was not the target, so it is pushed below
    # every fresh candidate. Large but finite: it degrades to a re-show rather
    # than an empty list when the pool is small.
    w_shown_penalty: float = 5.0

    # Visible popularity/quality priors. Deliberately small: they must never
    # outvote explicit session evidence.
    w_prior: float = 0.030
    # Soft aggregate-profile compatibility. Smaller still.
    w_profile: float = 0.020

    # Candidate pool sizes.
    bucket_pool_limit: int = 2000
    lexical_pool_limit: int = 1200
    # Tokens appearing in more than this fraction of the catalog are ignored
    # for scoring: they cost time and carry almost no discriminative weight.
    max_df_ratio: float = 0.25


@dataclass(frozen=True)
class QuestionConfig:
    """Weights for the score-aware question utility."""

    w_ig: float = 1.00          # normalised expected information gain
    w_top10: float = 0.35       # expected gain in top-10 posterior mass
    w_mrr: float = 0.25         # expected reciprocal-rank improvement
    w_coverage: float = 0.55    # answerability: will the customer reply usefully?
    w_discover: float = 0.20    # discovery bonus for broad questions
    w_missing: float = 0.45     # probability the answer is "no preference"
    w_repeat: float = 1.50      # penalty for re-asking an exhausted attribute
    w_turn: float = 0.02        # small per-turn cost of asking at all

    # Number of top-weighted candidates used for entropy estimates. Bounds the
    # per-turn cost of question selection.
    entropy_pool: int = 200
    # Softmax temperature converting ranking scores into a posterior. Lower is
    # sharper; too sharp collapses the posterior onto one candidate and makes
    # every question look worthless.
    posterior_temperature: float = 0.18
    # How many constraints the customer model assumes a product would yield.
    model_constraints_per_product: int = 4
    # How many constraints a single answer is assumed to disclose.
    disclosures_per_answer: int = 2


@dataclass(frozen=True)
class PolicyConfig:
    """Question-policy selection and recommendation behaviour."""

    # One of: "fixed", "entropy", "other_first", "hybrid".
    #
    # "other_first" ties with "hybrid" on the public set (0.95908 vs 0.95878 -
    # a single session's worth of difference). "hybrid" is the default anyway
    # because it degrades gracefully: it stops asking broad questions the moment
    # one stops paying off and falls back to entropy-selected typed attributes,
    # whereas "other_first" hard-codes the opening move. See docs/EXPERIMENTS.md.
    question_policy: str = "hybrid"

    # Hybrid/other_first: maximum number of broad "other" questions per session.
    max_other_asks: int = 2
    # Hybrid: only ask "other" while fewer than this many constraints are known.
    other_until_constraints: int = 4

    # Controlled exploration: products already shown in a scored top-10 are
    # demoted, so each turn presents fresh candidates.
    enable_exploration: bool = True

    # How many recommendations to return on each turn; the final entry repeats
    # for all later turns. Returning fewer than 10 early trades a little MTTC
    # for a better rank at conversion, because a target sitting at rank 7 today
    # is usually rank 1-2 tomorrow once its ten rivals are demoted and one more
    # requirement is known. ``(10,)`` is the Hit-Rate-safe default.
    turn_budget: tuple[int, ...] = (1, 2, 3, 5, 10)

    # Feature flags for ablations.
    #
    # The aggregate profile is off by default on measured evidence: its tags
    # ("fit", "comfort", "durability") match generic apparel copy almost
    # everywhere, so at ranking time it acts as a near-random tie-breaker among
    # otherwise equal candidates. Enabling it costs ~0.02 TechnicalScore.
    # The popularity prior is kept: it helps when evidence is thin.
    use_profile_prior: bool = False
    use_popularity_prior: bool = True


@dataclass(frozen=True)
class AgentConfig:
    ranking: RankingConfig = field(default_factory=RankingConfig)
    questions: QuestionConfig = field(default_factory=QuestionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def with_policy(self, **kwargs) -> "AgentConfig":
        """Return a copy with ``PolicyConfig`` fields overridden."""
        return replace(self, policy=replace(self.policy, **kwargs))

    def with_ranking(self, **kwargs) -> "AgentConfig":
        return replace(self, ranking=replace(self.ranking, **kwargs))

    def with_questions(self, **kwargs) -> "AgentConfig":
        return replace(self, questions=replace(self.questions, **kwargs))


DEFAULT_CONFIG = AgentConfig()
