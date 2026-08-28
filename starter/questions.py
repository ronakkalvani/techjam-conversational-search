"""Question-value estimation.

The agent keeps a weighted posterior over catalog candidates and picks the
clarification question with the highest expected value, not merely the highest
entropy reduction.

    H(C)        = -sum_c p(c) log2 p(c)
    ERE(a)      =  sum_v P(v) H(C | answer group v)
    IG(a)       =  H(C) - ERE(a)

    Utility(a)  =  w_ig       * normalised_information_gain(a)
                +  w_top10    * expected_top10_mass_gain(a)
                +  w_mrr      * expected_reciprocal_rank_gain(a)
                +  w_coverage * answerability(a)
                +  w_discover * discovery_bonus(a)
                -  w_missing  * missing_value_probability(a)
                -  w_repeat   * repeated_or_exhausted_penalty(a)
                -  w_turn     * additional_turn_cost

Answer groups come from a *customer model*: for each candidate product we
predict which short requirements its owner would still quote, type each one,
and group candidates by the answer a question about attribute ``a`` would
elicit. An attribute nobody could answer therefore scores zero answerability
however finely it would partition the catalog - which is why questions like
``brand`` correctly lose to ``material`` even though brands are highly
distinctive.

Costs are bounded: estimates run over the top ``entropy_pool`` candidates only.
"""

from __future__ import annotations

import math

from .catalog import Catalog
from .config import QuestionConfig
from .facets import classify_constraint
from .text import normalize

# Attribute asked when nothing else is informative.
FALLBACK_ATTRIBUTE = "feature"


def _entropy(weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    result = 0.0
    for weight in weights:
        if weight <= 0.0:
            continue
        p = weight / total
        result -= p * math.log2(p)
    return result


def _top_mass(weights: list[float], k: int = 10) -> float:
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    return sum(sorted(weights, reverse=True)[:k]) / total


def _expected_rr(weights: list[float], k: int = 10) -> float:
    """Expected reciprocal rank if the target is drawn from this group."""
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    ordered = sorted(weights, reverse=True)[:k]
    return sum(weight / total / (index + 1) for index, weight in enumerate(ordered))


class QuestionSelector:
    def __init__(self, catalog: Catalog, config: QuestionConfig) -> None:
        self.catalog = catalog
        self.config = config
        self._type_cache: dict[int, tuple[tuple[str, str], ...]] = {}

    # -- customer model ----------------------------------------------------

    def typed_constraints(self, doc: int) -> tuple[tuple[str, str], ...]:
        """(normalised text, attribute) pairs this product's owner might quote."""
        cached = self._type_cache.get(doc)
        if cached is None:
            pairs = []
            for text in self.catalog.predicted_constraints[doc][: self.config.model_constraints_per_product]:
                pairs.append((normalize(text), classify_constraint(text)))
            cached = tuple(pairs)
            if len(self._type_cache) > 60000:
                self._type_cache.clear()
            self._type_cache[doc] = cached
        return cached

    def _answer_group(self, doc: int, attribute: str, disclosed: set[str]) -> tuple[str, ...]:
        """What this candidate's owner would reveal if asked about ``attribute``."""
        limit = self.config.disclosures_per_answer
        matches: list[str] = []
        for text, kind in self.typed_constraints(doc):
            if text in disclosed:
                continue
            if attribute == "other" or kind == attribute:
                matches.append(text)
                if len(matches) >= limit:
                    break
        return tuple(matches)

    # -- utility -----------------------------------------------------------

    def posterior(self, scored: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """Softmax the ranking scores of the top candidates into weights."""
        pool = scored[: self.config.entropy_pool]
        if not pool:
            return []
        best = max(score for _doc, score in pool)
        temperature = max(self.config.posterior_temperature, 1e-3)
        weights = [(doc, math.exp((score - best) / temperature)) for doc, score in pool]
        total = sum(weight for _doc, weight in weights)
        if total <= 0.0:
            uniform = 1.0 / len(weights)
            return [(doc, uniform) for doc, _ in weights]
        return [(doc, weight / total) for doc, weight in weights]

    def evaluate(
        self,
        scored: list[tuple[int, float]],
        disclosed: set[str],
        asked: set[str],
        exhausted: set[str],
        candidate_attributes: tuple[str, ...],
        discovery_bonus: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Utility breakdown for every candidate attribute."""
        weights = self.posterior(scored)
        report: dict[str, dict[str, float]] = {}
        if not weights:
            return report

        all_weights = [weight for _doc, weight in weights]
        base_entropy = _entropy(all_weights)
        base_top10 = _top_mass(all_weights)
        base_rr = _expected_rr(all_weights)
        max_entropy = math.log2(len(all_weights)) if len(all_weights) > 1 else 1.0
        bonus = discovery_bonus or {}

        for attribute in candidate_attributes:
            groups: dict[tuple[str, ...], list[float]] = {}
            for doc, weight in weights:
                key = self._answer_group(doc, attribute, disclosed)
                groups.setdefault(key, []).append(weight)

            missing_mass = sum(groups.get((), []))
            residual = 0.0
            expected_top10 = 0.0
            expected_rr = 0.0
            for members in groups.values():
                mass = sum(members)
                if mass <= 0.0:
                    continue
                residual += mass * _entropy(members)
                expected_top10 += mass * _top_mass(members)
                expected_rr += mass * _expected_rr(members)

            information_gain = max(0.0, base_entropy - residual)
            normalised_ig = information_gain / max_entropy if max_entropy > 0 else 0.0
            answerability = 1.0 - missing_mass

            utility = (
                self.config.w_ig * normalised_ig
                + self.config.w_top10 * max(0.0, expected_top10 - base_top10)
                + self.config.w_mrr * max(0.0, expected_rr - base_rr)
                + self.config.w_coverage * answerability
                + self.config.w_discover * bonus.get(attribute, 0.0)
                - self.config.w_missing * missing_mass
                - self.config.w_repeat * (1.0 if attribute in exhausted else 0.0)
                - 0.25 * self.config.w_repeat * (1.0 if attribute in asked else 0.0)
                - self.config.w_turn
            )

            report[attribute] = {
                "utility": utility,
                "information_gain": information_gain,
                "normalised_ig": normalised_ig,
                "answerability": answerability,
                "missing_mass": missing_mass,
                "expected_top10_gain": expected_top10 - base_top10,
                "expected_rr_gain": expected_rr - base_rr,
                "groups": float(len(groups)),
            }
        return report

    def best_attribute(
        self,
        report: dict[str, dict[str, float]],
        minimum_utility: float = 0.0,
    ) -> str | None:
        """Highest-utility attribute, deterministic on ties."""
        if not report:
            return None
        ordered = sorted(report.items(), key=lambda item: (-item[1]["utility"], item[0]))
        attribute, metrics = ordered[0]
        if metrics["utility"] <= minimum_utility:
            return None
        return attribute
