"""Candidate fusion and reranking.

Scoring runs in two stages so that expensive signals never touch the whole
pool:

* **Stage 1** (whole pool, cheap): IDF-weighted constraint token coverage,
  free-text lexical relevance, category compatibility, visible priors.
* **Stage 2** (shortlist only): exact phrase containment, facet agreement and
  conflict, budget compatibility.

Every component is normalised to roughly [0, 1] before the weighted sum, so a
component cannot dominate merely because of its numeric scale. Ties break on
``parent_asin`` so ordering is fully deterministic.

    score(d) = w_constraint * constraint_coverage(d)
             + w_phrase_bonus * phrase_containment(d)      [stage 2]
             + w_lexical     * lexical(d)
             + w_category    * category_overlap(d)
             + w_bucket      * bucket_membership(d)
             + w_facet       * facet_agreement(d)          [stage 2]
             - w_conflict    * facet_conflict(d)           [stage 2]
             + w_budget      * budget_fit(d)               [stage 2]
             + w_prior       * popularity_prior(d)
             + w_profile     * profile_fit(d)
"""

from __future__ import annotations

from .catalog import Catalog
from .config import RankingConfig
from .facets import constraint_facet_values
from .retrieval import Retriever
from .text import content_tokens, normalize

# Shortlist size handed to stage 2.
_SHORTLIST = 300


class Ranker:
    def __init__(self, catalog: Catalog, retriever: Retriever, config: RankingConfig) -> None:
        self.catalog = catalog
        self.retriever = retriever
        self.config = config

    # -- stage 1 -----------------------------------------------------------

    def _constraint_coverage(
        self,
        pool: set[int],
        constraints: list[tuple[str, float, str]],
    ) -> tuple[dict[int, float], float]:
        """IDF-weighted share of each constraint's tokens present in a product."""
        accumulator: dict[int, float] = {}
        total_weight = 0.0
        for text, weight, _attribute in constraints:
            tokens = content_tokens(text)
            if not tokens:
                continue
            idfs = {token: self.catalog.token_idf(token) for token in tokens}
            denominator = sum(value for value in idfs.values() if value > 0.0)
            if denominator <= 0.0:
                continue
            total_weight += weight
            for token, idf in idfs.items():
                if idf <= 0.0:
                    continue
                share = weight * (idf / denominator)
                for doc in self.retriever.postings_set(token) & pool:
                    accumulator[doc] = accumulator.get(doc, 0.0) + share
        return accumulator, total_weight

    def _lexical_component(self, pool: set[int], tokens: list[str]) -> dict[int, float]:
        if not tokens:
            return {}
        idfs = {token: self.catalog.token_idf(token) for token in tokens}
        denominator = sum(value for value in idfs.values() if value > 0.0)
        if denominator <= 0.0:
            return {}
        scores: dict[int, float] = {}
        for token, idf in idfs.items():
            if idf <= 0.0:
                continue
            share = idf / denominator
            for doc in self.retriever.postings_set(token) & pool:
                scores[doc] = scores.get(doc, 0.0) + share
        return scores

    def _category_component(self, pool: set[int], category: str | None) -> dict[int, float]:
        if not category:
            return {}
        wanted = set(content_tokens(category))
        if not wanted:
            return {}
        scores: dict[int, float] = {}
        for doc in pool:
            doc_tokens = set(content_tokens(self.catalog.category_text[doc]))
            if not doc_tokens:
                continue
            overlap = len(wanted & doc_tokens)
            if overlap:
                scores[doc] = overlap / len(wanted)
        return scores

    # -- stage 2 -----------------------------------------------------------

    def _phrase_bonus(self, doc: int, constraints: list[tuple[str, float, str]], total_weight: float) -> float:
        if total_weight <= 0.0:
            return 0.0
        text = self.catalog.text[doc]
        earned = 0.0
        for constraint_text, weight, _attribute in constraints:
            needle = normalize(constraint_text)
            if len(needle) >= 4 and needle in text:
                earned += weight
        return earned / total_weight

    def _facet_component(
        self,
        doc: int,
        wanted_facets: dict[str, set[str]],
    ) -> tuple[float, float]:
        """Return (agreement, conflict) in [0, 1]."""
        if not wanted_facets:
            return 0.0, 0.0
        facets = self.catalog.facets(doc)
        agree = 0
        conflict = 0
        considered = 0
        for attribute, values in wanted_facets.items():
            if attribute == "budget":
                continue
            product_values = facets.get(attribute) or set()
            considered += 1
            if not product_values:
                continue  # unknown, never penalised
            if values & product_values:
                agree += 1
            else:
                conflict += 1
        if considered == 0:
            return 0.0, 0.0
        return agree / considered, conflict / considered

    def _budget_component(self, doc: int, wanted_price: float | None) -> float:
        if wanted_price is None:
            return 0.0
        price = self.catalog.price[doc]
        if price is None:
            return 0.0
        gap = abs(price - wanted_price) / max(wanted_price, 1.0)
        return max(0.0, 1.0 - min(gap, 1.0))

    def _profile_component(self, doc: int, profile_tokens: list[str]) -> float:
        if not profile_tokens:
            return 0.0
        text = self.catalog.text[doc]
        hits = sum(1 for token in profile_tokens if token in text)
        return hits / len(profile_tokens)

    # -- public API --------------------------------------------------------

    def rank(
        self,
        pool: list[int],
        bucket_members: set[int],
        constraints: list[tuple[str, float, str]],
        lexical_tokens: list[str],
        category: str | None,
        profile_tokens: list[str],
        use_profile: bool,
        use_popularity: bool,
    ) -> list[tuple[int, float]]:
        """Score the pool and return (doc, score) sorted best first."""
        config = self.config
        pool_set = set(pool)

        coverage, total_weight = self._constraint_coverage(pool_set, constraints)
        lexical = self._lexical_component(pool_set, lexical_tokens)
        category_scores = self._category_component(pool_set, category)

        stage1: list[tuple[float, int]] = []
        for doc in pool:
            score = 0.0
            if total_weight > 0.0:
                score += config.w_constraint * (coverage.get(doc, 0.0) / total_weight)
            score += config.w_lexical * lexical.get(doc, 0.0)
            score += config.w_category * category_scores.get(doc, 0.0)
            if doc in bucket_members:
                score += config.w_bucket
            if use_popularity:
                score += config.w_prior * self.catalog.priors[doc]
            stage1.append((score, doc))

        stage1.sort(key=lambda item: (-item[0], self.catalog.ids[item[1]]))
        shortlist = stage1[:_SHORTLIST]

        # Facet expectations implied by the active constraints.
        wanted_facets: dict[str, set[str]] = {}
        for text, weight, _attribute in constraints:
            if weight <= 0.0:
                continue
            for attribute, values in constraint_facet_values(text).items():
                wanted_facets.setdefault(attribute, set()).update(values)
        wanted_price: float | None = None
        for value in sorted(wanted_facets.get("budget", set())):
            try:
                wanted_price = float(value)
                break
            except ValueError:
                continue

        refined: list[tuple[float, int]] = []
        for score, doc in shortlist:
            total = score
            if constraints and total_weight > 0.0:
                total += config.w_phrase_bonus * self._phrase_bonus(doc, constraints, total_weight)
            agreement, conflict = self._facet_component(doc, wanted_facets)
            total += config.w_facet * agreement
            total -= config.w_conflict * conflict
            total += config.w_budget * self._budget_component(doc, wanted_price)
            if use_profile:
                total += config.w_profile * self._profile_component(doc, profile_tokens)
            refined.append((total, doc))

        refined.sort(key=lambda item: (-item[0], self.catalog.ids[item[1]]))
        tail = [(score, doc) for score, doc in stage1[_SHORTLIST:]]
        return [(doc, score) for score, doc in refined + tail]
