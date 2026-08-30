"""Small, deterministic offline vector retrieval.

This is deliberately self-contained: it does not download a model or require
NumPy/PyTorch. Each visible catalog document and query is represented by a
sparse TF-IDF vector made from normalised words, word pairs, subword n-grams,
and a small set of general apparel concept aliases. Cosine similarity is used
to add candidates to the normal category/lexical union.

It is a semantic-style vector route rather than a pretrained neural encoder.
That trade-off keeps the competition runtime reproducible and offline while
still covering vocabulary variation that exact token retrieval misses.
"""

from __future__ import annotations

import math
from array import array
from collections import Counter

from .text import TOKEN_RE, content_tokens, normalize, singular


# General language aliases, not target identifiers or evaluator labels. They
# provide a small distributional bridge when the customer and catalog use
# different but common apparel wording.
_CONCEPT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("athletic_shoe", ("sneaker", "sneakers", "trainer", "trainers", "athletic shoe", "running shoe")),
    ("footwear", ("shoe", "shoes", "footwear")),
    ("handbag", ("handbag", "handbags", "purse", "purses", "tote")),
    ("waterproof", ("waterproof", "water resistant", "rainproof", "weatherproof")),
    ("casual", ("casual", "everyday", "daily")),
    ("formal", ("formal", "dressy", "business")),
    ("comfort", ("comfortable", "comfort", "comfy", "cushioned")),
    ("lightweight", ("lightweight", "light")),
    ("durable", ("durable", "tough", "heavy duty")),
    ("breathable", ("breathable", "ventilated")),
    ("stretch", ("stretch", "stretchy", "elastic")),
    ("winter", ("winter", "cold weather")),
    ("athletic", ("gym", "workout", "training", "athletic")),
)


def _contains_term(text: str, term: str) -> bool:
    padded = f" {text} "
    return f" {normalize(term)} " in padded


def _char_features(token: str) -> set[str]:
    token = singular(token)
    if len(token) < 4:
        return set()
    padded = f"^{token}$"
    return {f"c:{padded[index:index + 3]}" for index in range(len(padded) - 2)}


def semantic_features(text: str, surface_text: str | None = None) -> set[str]:
    """Return a bounded sparse feature set for one document or query."""
    normalised = normalize(text)
    tokens = [singular(token) for token in content_tokens(normalised)[:80]]
    features = {f"w:{token}" for token in tokens}

    for left, right in zip(tokens, tokens[1:]):
        features.add(f"b:{left}|{right}")

    for concept, aliases in _CONCEPT_GROUPS:
        if any(_contains_term(normalised, alias) for alias in aliases):
            features.add(f"concept:{concept}")

    # Subword features are limited to title/category surface text so the
    # index stays compact while still helping with morphology and typos.
    surface = normalize(surface_text if surface_text is not None else text)
    for token in [singular(item) for item in TOKEN_RE.findall(surface)[:30]]:
        features.update(_char_features(token))
    return features


class SemanticIndex:
    """Sparse TF-IDF index over the catalog's visible metadata."""

    def __init__(self, catalog, max_df_ratio: float = 0.25) -> None:
        self.catalog = catalog
        self.max_df_ratio = max_df_ratio
        self._postings: dict[str, array] = {}
        self._idf: dict[str, float] = {}
        self._norms: list[float] = [0.0] * catalog.size
        self._build()

    def _build(self) -> None:
        document_features: list[set[str]] = []
        for doc in range(self.catalog.size):
            features = semantic_features(
                self.catalog.semantic_text[doc],
                self.catalog.semantic_surface[doc],
            )
            document_features.append(features)
            for feature in features:
                posting = self._postings.get(feature)
                if posting is None:
                    posting = array("i")
                    self._postings[feature] = posting
                posting.append(doc)

        limit = max(1, int(self.catalog.size * self.max_df_ratio))
        for feature, posting in list(self._postings.items()):
            frequency = len(posting)
            # Singleton features mostly encode unique product wording. They
            # are useful to exact lexical retrieval, but create a very large
            # vector index without helping cross-document similarity.
            if frequency < 2 or frequency > limit:
                del self._postings[feature]
                continue
            self._idf[feature] = math.log(1.0 + (self.catalog.size - frequency + 0.5) / (frequency + 0.5))

        for doc, features in enumerate(document_features):
            squared = sum(self._idf.get(feature, 0.0) ** 2 for feature in features)
            self._norms[doc] = math.sqrt(squared)

    def scores(self, query_text: str, limit: int) -> dict[int, float]:
        features = semantic_features(query_text)
        counts = Counter(features)
        query_weights = {
            feature: count * self._idf.get(feature, 0.0)
            for feature, count in counts.items()
            if self._idf.get(feature, 0.0) > 0.0
        }
        if not query_weights:
            return {}

        query_norm = math.sqrt(sum(value * value for value in query_weights.values()))
        if query_norm <= 0.0:
            return {}

        raw: dict[int, float] = {}
        for feature, query_weight in query_weights.items():
            idf = self._idf[feature]
            for doc in self._postings[feature]:
                raw[doc] = raw.get(doc, 0.0) + query_weight * idf

        scored = [
            (doc, value / (query_norm * self._norms[doc]))
            for doc, value in raw.items()
            if self._norms[doc] > 0.0
        ]
        scored.sort(key=lambda item: (-item[1], self.catalog.ids[item[0]]))
        return dict(scored[: max(1, limit)])
