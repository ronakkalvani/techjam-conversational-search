"""Catalog loading, normalisation and index construction.

The catalog is read once in ``Agent.__init__`` and is then immutable. Nothing
here mutates the source file, and every derived structure can be rebuilt
reproducibly from ``catalog.jsonl``.

Index layout
------------
``ids``            doc index -> parent_asin
``text``           doc index -> full lowercase text (phrase containment)
``postings``       token id -> array of doc indices (lexical scoring)
``bucket_docs``    coarse category key -> doc indices (primary candidate pool)
``priors``         doc index -> popularity/quality prior in [0, 1]

Memory footprint is roughly 120 MB for the 50,000-product frozen catalog, which
is why token sets are stored as interned integer arrays rather than Python sets
of strings.
"""

from __future__ import annotations

import json
import math
import re
from array import array
from pathlib import Path

from .facets import extract_facets
from .text import TOKEN_RE, flatten, normalize

# Root category components that carry no discriminative signal. The customer's
# opening line names the item at this same granularity, so the exclusion set is
# kept identical to the harness's: dropping more components (e.g. a bare
# "Jewelry") would shift the bucket key and silently lose recall.
_ROOT_CATEGORIES = frozenset(
    {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
)

_CLEAN_EDGE = " -;,.\t\n"


def coarse_category(values: list[str]) -> str:
    """Collapse a category path to its two most specific components.

    The customer's opening line names the kind of item they want at roughly
    this granularity, so this doubles as the primary retrieval bucket key.
    """
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _ROOT_CATEGORIES:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def _flatten_values(value: object) -> list[str]:
    """Split heterogeneous metadata into individually quotable strings."""
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(_CLEAN_EDGE)[:limit].rstrip()


class Catalog:
    """Immutable, in-memory view of the frozen competition catalog."""

    def __init__(self, catalog_path: str | Path, max_df_ratio: float = 0.25) -> None:
        self.path = Path(catalog_path)

        self.ids: list[str] = []
        self.index_of: dict[str, int] = {}
        self.text: list[str] = []
        self.titles: list[str] = []
        self.category_text: list[str] = []
        # Compact visible text used by the optional offline vector index.
        self.semantic_text: list[str] = []
        self.semantic_surface: list[str] = []
        self.bucket_key: list[str] = []
        self.price: list[float | None] = []
        self.predicted_constraints: list[tuple[str, ...]] = []

        self.vocab: dict[str, int] = {}
        self.postings: dict[int, array] = {}
        self._doc_freq: list[int] = []

        self.bucket_docs: dict[str, list[int]] = {}
        self.priors: list[float] = []

        self._facet_cache: dict[int, dict[str, set[str]]] = {}
        self._products: list[dict] = []

        self._load()
        self._finalise(max_df_ratio)

    # -- construction ------------------------------------------------------

    def _load(self) -> None:
        postings_build: dict[int, array] = {}
        ratings: list[tuple[float, float]] = []

        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                product = json.loads(line)
                parent_asin = str(product.get("parent_asin", "")).strip()
                if not parent_asin or parent_asin in self.index_of:
                    continue

                doc = len(self.ids)
                self.ids.append(parent_asin)
                self.index_of[parent_asin] = doc
                self._products.append(product)

                title = flatten(product.get("title"))
                categories = product.get("categories") or []
                category_text = normalize(flatten(categories))
                core = " ".join(
                    (
                        title,
                        flatten(product.get("features")),
                        flatten(product.get("details")),
                        flatten(categories),
                        flatten(product.get("store")),
                    )
                )
                full = normalize(core + " " + flatten(product.get("description")))

                self.titles.append(normalize(title))
                self.category_text.append(category_text)
                self.semantic_text.append(normalize(core))
                self.semantic_surface.append(normalize(f"{title} {flatten(categories)}"))
                self.text.append(full)

                key = coarse_category([str(value) for value in categories])
                self.bucket_key.append(key)
                self.bucket_docs.setdefault(key, []).append(doc)

                try:
                    price = float(product["price"]) if product.get("price") not in (None, "") else None
                except (TypeError, ValueError):
                    price = None
                self.price.append(price)

                self.predicted_constraints.append(self._model_constraints(product, full))

                # Lexical postings over the core fields (description excluded:
                # it is long, repetitive, and hurts precision more than it helps).
                seen: set[int] = set()
                for token in TOKEN_RE.findall(normalize(core)):
                    if len(token) < 2:
                        continue
                    token_id = self.vocab.get(token)
                    if token_id is None:
                        token_id = len(self.vocab)
                        self.vocab[token] = token_id
                        self._doc_freq.append(0)
                    seen.add(token_id)
                for token_id in seen:
                    self._doc_freq[token_id] += 1
                    bucket = postings_build.get(token_id)
                    if bucket is None:
                        bucket = array("i")
                        postings_build[token_id] = bucket
                    bucket.append(doc)

                try:
                    average = float(product.get("average_rating") or 0.0)
                except (TypeError, ValueError):
                    average = 0.0
                try:
                    count = float(product.get("rating_number") or 0.0)
                except (TypeError, ValueError):
                    count = 0.0
                ratings.append((average, count))

        self.postings = postings_build
        self._build_priors(ratings)

    def _model_constraints(self, product: dict, corpus: str) -> tuple[str, ...]:
        """Requirements this customer would plausibly quote for this product.

        This is a *model* of disclosure behaviour, not a guarantee: the customer
        quotes short phrases from the product's own features and details, with
        material and colour surfacing first because they are the most salient
        way to describe apparel. It drives question-value estimation only, and
        the agent falls back to observed answers whenever the model is wrong.
        """
        from .facets import COLOR_RE, MATERIAL_RE  # local import: avoids a cycle

        candidates = [
            *_flatten_values(product.get("features")),
            *_flatten_values(product.get("details")),
        ]
        material = MATERIAL_RE.search(corpus)
        color = COLOR_RE.search(corpus)
        if material:
            candidates.insert(0, material.group(1))
        if color:
            candidates.insert(1, f"color: {color.group(1)}")
        if product.get("price") not in (None, ""):
            candidates.append(f"budget around ${product['price']}")

        cleaned: list[str] = []
        for item in candidates:
            value = _clean_constraint(item)
            if value and value not in cleaned:
                cleaned.append(value)
            if len(cleaned) >= 4:
                break
        if not cleaned:
            cleaned = [_clean_constraint(flatten(product.get("title")) or "product")]
        return tuple(cleaned[:4])

    def _build_priors(self, ratings: list[tuple[float, float]]) -> None:
        """Popularity/quality prior in [0, 1] from visible rating fields only."""
        max_log = max((math.log1p(count) for _, count in ratings), default=1.0) or 1.0
        self.priors = [
            (0.6 * (average / 5.0) + 0.4 * (math.log1p(count) / max_log))
            for average, count in ratings
        ]

    def _finalise(self, max_df_ratio: float) -> None:
        self.size = len(self.ids)
        limit = max(1, int(self.size * max_df_ratio))
        self.idf: dict[int, float] = {}
        for token_id, frequency in enumerate(self._doc_freq):
            if frequency <= 0 or frequency > limit:
                continue
            self.idf[token_id] = math.log(1.0 + (self.size - frequency + 0.5) / (frequency + 0.5))
        self.bucket_docs = {key: docs for key, docs in self.bucket_docs.items()}

    # -- lookups -----------------------------------------------------------

    def token_id(self, token: str) -> int | None:
        return self.vocab.get(token)

    def token_idf(self, token: str) -> float:
        token_id = self.vocab.get(token)
        if token_id is None:
            # Unseen term: maximally rare, but capped so a typo cannot dominate.
            return 6.0
        return self.idf.get(token_id, 0.0)

    def facets(self, doc: int) -> dict[str, set[str]]:
        """Lazily extracted, cached facet sets for one product."""
        cached = self._facet_cache.get(doc)
        if cached is None:
            cached = extract_facets(self._products[doc])
            self._facet_cache[doc] = cached
        return cached

    def product(self, doc: int) -> dict:
        return self._products[doc]

    def bucket_candidates(self, key: str) -> list[int]:
        return self.bucket_docs.get(key, [])

    def bucket_keys(self) -> list[str]:
        return list(self.bucket_docs)
