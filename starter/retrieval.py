"""Multi-route candidate retrieval.

Two routes feed one pool:

* **Category route** - the customer names the kind of item they want in their
  opening line. Resolving that to a catalog bucket is the single most
  selective signal available (50,000 products down to a median of ~180).
* **Lexical route** - an IDF-weighted sweep of the inverted index over every
  token the customer has used.

The routes are unioned rather than intersected. Category resolution is a
*boost*, never an exclusive filter, so a paraphrased or unresolvable category
can never permanently eliminate the target.
"""

from __future__ import annotations

from .catalog import Catalog
from .semantic import SemanticIndex
from .text import content_tokens, normalize, token_variants


class Retriever:
    """Builds and scores the candidate pool for one turn."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._postings_cache: dict[int, set[int]] = {}
        self._bucket_tokens: dict[str, set[str]] = {
            key: set(content_tokens(key)) for key in catalog.bucket_docs
        }
        self._semantic_index: SemanticIndex | None = None

    # -- helpers -----------------------------------------------------------

    def postings_set(self, token: str) -> set[int]:
        """Doc indices containing ``token``, cached across turns."""
        token_id = self.catalog.token_id(token)
        if token_id is None:
            return set()
        cached = self._postings_cache.get(token_id)
        if cached is None:
            if len(self._postings_cache) > 30000:
                self._postings_cache.clear()
            cached = set(self.catalog.postings.get(token_id, ()))
            self._postings_cache[token_id] = cached
        return cached

    def resolve_bucket(self, category: str | None) -> tuple[str | None, float]:
        """Map a spoken category to a catalog bucket key.

        Returns the key and a confidence in [0, 1]. An exact match is the
        common case; the token-overlap fallback keeps a paraphrase useful.
        """
        if not category:
            return None, 0.0
        cleaned = " ".join(str(category).split())
        if cleaned in self.catalog.bucket_docs:
            return cleaned, 1.0

        # Case-insensitive exact match.
        lowered = cleaned.lower()
        for key in self.catalog.bucket_docs:
            if key.lower() == lowered:
                return key, 1.0

        wanted = set(content_tokens(cleaned))
        if not wanted:
            return None, 0.0
        best_key, best_score = None, 0.0
        for key, tokens in self._bucket_tokens.items():
            if not tokens:
                continue
            overlap = len(wanted & tokens)
            if not overlap:
                continue
            score = overlap / len(wanted | tokens)
            if score > best_score or (score == best_score and best_key is not None and key < best_key):
                best_key, best_score = key, score
        if best_score >= 0.34:
            return best_key, best_score
        return None, 0.0

    # -- pool construction -------------------------------------------------

    def lexical_scores(self, tokens: list[str], limit: int) -> dict[int, float]:
        """IDF-weighted accumulation over the inverted index."""
        scores: dict[int, float] = {}
        for token in tokens:
            for variant in token_variants(token):
                idf = self.catalog.token_idf(variant)
                if idf <= 0.0:
                    continue
                for doc in self.postings_set(variant):
                    scores[doc] = scores.get(doc, 0.0) + idf
                break  # the surface form is enough; avoid double counting
        if len(scores) <= limit:
            return scores
        top = sorted(scores.items(), key=lambda item: (-item[1], self.catalog.ids[item[0]]))[:limit]
        return dict(top)

    def prepare_semantic(self, max_df_ratio: float = 0.25) -> None:
        """Build the optional index before the first customer turn."""
        if self._semantic_index is None:
            self._semantic_index = SemanticIndex(self.catalog, max_df_ratio=max_df_ratio)

    def semantic_scores(
        self,
        query_text: str,
        limit: int,
        max_df_ratio: float = 0.25,
        minimum_score: float = 0.0,
    ) -> dict[int, float]:
        """Return cosine scores from the optional offline vector index."""
        self.prepare_semantic(max_df_ratio=max_df_ratio)
        scores = self._semantic_index.scores(query_text, limit)
        if minimum_score <= 0.0:
            return scores
        return {doc: score for doc, score in scores.items() if score >= minimum_score}

    def build_pool(
        self,
        bucket_key: str | None,
        query_tokens: list[str],
        bucket_limit: int,
        lexical_limit: int,
    ) -> tuple[list[int], set[int], dict[int, float]]:
        """Return (pool, bucket_members, lexical_scores)."""
        bucket_members: set[int] = set()
        if bucket_key:
            docs = self.catalog.bucket_candidates(bucket_key)
            if len(docs) > bucket_limit:
                # Keep the strongest priors when a bucket is unusually large.
                docs = sorted(docs, key=lambda d: (-self.catalog.priors[d], self.catalog.ids[d]))
                docs = docs[:bucket_limit]
            bucket_members = set(docs)

        lexical = self.lexical_scores(query_tokens, lexical_limit) if query_tokens else {}

        pool = set(bucket_members)
        pool.update(lexical)
        if not pool:
            # Degenerate input: fall back to the highest-prior products so the
            # contract is still satisfied with plausible recommendations.
            pool = set(
                sorted(range(self.catalog.size), key=lambda d: (-self.catalog.priors[d], self.catalog.ids[d]))[:200]
            )
        return sorted(pool), bucket_members, lexical
