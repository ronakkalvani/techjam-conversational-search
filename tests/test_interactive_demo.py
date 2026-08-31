"""Tests for the keyboard-driven terminal presentation helpers."""

from __future__ import annotations

from scripts.interactive_demo import _product_facts, matching_evidence
from starter.explanations import compose_message


def test_matching_evidence_prefers_live_query_overlap():
    product = {
        "features": ["Generic gift packaging", "Raw green crystal with adjustable cord"],
        "details": {"Material": "Natural stone"},
    }
    evidence = matching_evidence(
        product,
        ["I'm looking for a pendant. I want a green crystal on an adjustable cord."],
    )
    assert evidence == ["Raw green crystal with adjustable cord"]


def test_matching_evidence_falls_back_to_visible_metadata():
    product = {"features": ["First feature", "Second feature"], "details": {}}
    assert matching_evidence(product, ["unrelated wording"]) == ["First feature"]


def test_product_facts_tolerates_sparse_and_malformed_values():
    assert _product_facts({"price": "19.5", "average_rating": "4.4", "rating_number": "12"}) \
        == "$19.50 | 4.4/5 (12 ratings)"
    assert _product_facts({"price": "unknown", "average_rating": None}) == ""


def test_single_recommendation_message_is_grammatical():
    assert compose_message("color", 1, 1).startswith("Here is 1 option matching")
