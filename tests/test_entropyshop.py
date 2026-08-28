"""Unit tests for EntropyShop.

These run against small synthetic catalogs so they are fast and independent of
the 50,000-product frozen file. Integration tests that need the real catalog
skip themselves when it is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.agent import Agent  # noqa: E402
from starter.catalog import Catalog, coarse_category  # noqa: E402
from starter.config import ALLOWED_ATTRIBUTES, DEFAULT_CONFIG  # noqa: E402
from starter.constraints import ConstraintSet, parse_message  # noqa: E402
from starter.facets import classify_constraint, extract_facets  # noqa: E402
from starter.questions import QuestionSelector  # noqa: E402
from starter.retrieval import Retriever  # noqa: E402

REAL_CATALOG = ROOT / "data" / "catalog.jsonl"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _product(asin, title, features=None, details=None, categories=None,
             price=None, rating=4.0, count=10, store="ExampleStore",
             description=None):
    return {
        "parent_asin": asin,
        "title": title,
        "features": features or [],
        "description": description or [],
        "price": price,
        "categories": categories or ["Clothing, Shoes & Jewelry", "Women", "Tops"],
        "details": details or {},
        "average_rating": rating,
        "rating_number": count,
        "store": store,
    }


SYNTHETIC = [
    _product("A0000001", "Blue Cotton T-Shirt", ["100% Cotton", "Machine wash cold"],
             {"Department": "Womens", "Material": "Cotton"},
             ["Clothing, Shoes & Jewelry", "Women", "Tops", "T-Shirts"], price=19.99),
    _product("A0000002", "Red Leather Belt", ["100% Leather", "Buckle closure"],
             {"Department": "Mens", "Material": "Leather"},
             ["Clothing, Shoes & Jewelry", "Men", "Accessories", "Belts"], price=29.99),
    _product("A0000003", "Black Wool Scarf", ["Soft wool", "Lightweight"],
             {"Material": "Wool"},
             ["Clothing, Shoes & Jewelry", "Women", "Accessories", "Scarves"], price=24.50),
    _product("A0000004", "Green Nylon Running Jacket",
             ["Waterproof", "Lightweight", "Great for running"],
             {"Material": "Nylon"},
             ["Clothing, Shoes & Jewelry", "Women", "Tops", "T-Shirts"], price=59.00),
    # Deliberately broken metadata.
    _product("A0000005", "Mystery Item", None, None, [], price=None, rating=0, count=0, store=""),
]


@pytest.fixture(scope="module")
def synthetic_catalog(tmp_path_factory):
    path = tmp_path_factory.mktemp("catalog") / "catalog.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for product in SYNTHETIC:
            handle.write(json.dumps(product) + "\n")
    return Catalog(path)


@pytest.fixture(scope="module")
def synthetic_agent(tmp_path_factory):
    path = tmp_path_factory.mktemp("agent") / "catalog.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for product in SYNTHETIC:
            handle.write(json.dumps(product) + "\n")
    return Agent(path)


PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort"],
    "summary": "Prior purchases emphasize fit and comfort.",
}


def _assert_valid_response(response, top_k=10):
    """The official contract, asserted in full."""
    assert isinstance(response, dict)
    assert isinstance(response.get("message"), str)
    attribute = response.get("ask_attribute")
    assert attribute is None or attribute in ALLOWED_ATTRIBUTES
    recommendations = response.get("recommendations")
    assert isinstance(recommendations, list)
    assert len(recommendations) <= top_k
    seen = set()
    for item in recommendations:
        assert isinstance(item, dict)
        asin = item.get("parent_asin")
        assert isinstance(asin, str) and asin
        assert asin not in seen, "duplicate parent_asin"
        seen.add(asin)
        assert set(item).issubset({"parent_asin", "score"})
    usage = response.get("usage")
    if usage is not None:
        assert usage["prompt_tokens"] >= 0 and usage["completion_tokens"] >= 0
    assert set(response).issubset({"message", "ask_attribute", "recommendations", "usage"})


# --------------------------------------------------------------------------
# 1. Catalog
# --------------------------------------------------------------------------

def test_catalog_loads_and_indexes(synthetic_catalog):
    assert synthetic_catalog.size == len(SYNTHETIC)
    assert synthetic_catalog.index_of["A0000001"] == 0
    assert len(synthetic_catalog.bucket_docs) >= 1


def test_catalog_tolerates_missing_fields(synthetic_catalog):
    doc = synthetic_catalog.index_of["A0000005"]
    assert synthetic_catalog.price[doc] is None
    assert isinstance(synthetic_catalog.predicted_constraints[doc], tuple)
    assert synthetic_catalog.predicted_constraints[doc]  # never empty


def test_coarse_category_drops_root_only():
    assert coarse_category(["Clothing, Shoes & Jewelry", "Women", "Tops", "T-Shirts"]) == "Tops T-Shirts"
    assert coarse_category([]) == "clothing item"
    # "Shoes & Jewelry" is a real component once the root string is split.
    assert coarse_category(["Clothing, Shoes & Jewelry"]) == "Shoes & Jewelry"


def test_priors_are_bounded(synthetic_catalog):
    assert all(0.0 <= value <= 1.0 for value in synthetic_catalog.priors)


# --------------------------------------------------------------------------
# 2. Facets
# --------------------------------------------------------------------------

def test_facet_extraction_reads_details_first():
    facets = extract_facets(SYNTHETIC[1])
    assert "leather" in facets["material"]
    assert "examplestore" in facets["brand"]


def test_facet_extraction_handles_empty_product():
    facets = extract_facets(SYNTHETIC[4])
    assert all(isinstance(values, set) for values in facets.values())


def test_multivalued_facets_are_sets():
    facets = extract_facets(_product("X", "Cotton and Polyester Blend Shirt",
                                     ["95% cotton, 5% spandex"]))
    assert {"cotton", "polyester"} & facets["material"]
    assert len(facets["material"]) >= 2


@pytest.mark.parametrize("value,expected", [
    ("100% Leather", "material"),
    ("color: black", "color"),
    ("budget around $29.99", "budget"),
    ("Size: wide width", "size"),
    ("Department: Womens", "style"),
    ("Great for hiking", "use_case"),
    ("Buckle closure", "feature"),
    ("Triple Moon Pentagram Symbol", "feature"),
])
def test_constraint_classification(value, expected):
    assert classify_constraint(value) == expected


# --------------------------------------------------------------------------
# 3. Retrieval
# --------------------------------------------------------------------------

def test_bucket_resolution_exact_and_fuzzy(synthetic_catalog):
    retriever = Retriever(synthetic_catalog)
    key, confidence = retriever.resolve_bucket("Tops T-Shirts")
    assert key == "Tops T-Shirts" and confidence == 1.0
    # Paraphrase still resolves through token overlap.
    key, confidence = retriever.resolve_bucket("t-shirts")
    assert key == "Tops T-Shirts" and 0 < confidence < 1.0
    assert retriever.resolve_bucket("completely unrelated widget")[0] is None


def test_retrieval_is_stable(synthetic_catalog):
    retriever = Retriever(synthetic_catalog)
    first = retriever.build_pool("Tops T-Shirts", ["cotton"], 100, 100)
    second = retriever.build_pool("Tops T-Shirts", ["cotton"], 100, 100)
    assert first[0] == second[0]


def test_pool_never_empty(synthetic_catalog):
    retriever = Retriever(synthetic_catalog)
    pool, _members, _lex = retriever.build_pool(None, [], 100, 100)
    assert pool


# --------------------------------------------------------------------------
# 4. Constraints, overrides and boundaries
# --------------------------------------------------------------------------

def test_parse_opening_with_requirement():
    parsed = parse_message("I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.", 1)
    assert parsed.kind == "opening"
    assert parsed.category == "Jewelry Necklaces"
    assert parsed.new_constraints == ["Material:alloy"]


def test_parse_browsing_opening_has_no_constraint():
    parsed = parse_message("I'm looking for Shirts T-Shirts, but I'm still exploring.", 1)
    assert parsed.category == "Shirts T-Shirts"
    assert parsed.new_constraints == []


def test_parse_disclosure_splits_multiple():
    parsed = parse_message("For that, what matters is: leather; 100% Leather.", 2)
    assert parsed.new_constraints == ["leather", "100% Leather"]


def test_parse_exhausted_and_boundary():
    assert parse_message("I don't have an additional preference for material.", 3).exhausted_attribute == "material"
    assert parse_message("I don't have a preference for color; please use your judgment.", 3).unavailable_attribute == "color"


def test_parse_never_raises_on_junk():
    for junk in ("", "   ", "???", "\n\t", "a" * 5000, "I'm looking for"):
        assert parse_message(junk, 1) is not None


def test_soft_versus_hard_constraints():
    constraints = ConstraintSet()
    constraints.add("100% Leather", 1, hard=True)
    assert len(constraints) == 1
    assert constraints.items[0].hard is True
    constraints.add("100% Leather", 2)  # exact repeat ignored
    assert len(constraints) == 1


def test_override_deactivates_without_deleting_history():
    constraints = ConstraintSet()
    constraints.add("Buckle closure", 1, hard=True)
    retired = constraints.apply_override("Stainless Steel Band", 3)
    assert retired and retired[0].text == "Buckle closure"
    assert retired[0].active is False
    # History is preserved for explanation/debugging.
    assert "Buckle closure" in constraints.all_texts
    assert any(item.text == "Stainless Steel Band" and item.active for item in constraints.items)


def test_override_makes_excluded_candidates_eligible_again(synthetic_agent):
    """A product ruled out only by the old preference must come back."""
    agent, session = synthetic_agent, "override-session"
    agent.reset(session, PROFILE)
    agent.respond(session, "I'm looking for Accessories Belts. 100% Leather", 1, 10)
    agent.respond(session, "For that, what matters is: Buckle closure.", 2, 10)
    after = agent.respond(
        session, "Actually, ignore my earlier preference. What I need is: Soft wool.", 3, 10
    )
    _assert_valid_response(after, top_k=3)
    state = agent._sessions[session]
    assert state.override_count == 1
    # The wool scarf was excluded by the leather preference; it is now scored.
    scored = {agent.catalog.ids[doc] for doc, _ in agent._rank(state)}
    assert "A0000003" in scored


def test_boundary_attribute_not_reasked(synthetic_agent):
    agent, session = synthetic_agent, "boundary-session"
    agent.reset(session, PROFILE)
    agent.respond(session, "I'm looking for Tops T-Shirts, but I'm still exploring.", 1, 10)
    agent.respond(session, "I don't have a preference for color; please use your judgment.", 2, 10)
    state = agent._sessions[session]
    assert "color" in state.unavailable
    for turn in range(3, 8):
        response = agent.respond(session, "I don't have an additional preference for other.", turn, 10)
        assert response["ask_attribute"] != "color"
        _assert_valid_response(response)


def test_exhausted_attribute_not_repeated(synthetic_agent):
    agent, session = synthetic_agent, "exhausted-session"
    agent.reset(session, PROFILE)
    agent.respond(session, "I'm looking for Tops T-Shirts, but I'm still exploring.", 1, 10)
    asked = []
    for turn in range(2, 9):
        response = agent.respond(
            session, "I don't have an additional preference for other.", turn, 10
        )
        asked.append(response["ask_attribute"])
    # "other" is answered as exhausted, so it must not be asked again.
    assert asked.count("other") == 0


def test_agent_terminates_when_all_attributes_exhausted(synthetic_agent):
    agent, session = synthetic_agent, "terminate-session"
    agent.reset(session, PROFILE)
    agent.respond(session, "I'm looking for Tops T-Shirts, but I'm still exploring.", 1, 10)
    last = None
    for turn, attribute in enumerate(ALLOWED_ATTRIBUTES, start=2):
        last = agent.respond(
            session, f"I don't have an additional preference for {attribute}.", turn, 10
        )
        _assert_valid_response(last)
    assert last["ask_attribute"] is None
    assert last["recommendations"]  # still recommending


# --------------------------------------------------------------------------
# 5. Question value
# --------------------------------------------------------------------------

def test_entropy_prefers_the_splitting_question(synthetic_catalog):
    """On a catalog split cleanly by material, material must beat brand.

    All five products share one store, so a brand question cannot separate
    them however distinctive brands are in general.
    """
    selector = QuestionSelector(synthetic_catalog, DEFAULT_CONFIG.questions)
    scored = [(doc, 1.0) for doc in range(synthetic_catalog.size)]
    report = selector.evaluate(scored, set(), set(), set(), ("material", "brand"))
    assert report["material"]["utility"] > report["brand"]["utility"]
    assert report["brand"]["answerability"] == pytest.approx(0.0, abs=1e-6)


def test_information_gain_is_zero_when_nothing_left_to_disclose(synthetic_catalog):
    selector = QuestionSelector(synthetic_catalog, DEFAULT_CONFIG.questions)
    scored = [(doc, 1.0) for doc in range(synthetic_catalog.size)]
    disclosed = set()
    for doc in range(synthetic_catalog.size):
        disclosed.update(text for text, _kind in selector.typed_constraints(doc))
    report = selector.evaluate(scored, disclosed, set(), set(), ("material", "feature", "other"))
    for metrics in report.values():
        assert metrics["information_gain"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["answerability"] == pytest.approx(0.0, abs=1e-6)


def test_posterior_is_normalised(synthetic_catalog):
    selector = QuestionSelector(synthetic_catalog, DEFAULT_CONFIG.questions)
    weights = selector.posterior([(0, 1.5), (1, 1.2), (2, 0.4)])
    assert sum(weight for _doc, weight in weights) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 6. Agent contract and robustness
# --------------------------------------------------------------------------

def test_contract_valid_for_every_turn(synthetic_agent):
    agent, session = synthetic_agent, "contract-session"
    agent.reset(session, PROFILE)
    messages = [
        "I'm looking for Tops T-Shirts. A key requirement is: 100% Cotton.",
        "For that, what matters is: Machine wash cold.",
        "I don't have an additional preference for material.",
        "Actually, ignore my earlier preference. What I need is: Waterproof.",
        "Those options are not quite right yet. Ask me about one specific attribute.",
    ]
    for turn, message in enumerate(messages, start=1):
        _assert_valid_response(agent.respond(session, message, turn, 10))


def test_respond_survives_malformed_input(synthetic_agent):
    agent, session = synthetic_agent, "malformed-session"
    agent.reset(session, PROFILE)
    for bad in (None, "", 12345, {"unexpected": True}, "\x00\x01"):
        _assert_valid_response(agent.respond(session, bad, 1, 10))


def test_respond_without_reset_does_not_raise(synthetic_agent):
    _assert_valid_response(synthetic_agent.respond("never-reset", "I'm looking for Tops.", 1, 10))


def test_recommendations_capped_at_ten(synthetic_agent):
    agent, session = synthetic_agent, "cap-session"
    agent.reset(session, PROFILE)
    for turn in range(1, 11):
        response = agent.respond(session, "I'm looking for Tops T-Shirts.", turn, 10)
        assert len(response["recommendations"]) <= 10


def test_sessions_are_isolated(synthetic_agent):
    agent = synthetic_agent
    agent.reset("iso-a", PROFILE)
    agent.reset("iso-b", PROFILE)
    agent.respond("iso-a", "I'm looking for Accessories Belts. 100% Leather", 1, 10)
    agent.respond("iso-b", "I'm looking for Tops T-Shirts, but I'm still exploring.", 1, 10)
    state_a, state_b = agent._sessions["iso-a"], agent._sessions["iso-b"]
    assert len(state_a.constraints) == 1
    assert len(state_b.constraints) == 0
    assert state_a.bucket_key != state_b.bucket_key


def test_deterministic_across_identical_sessions(synthetic_agent):
    agent = synthetic_agent
    outputs = []
    for name in ("det-1", "det-2"):
        agent.reset(name, PROFILE)
        outputs.append([
            agent.respond(name, "I'm looking for Tops T-Shirts. A key requirement is: 100% Cotton.", 1, 10),
            agent.respond(name, "For that, what matters is: Machine wash cold.", 2, 10),
        ])
    assert outputs[0] == outputs[1]


def test_turn_budget_is_respected(synthetic_agent):
    agent, session = synthetic_agent, "budget-session"
    agent.reset(session, PROFILE)
    budget = DEFAULT_CONFIG.policy.turn_budget
    for turn in range(1, 6):
        response = agent.respond(session, "I'm looking for Tops T-Shirts.", turn, 10)
        expected = budget[min(turn - 1, len(budget) - 1)]
        assert len(response["recommendations"]) <= expected


def test_zero_token_usage(synthetic_agent):
    agent, session = synthetic_agent, "token-session"
    agent.reset(session, PROFILE)
    response = agent.respond(session, "I'm looking for Tops T-Shirts.", 1, 10)
    assert response["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}


# --------------------------------------------------------------------------
# 7. No label leakage
# --------------------------------------------------------------------------

def test_runtime_never_imports_evaluator_or_reads_labels():
    """Agent modules must not touch evaluator internals or the labelled set.

    Checked against the parsed AST rather than raw text, so that prose in a
    docstring discussing the evaluator is not mistaken for a dependency on it.
    """
    import ast

    forbidden_names = {"ground_truth", "intent_card", "scenario_type",
                       "sample_id", "behavior", "public_set", "difficulty_bucket"}
    for path in sorted((ROOT / "starter").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("evaluator"), path.name
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("evaluator"), path.name
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_names, f"{path.name}: .{node.attr}"
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden_names, f"{path.name}: {node.id}"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # String literals may not be used as label keys either; a
                # docstring is exempt because it is prose, not a lookup.
                if node.value in forbidden_names:
                    raise AssertionError(f"{path.name}: literal {node.value!r}")


def test_agent_module_graph_is_self_contained():
    import starter.agent as module

    for name in dir(module):
        value = getattr(module, name)
        origin = getattr(value, "__module__", "") or ""
        assert not origin.startswith("evaluator")


def test_no_hardcoded_catalog_identifiers():
    """No public target ASIN may be baked into the agent."""
    import re

    asin = re.compile(r"\bB0[A-Z0-9]{8}\b")
    for path in sorted((ROOT / "starter").glob("*.py")):
        assert not asin.search(path.read_text(encoding="utf-8")), path.name


# --------------------------------------------------------------------------
# 8. Integration against the real catalog (skipped when absent)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_CATALOG.exists(), reason="frozen catalog not downloaded")
def test_real_catalog_latency_and_shape():
    import time

    agent = Agent(REAL_CATALOG)
    agent.reset("real-1", PROFILE)
    started = time.perf_counter()
    response = agent.respond(
        "real-1", "I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.", 1, 10
    )
    elapsed = time.perf_counter() - started
    _assert_valid_response(response, top_k=10)
    assert elapsed < 2.0, f"first turn took {elapsed:.2f}s"
    assert all(item["parent_asin"] in agent.catalog.index_of for item in response["recommendations"])
