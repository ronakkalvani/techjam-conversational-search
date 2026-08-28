"""Tests for the video-friendly official-protocol demo runner."""

from __future__ import annotations

import pytest

from scripts.demo_session import run_session, select_sample


def test_select_sample_supports_explicit_id_and_scenario():
    samples = [
        {"sample_id": "one", "scenario_type": "buying"},
        {"sample_id": "two", "scenario_type": "boundary"},
    ]
    assert select_sample(samples, "two", "buying")["sample_id"] == "two"
    assert select_sample(samples, None, "boundary")["sample_id"] == "two"


def test_select_sample_reports_missing_selection():
    with pytest.raises(ValueError, match="was not found"):
        select_sample([], "missing", "buying")
    with pytest.raises(ValueError, match="no sample found"):
        select_sample([], None, "buying")


class _TwoTurnAgent:
    def __init__(self, target: str) -> None:
        self.target = target

    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        recommendations = [] if turn == 1 else [{"parent_asin": self.target}]
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def test_run_session_replays_customer_and_scores_target():
    target = "TARGET"
    sample = {
        "sample_id": "demo",
        "scenario_type": "buying",
        "ground_truth": {"parent_asin": target},
        "user_profile": {},
        "intent_card": {
            "target_category": "Blue Cotton Shirt",
            "hard_constraints": ["100% cotton"],
            "soft_preferences": ["machine washable"],
        },
        "behavior": {"scenario_type": "buying"},
    }
    products = {
        target: {
            "parent_asin": target,
            "title": "Blue Cotton Shirt",
            "categories": ["Clothing, Shoes & Jewelry", "Women", "Tops", "Shirts"],
        }
    }
    result = run_session(
        _TwoTurnAgent(target),
        sample,
        {target},
        {target: products[target]["categories"]},
        products,
    )

    assert result["hit"] is True
    assert result["hit_turn"] == 2
    assert result["target_rank"] == 1
    assert len(result["transcript"]) == 2
    assert result["transcript"][1]["recommendations"] == [target]
