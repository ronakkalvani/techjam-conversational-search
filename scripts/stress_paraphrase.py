"""Stress test: does the agent survive a paraphrasing customer?

The public evaluator speaks in fixed templates. The competition specification
warns that the organizer may add natural-language paraphrasing to the private
simulator ("If natural-language paraphrasing is added by the organizer, it
cannot decide correctness"). A parser tuned to exact template strings would
score well here and collapse there.

This script re-implements the *documented session protocol* with paraphrased
wording, leaving the scoring rules identical, and reports the degradation. It
never modifies the official evaluator; it is a development-only harness.

Usage:
    python3 scripts/stress_paraphrase.py --mode paraphrase
    python3 scripts/stress_paraphrase.py --mode template   # sanity control
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    ALLOWED_ATTRIBUTES, MAX_TURNS, TOP_K, catalog_index, classify_constraint,
    coarse_category, load_jsonl, materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent  # noqa: E402
from starter.catalog import Catalog  # noqa: E402
from starter.config import DEFAULT_CONFIG  # noqa: E402
from scripts.run_eval import build_config  # noqa: E402

# Paraphrases keep the same information content and the same structured
# semantics; only the surface wording changes. Selection is deterministic
# (indexed by sample) so runs are reproducible.
OPENING_BUYING = [
    "Hi! I need {category}. It's important that it has: {constraint}.",
    "Hoping to find {category} — must-have is {constraint}.",
    "Shopping for {category}. The one thing I can't compromise on: {constraint}.",
]
OPENING_BROWSING = [
    "I'm after {category}, though I haven't decided on specifics yet.",
    "Browsing {category} at the moment, nothing firm in mind.",
    "Show me {category} please — just seeing what's out there.",
]
OPENING_OVERRIDE = [
    "I want {category}. {old_value}",
    "Looking at {category} — {old_value}",
]
DISCLOSURE = [
    "What I really care about: {payload}.",
    "Key things for me are {payload}.",
    "Well, {payload} — those matter most.",
]
NO_ADDITIONAL = [
    "Nothing further on {attribute}, really.",
    "No strong feelings about {attribute} beyond what I've said.",
    "I don't have an additional preference for {attribute}.",
]
NO_PREFERENCE = [
    "No preference on {attribute} — your call.",
    "Honestly I don't have a preference for {attribute}; use your judgment.",
]
OVERRIDE = [
    "Hmm, actually forget what I said earlier. What I need is: {new_value}.",
    "Change of plan — ignore my earlier preference. What I want is: {new_value}.",
]
PROMPT = [
    "Not quite what I meant. Ask me about one specific attribute.",
    "These aren't right yet — ask me something specific.",
]


def pick(options: list[str], seed: int) -> str:
    return options[seed % len(options)]


def build_messages(mode: str):
    """Return (opening_fn, reply_fn) for the requested wording mode."""
    if mode == "template":
        def opening(sample, category, disclosed, seed):
            scenario = sample["scenario_type"]
            if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
                constraint = str(sample["intent_card"]["hard_constraints"][0])
                disclosed.add(constraint)
                return f"I'm looking for {category}. A key requirement is: {constraint}."
            if scenario == "intent_override":
                return f"I'm looking for {category}. {sample['behavior']['override']['old_value']}"
            return f"I'm looking for {category}, but I'm still exploring."

        def reply(kind, seed, **kwargs):
            if kind == "prompt":
                return "Those options are not quite right yet. Ask me about one specific attribute."
            if kind == "boundary":
                return f"I don't have a preference for {kwargs['attribute']}; please use your judgment."
            if kind == "exhausted":
                return f"I don't have an additional preference for {kwargs['attribute']}."
            if kind == "override":
                return f"Actually, ignore my earlier preference. What I need is: {kwargs['new_value']}."
            return "For that, what matters is: " + "; ".join(kwargs["matches"]) + "."
        return opening, reply

    def opening(sample, category, disclosed, seed):
        scenario = sample["scenario_type"]
        if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
            constraint = str(sample["intent_card"]["hard_constraints"][0])
            disclosed.add(constraint)
            return pick(OPENING_BUYING, seed).format(category=category, constraint=constraint)
        if scenario == "intent_override":
            return pick(OPENING_OVERRIDE, seed).format(
                category=category, old_value=sample["behavior"]["override"]["old_value"]
            )
        return pick(OPENING_BROWSING, seed).format(category=category)

    def reply(kind, seed, **kwargs):
        if kind == "prompt":
            return pick(PROMPT, seed)
        if kind == "boundary":
            return pick(NO_PREFERENCE, seed).format(attribute=kwargs["attribute"])
        if kind == "exhausted":
            return pick(NO_ADDITIONAL, seed).format(attribute=kwargs["attribute"])
        if kind == "override":
            return pick(OVERRIDE, seed).format(new_value=kwargs["new_value"])
        return pick(DISCLOSURE, seed).format(payload="; ".join(kwargs["matches"]))

    return opening, reply


def run(agent, samples, catalog_ids, categories, products, mode: str) -> dict:
    """Mirror of the official session protocol with configurable wording."""
    opening_fn, reply_fn = build_messages(mode)
    sessions = []

    for index, sample in enumerate(samples):
        session_id = f"stress_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}

        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = opening_fn(
            effective, coarse_category(categories.get(target, [])), disclosed, index
        )

        hit_turn = None
        best_rank = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break

            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = reply_fn("override", index + turn, new_value=new_value)
                continue

            attribute = response.get("ask_attribute")
            attribute = attribute if isinstance(attribute, str) else None
            if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
                user_message = reply_fn("boundary", index + turn, attribute=attribute)
                boundary_used = True
                continue
            if not attribute:
                user_message = reply_fn("prompt", index + turn)
                continue
            if attribute not in ALLOWED_ATTRIBUTES:
                attribute = "other"
            pool = [
                *[str(v) for v in card.get("hard_constraints", [])],
                *[str(v) for v in card.get("soft_preferences", [])],
            ]
            matches = [
                value for value in pool
                if value not in disclosed
                and (attribute == "other" or classify_constraint(value) == attribute)
            ][:2]
            if not matches:
                user_message = reply_fn("exhausted", index + turn, attribute=attribute)
            else:
                disclosed.update(matches)
                user_message = reply_fn("disclosure", index + turn, matches=matches)

        sessions.append({
            "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })

    def summarise(items):
        if not items:
            return {}
        hit = sum(int(i["hit"]) for i in items) / len(items)
        mrr = statistics.fmean(i["reciprocal_rank"] for i in items)
        mttc = statistics.fmean(
            i["first_hit_turn"] if i["first_hit_turn"] is not None else MAX_TURNS + 1
            for i in items
        )
        efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
        return {
            "n": len(items),
            "hit_rate_at_10": round(hit, 4),
            "mrr": round(mrr, 4),
            "mttc": round(mttc, 3),
            "efficiency": round(efficiency, 4),
            "technical_score": round(0.5 * hit + 0.3 * mrr + 0.2 * efficiency, 5),
        }

    grouped = defaultdict(list)
    for item in sessions:
        grouped[item["scenario_type"]].append(item)
    return {
        "overall": summarise(sessions),
        "by_scenario": {name: summarise(items) for name, items in sorted(grouped.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--mode", default="paraphrase", choices=("paraphrase", "template"))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    shared = Catalog(args.catalog)
    agent = Agent(args.catalog, config=build_config(args.overrides), catalog=shared)

    result = run(agent, samples, catalog_ids, categories, products, args.mode)
    result["mode"] = args.mode
    result["overrides"] = args.overrides
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
