"""Run one readable EntropyShop session through the official customer protocol.

The demo uses the same frozen catalog, public samples, customer replies, and
hit definition as ``evaluator.local_evaluator``. It only adds presentation:
turn-by-turn messages, titled recommendations, timing, and a final reveal.

Examples:
    python scripts/demo_session.py --sample-id public_0094
    python scripts/demo_session.py --scenario intent_override
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent  # noqa: E402


def select_sample(samples: list[dict], sample_id: str | None, scenario: str) -> dict:
    """Select an explicit sample or the first sample in a requested scenario."""
    if sample_id:
        for sample in samples:
            if sample.get("sample_id") == sample_id:
                return sample
        raise ValueError(f"sample {sample_id!r} was not found")

    for sample in samples:
        if sample.get("scenario_type") == scenario:
            return sample
    raise ValueError(f"no sample found for scenario {scenario!r}")


def run_session(
    agent: Agent,
    sample: dict,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    *,
    max_turns: int = MAX_TURNS,
    top_k: int = TOP_K,
) -> dict:
    """Run one session with the official simulator rules and retain a transcript."""
    sample_id = str(sample["sample_id"])
    scenario = str(sample["scenario_type"])
    target = str(sample["ground_truth"]["parent_asin"])
    intent_card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}

    session_id = f"demo_{sample_id}"
    agent.reset(session_id, sample.get("user_profile") or {})
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = scenario != "intent_override"
    user_message = initial_message(
        effective_sample,
        coarse_category(categories.get(target, [])),
        disclosed,
    )

    transcript: list[dict] = []
    hit_turn: int | None = None
    target_rank: int | None = None

    for turn in range(1, min(max_turns, MAX_TURNS) + 1):
        started = time.perf_counter()
        response = agent.respond(session_id, user_message, turn, top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)

        scored_rank = ranked.index(target) + 1 if override_applied and target in ranked else None
        transcript.append(
            {
                "turn": turn,
                "customer": user_message,
                "agent_message": response.get("message", ""),
                "ask_attribute": response.get("ask_attribute"),
                "recommendations": ranked,
                "latency_ms": round(latency_ms, 2),
                "target_rank": scored_rank,
            }
        )
        if scored_rank is not None:
            hit_turn, target_rank = turn, scored_rank
            break
        if turn == min(max_turns, MAX_TURNS):
            break

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            user_message, boundary_used = customer_reply(
                effective_sample,
                response.get("ask_attribute"),
                disclosed,
                boundary_used,
            )

    return {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "catalog_size": len(catalog_ids),
        "target": target,
        "target_title": str(products.get(target, {}).get("title") or "Unknown product"),
        "hit": hit_turn is not None,
        "hit_turn": hit_turn,
        "target_rank": target_rank,
        "transcript": transcript,
    }


def _shorten(value: str, width: int = 92) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= width else value[: width - 1].rstrip() + "…"


def print_transcript(result: dict, products: dict[str, dict], display_limit: int) -> None:
    """Render a video-friendly terminal transcript."""
    rule = "=" * 78
    print(rule)
    print("ENTROPYSHOP — OFFICIAL-PROTOCOL DEMO")
    print(rule)
    print(f"Sample:   {result['sample_id']}")
    print(f"Scenario: {str(result['scenario_type']).replace('_', ' ').title()}")
    print(f"Catalog:  {result['catalog_size']:,} real products")
    print("Target:   hidden until the session ends")

    for item in result["transcript"]:
        print(f"\n--- TURN {item['turn']} ---")
        print(f"CUSTOMER  > {item['customer']}")
        print(f"AGENT     > {item['agent_message']}")
        requested = item["ask_attribute"] or "none"
        print(f"ASK FIELD > {requested}")
        print("RECOMMENDATIONS")
        shown = item["recommendations"][:display_limit]
        if not shown:
            print("  (none)")
        for rank, parent_asin in enumerate(shown, 1):
            title = products.get(parent_asin, {}).get("title") or "Unknown product"
            print(f"  {rank:>2}. {_shorten(title)}")
            print(f"      {parent_asin}")
        print(f"LATENCY   > {item['latency_ms']:.2f} ms")

    print(f"\n{rule}")
    if result["hit"]:
        print(
            f"SUCCESS — target found on turn {result['hit_turn']} "
            f"at rank {result['target_rank']}"
        )
    else:
        print("MISS — target was not found within the configured turn limit")
    print(f"TARGET  > {result['target_title']}")
    print(f"ASIN    > {result['target']}")
    print(rule)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show one official EntropyShop session")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--sample-id", default=None)
    parser.add_argument(
        "--scenario",
        choices=("buying", "browsing", "intent_override", "boundary"),
        default="buying",
        help="used when --sample-id is omitted",
    )
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--display-limit", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    if not args.json:
        print("Loading the official catalog and initializing EntropyShop...", flush=True)
    started = time.perf_counter()
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    sample = select_sample(samples, args.sample_id, args.scenario)
    agent = Agent(args.catalog)
    initialization_seconds = time.perf_counter() - started
    if not args.json:
        print(f"Ready in {initialization_seconds:.2f} seconds.\n", flush=True)
    result = run_session(
        agent,
        sample,
        catalog_ids,
        categories,
        products,
        max_turns=max(1, args.max_turns),
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_transcript(result, products, max(1, args.display_limit))


if __name__ == "__main__":
    main()
