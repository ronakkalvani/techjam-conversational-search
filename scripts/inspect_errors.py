"""Diagnose where sessions lose rank or miss entirely.

Groups each session into one failure category so that the next iteration is
driven by evidence rather than guesswork.

This script is *development-only*. It reads public labels, which the agent
runtime never does.

Usage:
    python3 scripts/inspect_errors.py --limit 200
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, coarse_category, load_jsonl, materialize_hidden_fields,
)
from starter.agent import Agent  # noqa: E402
from starter.catalog import Catalog  # noqa: E402
from starter.config import DEFAULT_CONFIG  # noqa: E402
from starter.text import content_tokens, normalize  # noqa: E402

CATEGORIES = (
    "hit_rank_1",
    "hit_rank_2_3",
    "hit_rank_4_10",
    "miss_target_absent_from_pool",
    "miss_category_mismatch",
    "miss_metadata_missing",
    "miss_ranked_below_cutoff",
    "miss_other",
)


def classify_session(agent, catalog, sample, session, products) -> tuple[str, dict]:
    """Assign one diagnostic category, with supporting detail."""
    target = str(sample["ground_truth"]["parent_asin"])
    detail: dict = {"sample_id": sample["sample_id"], "scenario": sample["scenario_type"]}

    if session["hit"]:
        rank = session["best_rank"]
        detail["rank"] = rank
        detail["turn"] = session["first_hit_turn"]
        if rank == 1:
            return "hit_rank_1", detail
        if rank <= 3:
            return "hit_rank_2_3", detail
        return "hit_rank_4_10", detail

    doc = catalog.index_of.get(target)
    if doc is None:
        return "miss_other", detail

    # Was the target even reachable? Replay the opening turn's pool.
    spoken_category = coarse_category([str(v) for v in (products[target].get("categories") or [])])
    resolved, confidence = agent.retriever.resolve_bucket(spoken_category)
    detail["spoken_category"] = spoken_category
    detail["resolved_bucket"] = resolved
    detail["bucket_confidence"] = round(confidence, 3)

    if resolved is None:
        return "miss_category_mismatch", detail
    if doc not in set(catalog.bucket_candidates(resolved)):
        return "miss_category_mismatch", detail

    card, _behavior = materialize_hidden_fields(sample, products)
    constraints = list(card["hard_constraints"]) + list(card["soft_preferences"])
    text = catalog.text[doc]
    matched = sum(1 for value in constraints if normalize(value) in text)
    detail["constraints"] = len(constraints)
    detail["constraints_matching_target_text"] = matched
    if matched == 0:
        return "miss_metadata_missing", detail

    detail["bucket_size"] = len(catalog.bucket_candidates(resolved))
    return "miss_ranked_below_cutoff", detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--results", default=None, help="reuse a results.json instead of re-running")
    parser.add_argument("--output", default=str(ROOT / "docs" / "error_analysis.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)

    shared = Catalog(args.catalog)
    agent = Agent(args.catalog, config=DEFAULT_CONFIG, catalog=shared)

    if args.results:
        sessions = json.loads(Path(args.results).read_text())["sessions"]
    else:
        from evaluator.local_evaluator import evaluate

        sessions = evaluate(agent, samples, catalog_ids, categories, products)["sessions"]

    by_id = {item["sample_id"]: item for item in sessions}
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for sample in samples:
        session = by_id.get(sample["sample_id"])
        if session is None:
            continue
        name, detail = classify_session(agent, shared, sample, session, products)
        buckets[name].append(detail)

    total = sum(len(items) for items in buckets.values())
    print(f"{'category':34s} {'count':>6s} {'share':>7s}")
    print("-" * 50)
    for name in CATEGORIES:
        items = buckets.get(name, [])
        if not items:
            continue
        print(f"{name:34s} {len(items):6d} {len(items)/total:6.1%}")

    # Where is rank being lost among successful sessions?
    low = buckets.get("hit_rank_4_10", []) + buckets.get("hit_rank_2_3", [])
    if low:
        by_scenario = collections.Counter(item["scenario"] for item in low)
        by_turn = collections.Counter(item["turn"] for item in low)
        print("\nnon-rank-1 hits by scenario:", dict(by_scenario))
        print("non-rank-1 hits by turn    :", dict(sorted(by_turn.items())))

    Path(args.output).write_text(json.dumps(buckets, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
