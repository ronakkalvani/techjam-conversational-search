"""Run the required ablation grid and write a comparison table.

Each row is a full 200-session public-set evaluation using the unmodified
official evaluator.

Usage:
    python3 scripts/compare_policies.py --output docs/experiments.json
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

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from scripts.run_eval import TimedAgent, build_config, summarise  # noqa: E402
from starter.agent import Agent  # noqa: E402

# (label, [config overrides])
EXPERIMENTS: list[tuple[str, list[str]]] = [
    # Retrieval/ranking build-up.
    ("lexical_only", [
        "ranking.w_constraint=0.0", "ranking.w_phrase_bonus=0.0", "ranking.w_facet=0.0",
        "ranking.w_conflict=0.0", "ranking.w_category=0.0", "ranking.w_bucket=0.0",
        "ranking.w_lexical=1.0", "policy.enable_exploration=false",
        "policy.question_policy=entropy",
    ]),
    ("retrieval_plus_category", [
        "ranking.w_constraint=0.0", "ranking.w_phrase_bonus=0.0", "ranking.w_facet=0.0",
        "ranking.w_conflict=0.0", "ranking.w_lexical=1.0",
        "policy.enable_exploration=false", "policy.question_policy=entropy",
    ]),
    ("retrieval_plus_facets", [
        "policy.enable_exploration=false", "policy.question_policy=entropy",
        "ranking.w_phrase_bonus=0.0",
    ]),
    ("full_ranker_no_exploration", [
        "policy.enable_exploration=false", "policy.question_policy=entropy",
    ]),

    # Question policies (exploration on).
    ("fixed_question_order", ["policy.question_policy=fixed"]),
    ("policy_entropy", ["policy.question_policy=entropy"]),
    ("policy_other_first", ["policy.question_policy=other_first"]),
    ("policy_hybrid", ["policy.question_policy=hybrid"]),

    # Prior ablations on the default policy.
    ("hybrid_no_profile_prior", ["policy.use_profile_prior=false"]),
    ("hybrid_no_popularity_prior", ["policy.use_popularity_prior=false"]),

    # Exploration (previously-shown demotion).
    ("hybrid_no_exploration", ["policy.enable_exploration=false"]),

    # Recommendation-budget trade-off.
    ("budget_turn1_5", ["policy.turn_budget=(5,10)"]),
    ("budget_turn1_3", ["policy.turn_budget=(3,10)"]),
    ("budget_turn1_2", ["policy.turn_budget=(2,10)"]),
    ("budget_turn1_1", ["policy.turn_budget=(1,10)"]),
    ("budget_3_3_then_10", ["policy.turn_budget=(3,3,10)"]),
    ("budget_2_3_5_10", ["policy.turn_budget=(2,3,5,10)"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "docs" / "experiments.json"))
    parser.add_argument("--only", default=None, help="comma-separated label filter")
    args = parser.parse_args()

    keep = set(args.only.split(",")) if args.only else None
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)

    # Build the index once and share it: it is immutable, so this only removes
    # ~13s of redundant work per row.
    from starter.catalog import Catalog

    shared_started = time.perf_counter()
    shared_catalog = Catalog(args.catalog)
    shared_init = time.perf_counter() - shared_started
    print(f"catalog built in {shared_init:.1f}s, shared across experiments\n", flush=True)

    rows = []
    for label, overrides in EXPERIMENTS:
        if keep and label not in keep:
            continue
        config = build_config(overrides)
        started = time.perf_counter()
        agent = Agent(args.catalog, config=config, catalog=shared_catalog)
        init_seconds = time.perf_counter() - started + shared_init
        timed = TimedAgent(agent)
        result = evaluate(timed, samples, catalog_ids, categories, products)
        row = summarise(result, label, init_seconds, timed.latencies)
        row["overrides"] = overrides
        rows.append(row)
        print(
            f"{label:32s} hit={row['hit_rate_at_10']:.3f} mrr={row['mrr']:.3f} "
            f"mttc={row['mttc']:.2f} eff={row['efficiency']:.3f} score={row['technical_score']:.5f}",
            flush=True,
        )

    rows.sort(key=lambda item: -item["technical_score"])
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    print(f"best: {rows[0]['label']} score={rows[0]['technical_score']:.5f}")


if __name__ == "__main__":
    main()
