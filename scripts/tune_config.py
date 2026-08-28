"""Sweep interpretable global weights against the public set.

Deliberately coarse: the point is to choose *global, explainable* settings
(a recommendation budget, a prior on/off) from measured behaviour, never to
fit individual public targets. Every setting swept here is a single number
that generalises across sessions.

Usage:
    python3 scripts/tune_config.py --grid budget
    python3 scripts/tune_config.py --grid final
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
from starter.catalog import Catalog  # noqa: E402

GRIDS: dict[str, list[tuple[str, list[str]]]] = {
    "budget": [
        ("b_1_2_3_5_10", ["policy.turn_budget=(1,2,3,5,10)"]),
        ("b_1_2_3_10", ["policy.turn_budget=(1,2,3,10)"]),
        ("b_1_1_2_3_5_10", ["policy.turn_budget=(1,1,2,3,5,10)"]),
        ("b_1_3_10", ["policy.turn_budget=(1,3,10)"]),
    ],
    "final": [
        ("b_2_3_5_10_noprofile", [
            "policy.turn_budget=(2,3,5,10)", "policy.use_profile_prior=false",
        ]),
        ("b_1_2_3_5_10_noprofile", [
            "policy.turn_budget=(1,2,3,5,10)", "policy.use_profile_prior=false",
        ]),
        ("b_1_2_3_5_10_noprofile_otherfirst", [
            "policy.turn_budget=(1,2,3,5,10)", "policy.use_profile_prior=false",
            "policy.question_policy=other_first",
        ]),
        ("b_1_2_3_5_10_noprofile_temp010", [
            "policy.turn_budget=(1,2,3,5,10)", "policy.use_profile_prior=false",
            "questions.posterior_temperature=0.10",
        ]),
    ],
    "robust": [
        # Conservative fallbacks kept for the generalisation discussion.
        ("safe_full_budget", ["policy.use_profile_prior=false"]),
        ("safe_budget_3_5_10", [
            "policy.turn_budget=(3,5,10)", "policy.use_profile_prior=false",
        ]),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--grid", default="budget")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    grid = GRIDS.get(args.grid)
    if grid is None:
        raise SystemExit(f"unknown grid {args.grid!r}; choose from {sorted(GRIDS)}")

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    started = time.perf_counter()
    shared = Catalog(args.catalog)
    shared_init = time.perf_counter() - started

    rows = []
    for label, overrides in grid:
        config = build_config(overrides)
        agent = Agent(args.catalog, config=config, catalog=shared)
        timed = TimedAgent(agent)
        result = evaluate(timed, samples, catalog_ids, categories, products)
        row = summarise(result, label, shared_init, timed.latencies)
        row["overrides"] = overrides
        rows.append(row)
        print(
            f"{label:36s} hit={row['hit_rate_at_10']:.3f} mrr={row['mrr']:.3f} "
            f"mttc={row['mttc']:.2f} score={row['technical_score']:.5f}",
            flush=True,
        )

    rows.sort(key=lambda item: -item["technical_score"])
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nbest: {rows[0]['label']} score={rows[0]['technical_score']:.5f}")


if __name__ == "__main__":
    main()
