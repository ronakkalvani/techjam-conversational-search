"""Evaluate EntropyShop on deterministic public-session splits.

Aggregate output is the default. Per-session output is deliberately unavailable
for the internal ``test`` split so it remains useful as a future-change guard.

Usage:
    python scripts/run_split_eval.py --split development
    python scripts/run_split_eval.py --split validation
    python scripts/run_split_eval.py --split test
    python scripts/run_split_eval.py --split folds
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from scripts.make_splits import (  # noqa: E402
    fold_names_for,
    sample_ids_for,
    select_samples,
    validate_manifest,
)
from scripts.run_eval import TimedAgent, build_config, summarise  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.catalog import Catalog  # noqa: E402


def requested_selections(manifest: dict, selection: str) -> list[str]:
    if selection == "folds":
        return list(manifest["folds"])
    sample_ids_for(manifest, selection)  # validate before expensive catalog init
    return [selection]


def includes_internal_test(manifest: dict, selections: list[str]) -> bool:
    """Return whether any requested selection contains the reserved test fold."""
    selected_folds = {
        fold_name for selection in selections for fold_name in fold_names_for(manifest, selection)
    }
    return bool(selected_folds & set(manifest["splits"]["test"]))


def fold_summary(rows: list[dict]) -> dict | None:
    """Summarise variability when all folds are evaluated independently."""
    if len(rows) < 2:
        return None
    scores = [float(row["technical_score"]) for row in rows]
    return {
        "fold_count": len(rows),
        "technical_score_mean": round(statistics.fmean(scores), 6),
        "technical_score_min": round(min(scores), 6),
        "technical_score_max": round(max(scores), 6),
        "technical_score_stdev": round(statistics.stdev(scores), 6),
    }


def write_payload(path: str | Path, payload: dict) -> Path:
    """Write a result file, creating an explicitly requested parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario-stratified split evaluations")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument(
        "--manifest", default=str(ROOT / "data" / "splits" / "public_v1.json")
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--include-sessions",
        action="store_true",
        help="include per-session records for development/validation/folds",
    )
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    all_samples = load_jsonl(args.dataset)
    validate_manifest(manifest, all_samples)
    selections = requested_selections(manifest, args.split)
    if args.include_sessions and includes_internal_test(manifest, selections):
        raise SystemExit("per-session output is disabled for the internal test split")

    catalog_ids, categories, products = catalog_index(args.catalog)
    started = time.perf_counter()
    shared_catalog = Catalog(args.catalog)
    init_seconds = time.perf_counter() - started
    config = build_config(args.overrides)

    rows: list[dict] = []
    session_records: dict[str, list[dict]] = {}
    for selection in selections:
        sample_ids = sample_ids_for(manifest, selection)
        samples = select_samples(all_samples, sample_ids)
        timed = TimedAgent(Agent(args.catalog, config=config, catalog=shared_catalog))
        result = evaluate(timed, samples, catalog_ids, categories, products)
        row = summarise(result, selection, init_seconds, timed.latencies)
        row["sample_count"] = len(samples)
        row["scenario_counts"] = {
            scenario: sum(sample["scenario_type"] == scenario for sample in samples)
            for scenario in ("buying", "browsing", "intent_override", "boundary")
        }
        rows.append(row)
        if args.include_sessions:
            session_records[selection] = result["sessions"]
        print(json.dumps(row, indent=2), flush=True)

    payload = {
        "manifest": str(args.manifest),
        "selection": args.split,
        "overrides": args.overrides,
        "rows": rows,
        "fold_summary": fold_summary(rows),
    }
    if session_records:
        payload["sessions"] = session_records
    if args.output:
        output = write_payload(args.output, payload)
        print(f"wrote {output}")
    if payload["fold_summary"]:
        print(json.dumps({"fold_summary": payload["fold_summary"]}, indent=2))


if __name__ == "__main__":
    main()
