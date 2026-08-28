"""Run the official evaluator against a configured EntropyShop agent.

The official evaluator is imported unmodified. Only this script (never the
agent runtime) is allowed to touch evaluator internals.

Usage:
    python3 scripts/run_eval.py --label default
    python3 scripts/run_eval.py --set policy.question_policy=entropy --label entropy
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.config import DEFAULT_CONFIG, AgentConfig  # noqa: E402


def _coerce(raw: str):
    lowered = raw.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    if raw.strip().startswith("(") and raw.strip().endswith(")"):
        inner = raw.strip()[1:-1]
        return tuple(int(part) for part in inner.split(",") if part.strip())
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def build_config(overrides: list[str]) -> AgentConfig:
    """Apply ``section.field=value`` overrides to the default configuration."""
    config = DEFAULT_CONFIG
    for override in overrides:
        if "=" not in override:
            raise SystemExit(f"bad --set value: {override!r}")
        path, raw = override.split("=", 1)
        section, _, field = path.partition(".")
        if not field:
            raise SystemExit(f"--set needs section.field, got {path!r}")
        value = _coerce(raw)
        current = getattr(config, section, None)
        if current is None:
            raise SystemExit(f"unknown config section {section!r}")
        if not hasattr(current, field):
            raise SystemExit(f"unknown field {section}.{field}")
        config = replace(config, **{section: replace(current, **{field: value})})
    return config


def summarise(result: dict, label: str, init_seconds: float, latencies: list[float]) -> dict:
    scenario = result.get("scenario_metrics", {})
    row = {
        "label": label,
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "technical_score": result["recommended_technical_score"],
        "total_tokens": result["reported_token_usage"]["total_tokens"],
        "init_seconds": round(init_seconds, 2),
    }
    if latencies:
        ordered = sorted(latencies)
        row["latency_ms_mean"] = round(statistics.fmean(ordered) * 1000, 2)
        row["latency_ms_p95"] = round(ordered[int(0.95 * (len(ordered) - 1))] * 1000, 2)
    for name in ("buying", "browsing", "intent_override", "boundary"):
        metrics = scenario.get(name)
        if metrics:
            row[f"{name}_hit"] = metrics["hit_rate_at_10"]
            row[f"{name}_mrr"] = metrics["mrr"]
            row[f"{name}_mttc"] = metrics["mttc"]
    return row


class TimedAgent:
    """Wrapper that records per-turn latency without altering behaviour."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self.latencies: list[float] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        started = time.perf_counter()
        try:
            return self._agent.respond(session_id, user_message, turn, top_k)
        finally:
            self.latencies.append(time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="EntropyShop experiment runner")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--label", default="default")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = build_config(args.overrides)
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)

    started = time.perf_counter()
    agent = Agent(args.catalog, config=config)
    init_seconds = time.perf_counter() - started

    timed = TimedAgent(agent)
    result = evaluate(timed, samples, catalog_ids, categories, products)
    row = summarise(result, args.label, init_seconds, timed.latencies)

    if args.output:
        payload = {"label": args.label, "overrides": args.overrides, "summary": row, **result}
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(row, indent=2))
    else:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
