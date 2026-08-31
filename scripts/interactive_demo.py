"""Run EntropyShop as a genuinely live terminal conversation.

Unlike ``demo_session.py``, this command does not select a public evaluation
sample, simulate a customer, or know a hidden target. Every customer message
comes from stdin and is passed directly to the same submission ``Agent``.

Example:
    python scripts/interactive_demo.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.agent import Agent  # noqa: E402
from starter.text import content_tokens, flatten  # noqa: E402


COMMANDS = {
    ":help": "show the live-demo instructions",
    ":examples": "show fresh query ideas (they are not evaluator sessions)",
    ":new": "start a clean shopping session",
    ":quit": "exit the demo",
}

EXAMPLE_OPENINGS = (
    "I'm looking for wrist watches, but I'm still exploring.",
    "I'm looking for ankle boots, but I'm still exploring.",
    "I'm looking for pendant necklaces, but I'm still exploring.",
)


def _shorten(value: object, width: int = 100) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def _metadata_fragments(product: dict) -> list[str]:
    """Return readable, catalog-grounded fragments for a product."""
    fragments: list[str] = []
    features = product.get("features")
    if isinstance(features, list):
        fragments.extend(str(value) for value in features if value not in (None, ""))
    elif features not in (None, ""):
        fragments.append(str(features))

    details = product.get("details")
    if isinstance(details, dict):
        fragments.extend(
            f"{key}: {flatten(value)}"
            for key, value in details.items()
            if value not in (None, "", [])
        )
    return [" ".join(fragment.split()) for fragment in fragments if fragment.strip()]


def matching_evidence(product: dict, messages: list[str], limit: int = 1) -> list[str]:
    """Pick visible metadata snippets that overlap the live conversation."""
    query_tokens = set(content_tokens(" ".join(messages)))
    ranked: list[tuple[int, int, str]] = []
    for index, fragment in enumerate(_metadata_fragments(product)):
        overlap = len(query_tokens & set(content_tokens(fragment)))
        ranked.append((overlap, -index, fragment))
    ranked.sort(reverse=True)
    positive = [fragment for overlap, _index, fragment in ranked if overlap > 0]
    fallback = [fragment for _overlap, _index, fragment in ranked]
    return (positive or fallback)[: max(0, limit)]


def _product_facts(product: dict) -> str:
    facts: list[str] = []
    price = product.get("price")
    try:
        if price not in (None, ""):
            facts.append(f"${float(price):,.2f}")
    except (TypeError, ValueError):
        pass

    try:
        rating = float(product.get("average_rating") or 0)
        count = int(float(product.get("rating_number") or 0))
        if rating > 0:
            rating_text = f"{rating:.1f}/5"
            if count > 0:
                rating_text += f" ({count:,} ratings)"
            facts.append(rating_text)
    except (TypeError, ValueError):
        pass
    return " | ".join(facts)


def state_summary(agent: Agent, session_id: str) -> str:
    """Expose the small observable belief state for presentation."""
    state = agent._sessions.get(session_id)
    if state is None:
        return "category: unresolved | requirements: none yet"
    category = state.category or "unresolved"
    requirements = [constraint.text for constraint in state.constraints.active]
    rendered = "; ".join(requirements) if requirements else "none yet"
    return f"category: {_shorten(category, 45)} | requirements: {_shorten(rendered, 90)}"


def print_response(
    agent: Agent,
    session_id: str,
    response: dict,
    messages: list[str],
    latency_ms: float,
    display_limit: int,
) -> None:
    """Render one response with titles and supporting catalog metadata."""
    print(f"\nAGENT       > {response.get('message', '')}")
    print(f"INTERPRETED > {state_summary(agent, session_id)}")
    print(f"ASK FIELD   > {response.get('ask_attribute') or 'none'}")
    print("RECOMMENDATIONS")

    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        print("  (none)")
    else:
        for rank, item in enumerate(recommendations[:display_limit], 1):
            parent_asin = str(item.get("parent_asin", "")) if isinstance(item, dict) else str(item)
            doc = agent.catalog.index_of.get(parent_asin)
            product = agent.catalog.product(doc) if doc is not None else {}
            title = product.get("title") or "Unknown product"
            print(f"  {rank:>2}. {_shorten(title)}")
            facts = _product_facts(product)
            suffix = f" | {facts}" if facts else ""
            print(f"      ASIN {parent_asin}{suffix}")
            evidence = matching_evidence(product, messages)
            if evidence:
                print(f"      MATCH: {_shorten(evidence[0], 106)}")

    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    tokens = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
    print(f"RUNTIME     > {latency_ms:.2f} ms | offline | deterministic | {tokens} model tokens")


def print_help() -> None:
    print("\nHOW TO USE LIVE MODE")
    print("  Start with the product type, end that sentence, then add requirements.")
    print("  Example: I'm looking for ankle boots. I need waterproof leather boots for winter.")
    print("  On later turns, answer the exact question the agent asks.")
    print("  You can say: I don't have a preference for color; please use your judgment.")
    print("  For a change of mind: Actually, ignore my earlier preference. What I need is: suede.")
    print("  Continue only when the shown options are not right; continuing marks them as rejected.")
    print("\nCOMMANDS")
    for command, description in COMMANDS.items():
        print(f"  {command:<10} {description}")


def print_examples() -> None:
    print("\nFRESH QUERY IDEAS")
    for example in EXAMPLE_OPENINGS:
        print(f"  - {example}")
    print("  Answer the agent's next question in your own words.")
    print("  These are query ideas only; no customer replies or hidden answers are loaded.")


def run_interactive(agent: Agent, *, top_k: int, display_limit: int, max_turns: int) -> None:
    """Read customer turns until EOF or ``:quit``."""
    session_number = 1
    session_id = f"live_{session_number}"
    turn = 1
    messages: list[str] = []
    agent.reset(session_id, {})

    rule = "=" * 82
    print(rule)
    print("ENTROPYSHOP — LIVE INTERACTIVE DEMO")
    print(rule)
    print(f"Catalog: {agent.catalog.size:,} products | Input source: your keyboard")
    print("No evaluation sample, simulated customer, hidden target, network, or LLM is used.")
    print("Type :help for guidance, :examples for ideas, :new to reset, or :quit to exit.\n")

    while True:
        if turn > max_turns:
            print(f"\nReached the {max_turns}-turn evaluator limit. Type :new or :quit.")
        prompt = f"YOU [session {session_number}, turn {min(turn, max_turns)}] > "
        try:
            user_message = input(prompt).strip()
        except EOFError:
            print("\nEnd of input. Live demo closed.")
            return
        except KeyboardInterrupt:
            print("\nLive demo closed.")
            return

        if not user_message:
            continue
        command = user_message.lower()
        if command == ":quit":
            print("Live demo closed.")
            return
        if command == ":help":
            print_help()
            continue
        if command == ":examples":
            print_examples()
            continue
        if command == ":new":
            session_number += 1
            session_id = f"live_{session_number}"
            turn = 1
            messages = []
            agent.reset(session_id, {})
            print(f"\nStarted clean session {session_number}; previous constraints were discarded.\n")
            continue
        if command.startswith(":"):
            print("Unknown command. Type :help to see the available commands.")
            continue
        if turn > max_turns:
            print("Start a clean session with :new before sending another shopping message.")
            continue

        messages.append(user_message)
        started = time.perf_counter()
        response = agent.respond(session_id, user_message, turn, top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        print_response(agent, session_id, response, messages, latency_ms, display_limit)
        turn += 1
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a keyboard-driven EntropyShop session")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--top-k", type=int, default=10, help="maximum recommendations requested")
    parser.add_argument("--display-limit", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.is_file():
        parser.error(f"catalog was not found: {catalog_path}")
    print("Loading the catalog and building the offline indexes...", flush=True)
    started = time.perf_counter()
    agent = Agent(catalog_path)
    print(f"Ready in {time.perf_counter() - started:.2f} seconds.\n", flush=True)
    run_interactive(
        agent,
        top_k=max(1, args.top_k),
        display_limit=max(1, args.display_limit),
        max_turns=max(1, args.max_turns),
    )


if __name__ == "__main__":
    main()
