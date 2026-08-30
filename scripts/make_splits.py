"""Create deterministic scenario-stratified folds for the public sessions.

The official ``data/public_set.jsonl`` is never modified. This script writes a
small manifest containing sample IDs only, plus provenance and balance checks.

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import load_jsonl  # noqa: E402

DEFAULT_SEED = "entropyshop-public-v1"
DEFAULT_FOLDS = 5
DEFAULT_OUTPUT = ROOT / "data" / "splits" / "public_v1.json"


def _stable_key(seed: str, scenario: str, sample_id: str) -> str:
    payload = f"{seed}\0{scenario}\0{sample_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    samples: list[dict],
    *,
    seed: str = DEFAULT_SEED,
    fold_count: int = DEFAULT_FOLDS,
    source: str = "data/public_set.jsonl",
    source_sha256: str | None = None,
) -> dict:
    """Return a deterministic fold manifest stratified by scenario type."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")

    groups: dict[str, list[str]] = defaultdict(list)
    all_ids: list[str] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "").strip()
        scenario = str(sample.get("scenario_type") or "").strip()
        if not sample_id or not scenario:
            raise ValueError("every sample needs sample_id and scenario_type")
        all_ids.append(sample_id)
        groups[scenario].append(sample_id)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("sample IDs must be unique")

    fold_names = [f"fold_{index}" for index in range(1, fold_count + 1)]
    assigned: dict[str, list[str]] = {name: [] for name in fold_names}
    for scenario, sample_ids in sorted(groups.items()):
        ordered = sorted(sample_ids, key=lambda value: _stable_key(seed, scenario, value))
        for index, sample_id in enumerate(ordered):
            assigned[fold_names[index % fold_count]].append(sample_id)

    by_id = {str(sample["sample_id"]): sample for sample in samples}
    folds: dict[str, dict] = {}
    for fold_name in fold_names:
        sample_ids = sorted(assigned[fold_name])
        scenario_counts = Counter(str(by_id[value]["scenario_type"]) for value in sample_ids)
        folds[fold_name] = {
            "sample_ids": sample_ids,
            "scenario_counts": dict(sorted(scenario_counts.items())),
        }

    if fold_count == 5:
        named_splits = {
            "development": fold_names[:3],
            "validation": [fold_names[3]],
            "test": [fold_names[4]],
        }
    else:
        named_splits = {
            "development": fold_names[:-2],
            "validation": [fold_names[-2]],
            "test": [fold_names[-1]],
        }

    manifest = {
        "version": 1,
        "source": source,
        "source_sha256": source_sha256,
        "seed": seed,
        "fold_count": fold_count,
        "stratify_by": ["scenario_type"],
        "caveat": (
            "The current EntropyShop baseline was developed using all 200 public sessions. "
            "This split is a future-change guard, not an unbiased retrospective test."
        ),
        "folds": folds,
        "splits": named_splits,
    }
    validate_manifest(manifest, samples)
    return manifest


def validate_manifest(manifest: dict, samples: list[dict]) -> None:
    """Raise when a manifest omits, duplicates, or invents public samples."""
    expected = {str(sample["sample_id"]) for sample in samples}
    observed: list[str] = []
    folds = manifest.get("folds") or {}
    if len(folds) != int(manifest.get("fold_count", 0)):
        raise ValueError("fold count does not match fold entries")
    for fold in folds.values():
        observed.extend(str(value) for value in fold.get("sample_ids") or [])
    if len(observed) != len(set(observed)):
        raise ValueError("a sample appears in more than one fold")
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise ValueError(f"manifest/sample mismatch; missing={missing}, extra={extra}")

    fold_names = set(folds)
    split_folds = [name for names in (manifest.get("splits") or {}).values() for name in names]
    if set(split_folds) != fold_names or len(split_folds) != len(set(split_folds)):
        raise ValueError("named splits must partition the folds exactly once")


def fold_names_for(manifest: dict, selection: str) -> list[str]:
    """Resolve a named split, fold, or the complete set to fold names."""
    folds = manifest["folds"]
    if selection == "all":
        return list(folds)
    if selection in folds:
        return [selection]
    selected_folds = (manifest.get("splits") or {}).get(selection)
    if not selected_folds:
        available = [*manifest.get("splits", {}), *folds, "all"]
        raise ValueError(f"unknown split {selection!r}; choose from {', '.join(available)}")
    return list(selected_folds)


def sample_ids_for(manifest: dict, selection: str) -> set[str]:
    """Resolve a named split, fold, or the complete set to sample IDs."""
    folds = manifest["folds"]
    selected_folds = fold_names_for(manifest, selection)
    return {
        str(sample_id)
        for fold_name in selected_folds
        for sample_id in folds[fold_name]["sample_ids"]
    }


def select_samples(samples: list[dict], sample_ids: set[str]) -> list[dict]:
    """Select IDs while preserving the organizer's source-file order."""
    selected = [sample for sample in samples if str(sample.get("sample_id")) in sample_ids]
    found = {str(sample["sample_id"]) for sample in selected}
    if found != sample_ids:
        raise ValueError(f"dataset is missing manifest IDs: {sorted(sample_ids - found)}")
    return selected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate public-session folds")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--check", action="store_true", help="verify the committed manifest")
    parser.add_argument("--force", action="store_true", help="replace an existing manifest")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    source_path = args.dataset.relative_to(ROOT) if args.dataset.is_relative_to(ROOT) else args.dataset
    manifest = build_manifest(
        samples,
        seed=args.seed,
        fold_count=args.folds,
        source=source_path.as_posix(),
        source_sha256=_file_sha256(args.dataset),
    )
    rendered = json.dumps(manifest, indent=2) + "\n"

    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"split manifest is missing or stale: {args.output}")
        print(f"manifest verified: {args.output}")
    else:
        if args.output.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {args.output}; use --force")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")

    for name, fold in manifest["folds"].items():
        print(f"{name}: n={len(fold['sample_ids'])} {fold['scenario_counts']}")


if __name__ == "__main__":
    main()
