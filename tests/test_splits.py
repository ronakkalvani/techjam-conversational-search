"""Tests for deterministic public-session fold infrastructure."""

from __future__ import annotations

import copy

import pytest

from scripts.make_splits import (
    build_manifest,
    fold_names_for,
    sample_ids_for,
    select_samples,
    validate_manifest,
)
from scripts.run_split_eval import (
    fold_summary,
    includes_internal_test,
    requested_selections,
    write_payload,
)


SCENARIO_COUNTS = {
    "buying": 80,
    "browsing": 80,
    "intent_override": 30,
    "boundary": 10,
}


def _samples() -> list[dict]:
    return [
        {"sample_id": f"{scenario}_{index:03d}", "scenario_type": scenario}
        for scenario, count in SCENARIO_COUNTS.items()
        for index in range(count)
    ]


def test_manifest_is_deterministic_and_input_order_independent():
    samples = _samples()
    forward = build_manifest(samples)
    reverse = build_manifest(list(reversed(samples)))
    assert forward == reverse


def test_folds_have_exact_official_scenario_mix():
    manifest = build_manifest(_samples())
    expected = {"boundary": 2, "browsing": 16, "buying": 16, "intent_override": 6}
    assert len(manifest["folds"]) == 5
    for fold in manifest["folds"].values():
        assert len(fold["sample_ids"]) == 40
        assert fold["scenario_counts"] == expected


def test_named_splits_partition_every_sample_once():
    samples = _samples()
    manifest = build_manifest(samples)
    development = sample_ids_for(manifest, "development")
    validation = sample_ids_for(manifest, "validation")
    test = sample_ids_for(manifest, "test")
    assert (len(development), len(validation), len(test)) == (120, 40, 40)
    assert not development & validation
    assert not development & test
    assert not validation & test
    assert development | validation | test == sample_ids_for(manifest, "all")
    assert fold_names_for(manifest, "test") == ["fold_5"]


def test_manifest_validation_rejects_duplicate_assignment():
    samples = _samples()
    manifest = build_manifest(samples)
    corrupted = copy.deepcopy(manifest)
    duplicate = corrupted["folds"]["fold_1"]["sample_ids"][0]
    corrupted["folds"]["fold_2"]["sample_ids"].append(duplicate)
    with pytest.raises(ValueError, match="more than one fold"):
        validate_manifest(corrupted, samples)


def test_selection_preserves_source_order_and_rejects_missing_ids():
    samples = _samples()
    wanted = {samples[7]["sample_id"], samples[2]["sample_id"]}
    selected = select_samples(samples, wanted)
    assert [sample["sample_id"] for sample in selected] == [
        samples[2]["sample_id"],
        samples[7]["sample_id"],
    ]
    with pytest.raises(ValueError, match="missing manifest IDs"):
        select_samples(samples, {"not_present"})


def test_fold_selection_and_summary_helpers():
    manifest = build_manifest(_samples())
    assert requested_selections(manifest, "folds") == [
        "fold_1", "fold_2", "fold_3", "fold_4", "fold_5"
    ]
    rows = [
        {"technical_score": 0.90},
        {"technical_score": 0.95},
        {"technical_score": 1.00},
    ]
    summary = fold_summary(rows)
    assert summary == {
        "fold_count": 3,
        "technical_score_mean": 0.95,
        "technical_score_min": 0.9,
        "technical_score_max": 1.0,
        "technical_score_stdev": 0.05,
    }
    assert includes_internal_test(manifest, ["validation"]) is False
    assert includes_internal_test(manifest, ["test"]) is True
    assert includes_internal_test(manifest, ["fold_5"]) is True
    assert includes_internal_test(manifest, requested_selections(manifest, "folds")) is True


def test_result_writer_creates_requested_parent(tmp_path):
    output = tmp_path / "nested" / "folds.json"
    assert write_payload(output, {"ok": True}) == output
    assert output.read_text(encoding="utf-8") == '{\n  "ok": true\n}\n'
