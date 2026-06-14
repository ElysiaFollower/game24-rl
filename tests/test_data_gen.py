"""Tests for SFT data generation."""

from game24_rl.data_gen import build_sft_records, format_prompt
from game24_rl.datasets import build_split_manifest, validate_no_split_overlap
from game24_rl.verifier import verify_answer


def test_split_manifest_has_expected_counts_and_no_overlap() -> None:
    manifest = build_split_manifest()

    assert manifest["counts"]["total"] == 1820
    assert manifest["counts"]["solvable"] == 1362
    assert manifest["counts"]["unsolvable"] == 458
    assert manifest["counts"]["train"] == 1089
    assert manifest["counts"]["validation"] == 136
    assert manifest["counts"]["test"] == 137

    validate_no_split_overlap(manifest)
    split_keys = {
        split: {record["key"] for record in records}
        for split, records in manifest["splits"].items()
    }
    assert split_keys["train"].isdisjoint(split_keys["validation"])
    assert split_keys["train"].isdisjoint(split_keys["test"])
    assert split_keys["validation"].isdisjoint(split_keys["test"])


def test_split_manifest_is_deterministic_for_seed() -> None:
    first = build_split_manifest(seed=1234)
    second = build_split_manifest(seed=1234)
    different = build_split_manifest(seed=4321)

    assert first == second
    assert first["splits"]["train"] != different["splits"]["train"]


def test_build_sft_records_are_verifier_valid() -> None:
    records = build_sft_records([(8, 2, 7, 3)], traces_per_puzzle=2)

    assert 1 <= len(records) <= 2
    for record in records:
        assert record["prompt"]
        assert "<think>" in record["completion"]
        assert "<answer>" in record["completion"]
        result = verify_answer(record["completion"], record["numbers"])
        assert result.valid, result.reason


def test_prompt_ends_with_separator_before_completion() -> None:
    prompt = format_prompt((1, 6, 9, 12))

    assert prompt.endswith("\n")
    assert not prompt.rstrip().endswith("<think>")


def test_build_sft_records_rejects_unsolvable_puzzles() -> None:
    try:
        build_sft_records([(1, 1, 1, 1)])
    except ValueError as exc:
        assert "unsolvable" in str(exc)
    else:
        raise AssertionError("expected unsolvable puzzle to fail")
