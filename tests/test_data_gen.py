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


def test_qwen_chat_prompt_has_assistant_boundary() -> None:
    prompt = format_prompt((1, 6, 9, 12), prompt_style="qwen_chat")

    assert prompt.startswith("<|im_start|>system\n")
    assert "<|im_start|>user\n1 6 9 12\n<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_qwen_chat_target_prompt_includes_target() -> None:
    prompt = format_prompt((44, 19, 35), target=98, prompt_style="qwen_chat_target")

    assert prompt.startswith("<|im_start|>system\n")
    assert "<|im_start|>user\nNumbers: 44 19 35\nTarget: 98\n<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_checked_success_trace_keeps_verifier_contract() -> None:
    records = build_sft_records(
        [(8, 2, 7, 3)],
        traces_per_puzzle=1,
        trace_type="checked_success",
        prompt_style="qwen_chat",
    )

    record = records[0]
    assert "Check: the final expression" in record["completion"]
    assert "= 24." in record["completion"]
    assert record["trace_type"] == "checked_success"
    assert record["prompt_style"] == "qwen_chat"
    assert verify_answer(record["completion"], record["numbers"]).valid


def test_build_sft_records_rejects_unsolvable_puzzles() -> None:
    try:
        build_sft_records([(1, 1, 1, 1)])
    except ValueError as exc:
        assert "unsolvable" in str(exc)
    else:
        raise AssertionError("expected unsolvable puzzle to fail")


def test_build_sft_records_rejects_unknown_styles() -> None:
    try:
        build_sft_records([(8, 2, 7, 3)], trace_type="unknown")
    except ValueError as exc:
        assert "unsupported trace_type" in str(exc)
    else:
        raise AssertionError("expected unsupported trace_type to fail")

    try:
        build_sft_records([(8, 2, 7, 3)], prompt_style="unknown")
    except ValueError as exc:
        assert "unsupported prompt_style" in str(exc)
    else:
        raise AssertionError("expected unsupported prompt_style to fail")
