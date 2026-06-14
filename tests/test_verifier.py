"""Tests for strict answer verification."""

import pytest

from game24_rl.verifier import (
    evaluate_arithmetic_expression,
    extract_answer_expression,
    verify_answer,
)


def test_accepts_valid_answer_contract() -> None:
    result = verify_answer(
        "<think>short trace</think><answer>((8 - 2) * (7 - 3))</answer>",
        puzzle=(8, 2, 7, 3),
    )

    assert result.valid
    assert result.reason == "ok"
    assert result.value == 24
    assert result.numbers == (2, 3, 7, 8)


@pytest.mark.parametrize(
    ("output", "reason_prefix"),
    [
        ("((8 - 2) * (7 - 3))", "answer_contract"),
        (
            "<answer>((8 - 2) * (7 - 3))</answer><answer>24</answer>",
            "answer_contract",
        ),
        ("<answer></answer>", "answer_contract"),
    ],
)
def test_rejects_bad_answer_contract(output: str, reason_prefix: str) -> None:
    result = verify_answer(output, puzzle=(8, 2, 7, 3))

    assert not result.valid
    assert result.reason.startswith(reason_prefix)


@pytest.mark.parametrize(
    "output",
    [
        "<answer>((8 - 2) * (7 - 4))</answer>",
        "<answer>((8 - 2) * (7 - 2))</answer>",
        "<answer>((8 - 2) * 7)</answer>",
    ],
)
def test_rejects_wrong_repeated_or_omitted_numbers(output: str) -> None:
    result = verify_answer(output, puzzle=(8, 2, 7, 3))

    assert not result.valid
    assert result.reason == "wrong_numbers"


def test_rejects_wrong_target_value() -> None:
    result = verify_answer("<answer>8 + 2 + 7 + 3</answer>", puzzle=(8, 2, 7, 3))

    assert not result.valid
    assert result.reason == "wrong_value"
    assert result.value == 20


@pytest.mark.parametrize(
    ("output", "reason_prefix"),
    [
        ("<answer>((8 - 2) * (7 - ))</answer>", "syntax_error"),
        ("<answer>8.0 + 2 + 7 + 7</answer>", "unsupported_expression"),
        ("<answer>x + 2 + 7 + 8</answer>", "unsupported_expression"),
        ("<answer>abs(8) + 2 + 7 + 7</answer>", "unsupported_expression"),
        ("<answer>(8).__class__</answer>", "unsupported_expression"),
        ("<answer>-8 + 2 + 7 + 23</answer>", "unsupported_expression"),
        ("<answer>8 / (3 - 3) + 2 + 7</answer>", "division_by_zero"),
    ],
)
def test_rejects_unsupported_expression_forms(
    output: str,
    reason_prefix: str,
) -> None:
    result = verify_answer(output, puzzle=(8, 2, 7, 3))

    assert not result.valid
    assert result.reason.startswith(reason_prefix)


def test_extracts_only_single_answer_expression() -> None:
    assert (
        extract_answer_expression("<think>x</think><answer>8 + 2</answer>") == "8 + 2"
    )


def test_evaluates_fraction_arithmetic_without_float_rounding() -> None:
    evaluated = evaluate_arithmetic_expression("(8 / 3) * 9")

    assert evaluated.value == 24
    assert evaluated.numbers == (8, 3, 9)
