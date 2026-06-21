"""Tests for Countdown-style arithmetic target solving."""

from game24_rl.countdown_solver import (
    count_countdown_solutions,
    normalize_countdown_numbers,
    solve_countdown_puzzle,
)
from game24_rl.verifier import verify_answer


def test_normalizes_three_or_four_countdown_numbers() -> None:
    assert normalize_countdown_numbers([8, 3, 7]) == (3, 7, 8)
    assert normalize_countdown_numbers([8, 3, 7, 2]) == (2, 3, 7, 8)


def test_solves_three_number_target_puzzle() -> None:
    solutions = solve_countdown_puzzle([3, 7, 8], target=32, max_solutions=3)

    assert solutions
    for solution in solutions:
        result = verify_answer(
            f"<answer>{solution.expression}</answer>",
            puzzle=[3, 7, 8],
            target=32,
        )
        assert result.valid, result.reason
        assert len(solution.trace) == 2


def test_counts_unique_solutions_and_cap() -> None:
    uncapped = count_countdown_solutions([1, 2, 3], target=6)
    capped = count_countdown_solutions([1, 2, 3], target=6, cap=2)

    assert uncapped.count >= 2
    assert not uncapped.capped
    assert capped.count == 2
    assert capped.capped


def test_reports_zero_solution_count() -> None:
    result = count_countdown_solutions([1, 1, 1], target=100)

    assert result.count == 0
    assert not result.capped
