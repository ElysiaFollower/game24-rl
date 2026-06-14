"""Tests for exact 24-point solving."""

from game24_rl.solver import (
    classify_standard_puzzles,
    enumerate_standard_puzzles,
    is_solvable,
    solve_puzzle,
)
from game24_rl.verifier import verify_answer


def test_enumerates_all_standard_multisets() -> None:
    puzzles = enumerate_standard_puzzles()

    assert len(puzzles) == 1820
    assert puzzles[0] == (1, 1, 1, 1)
    assert puzzles[-1] == (13, 13, 13, 13)
    assert len(set(puzzles)) == len(puzzles)


def test_classifies_expected_standard_counts() -> None:
    classification = classify_standard_puzzles()

    assert classification.total == 1820
    assert len(classification.solvable) == 1362
    assert len(classification.unsolvable) == 458


def test_solves_known_puzzles_and_returns_verifiable_expression() -> None:
    solutions = solve_puzzle((8, 2, 7, 3), max_solutions=3)

    assert solutions
    for solution in solutions:
        result = verify_answer(
            f"<answer>{solution.expression}</answer>",
            puzzle=(8, 2, 7, 3),
        )
        assert result.valid, result.reason
        assert len(solution.trace) == 3


def test_reports_unsolvable_puzzle() -> None:
    assert not is_solvable((1, 1, 1, 1))
    assert solve_puzzle((1, 1, 1, 1)) == []
