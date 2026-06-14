"""Exact 24-point solver utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import combinations_with_replacement

STANDARD_MIN_NUMBER = 1
STANDARD_MAX_NUMBER = 13
PUZZLE_SIZE = 4
DEFAULT_TARGET = 24


@dataclass(frozen=True)
class Solution:
    """A deterministic arithmetic solution for one 24-point puzzle."""

    numbers: tuple[int, ...]
    expression: str
    trace: tuple[str, ...]
    target: Fraction = Fraction(DEFAULT_TARGET)


@dataclass(frozen=True)
class PuzzleClassification:
    """Solvable and unsolvable puzzle partitions for a standard puzzle set."""

    solvable: tuple[tuple[int, ...], ...]
    unsolvable: tuple[tuple[int, ...], ...]

    @property
    def total(self) -> int:
        """Returns the total number of classified puzzles."""

        return len(self.solvable) + len(self.unsolvable)


@dataclass(frozen=True)
class _Term:
    value: Fraction
    expression: str


def normalize_puzzle(numbers: Sequence[int]) -> tuple[int, ...]:
    """Returns the sorted multiset identity for a four-number puzzle.

    Args:
        numbers: Four integer puzzle numbers.

    Raises:
        ValueError: If the puzzle does not contain exactly four integers.
    """

    normalized: list[int] = []
    for number in numbers:
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError(f"puzzle numbers must be integers: {numbers!r}")
        normalized.append(number)

    if len(normalized) != PUZZLE_SIZE:
        raise ValueError(f"expected {PUZZLE_SIZE} puzzle numbers: {numbers!r}")

    return tuple(sorted(normalized))


def enumerate_standard_puzzles(
    max_number: int = STANDARD_MAX_NUMBER,
) -> list[tuple[int, ...]]:
    """Enumerates standard 24-point puzzle multisets.

    Args:
        max_number: Largest allowed integer. The standard game uses 13.

    Returns:
        Lexicographically sorted four-number multisets from 1..max_number.
    """

    if max_number < STANDARD_MIN_NUMBER:
        raise ValueError("max_number must be at least 1")
    return list(
        combinations_with_replacement(
            range(STANDARD_MIN_NUMBER, max_number + 1),
            PUZZLE_SIZE,
        )
    )


def format_fraction(value: Fraction) -> str:
    """Formats a rational value for traces."""

    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def is_solvable(
    numbers: Sequence[int], target: int | Fraction = DEFAULT_TARGET
) -> bool:
    """Returns whether a puzzle can exactly reach the target.

    Args:
        numbers: Four integer puzzle numbers.
        target: Target value, normally 24.
    """

    puzzle = normalize_puzzle(numbers)
    values = tuple(Fraction(number) for number in puzzle)
    return _can_reach(tuple(sorted(values)), Fraction(target))


def classify_standard_puzzles(
    max_number: int = STANDARD_MAX_NUMBER,
    target: int | Fraction = DEFAULT_TARGET,
) -> PuzzleClassification:
    """Classifies every standard multiset as solvable or unsolvable."""

    solvable: list[tuple[int, ...]] = []
    unsolvable: list[tuple[int, ...]] = []
    for puzzle in enumerate_standard_puzzles(max_number=max_number):
        if is_solvable(puzzle, target=target):
            solvable.append(puzzle)
        else:
            unsolvable.append(puzzle)
    return PuzzleClassification(tuple(solvable), tuple(unsolvable))


def solve_puzzle(
    numbers: Sequence[int],
    target: int | Fraction = DEFAULT_TARGET,
    max_solutions: int = 1,
) -> list[Solution]:
    """Finds deterministic exact solutions for a puzzle.

    Args:
        numbers: Four integer puzzle numbers.
        target: Target value, normally 24.
        max_solutions: Maximum number of unique expressions to return.

    Returns:
        Up to ``max_solutions`` solutions. An empty list means unsolvable.
    """

    if max_solutions < 1:
        raise ValueError("max_solutions must be at least 1")

    puzzle = normalize_puzzle(numbers)
    target_fraction = Fraction(target)
    terms = tuple(_Term(Fraction(number), str(number)) for number in puzzle)
    solutions: list[Solution] = []
    seen_expressions: set[str] = set()
    _search_solutions(
        state=tuple(sorted(terms, key=_term_sort_key)),
        trace=(),
        puzzle=puzzle,
        target=target_fraction,
        max_solutions=max_solutions,
        seen_expressions=seen_expressions,
        solutions=solutions,
    )
    return solutions


@cache
def _can_reach(values: tuple[Fraction, ...], target: Fraction) -> bool:
    if len(values) == 1:
        return values[0] == target

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            rest = [values[k] for k in range(len(values)) if k not in {i, j}]
            left = values[i]
            right = values[j]
            candidates = {
                left + right,
                left * right,
                left - right,
                right - left,
            }
            if right != 0:
                candidates.add(left / right)
            if left != 0:
                candidates.add(right / left)

            for value in candidates:
                next_values = tuple(sorted([*rest, value]))
                if _can_reach(next_values, target):
                    return True
    return False


def _search_solutions(
    state: tuple[_Term, ...],
    trace: tuple[str, ...],
    puzzle: tuple[int, ...],
    target: Fraction,
    max_solutions: int,
    seen_expressions: set[str],
    solutions: list[Solution],
) -> None:
    if len(solutions) >= max_solutions:
        return

    if len(state) == 1:
        expression = state[0].expression
        if state[0].value == target and expression not in seen_expressions:
            seen_expressions.add(expression)
            solutions.append(
                Solution(
                    numbers=puzzle, expression=expression, trace=trace, target=target
                )
            )
        return

    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            rest = tuple(state[k] for k in range(len(state)) if k not in {i, j})
            for left, operator, right, value in _candidate_operations(
                state[i], state[j]
            ):
                new_term = _Term(
                    value, f"({left.expression} {operator} {right.expression})"
                )
                new_state = tuple(sorted([*rest, new_term], key=_term_sort_key))
                next_trace = trace + (
                    _format_trace_line(left, operator, right, value, rest),
                )
                _search_solutions(
                    state=new_state,
                    trace=next_trace,
                    puzzle=puzzle,
                    target=target,
                    max_solutions=max_solutions,
                    seen_expressions=seen_expressions,
                    solutions=solutions,
                )
                if len(solutions) >= max_solutions:
                    return


def _candidate_operations(
    left: _Term,
    right: _Term,
) -> list[tuple[_Term, str, _Term, Fraction]]:
    candidates = [
        (left, "+", right, left.value + right.value),
        (left, "*", right, left.value * right.value),
        (left, "-", right, left.value - right.value),
        (right, "-", left, right.value - left.value),
    ]
    if right.value != 0:
        candidates.append((left, "/", right, left.value / right.value))
    if left.value != 0:
        candidates.append((right, "/", left, right.value / left.value))
    return candidates


def _format_trace_line(
    left: _Term,
    operator: str,
    right: _Term,
    value: Fraction,
    rest: tuple[_Term, ...],
) -> str:
    remaining = [
        format_fraction(value),
        *(format_fraction(term.value) for term in rest),
    ]
    return (
        f"({format_fraction(left.value)}) {operator} ({format_fraction(right.value)}) "
        f"= {format_fraction(value)}, left: {', '.join(remaining)}"
    )


def _term_sort_key(term: _Term) -> tuple[Fraction, str]:
    return (term.value, term.expression)
