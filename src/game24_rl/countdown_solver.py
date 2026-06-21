"""Exact DFS utilities for Countdown-style arithmetic target puzzles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from game24_rl.solver import Solution, format_fraction

MIN_COUNTDOWN_NUMBERS = 3
MAX_COUNTDOWN_NUMBERS = 4


@dataclass(frozen=True)
class CountdownSolutionCount:
    """Unique expression count for one Countdown-style puzzle."""

    numbers: tuple[int, ...]
    target: Fraction
    count: int
    capped: bool


@dataclass(frozen=True)
class _Term:
    value: Fraction
    expression: str


def normalize_countdown_numbers(numbers: Sequence[int]) -> tuple[int, ...]:
    """Returns a sorted 3- or 4-number multiset identity."""

    normalized: list[int] = []
    for number in numbers:
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError(f"countdown numbers must be integers: {numbers!r}")
        normalized.append(number)

    if not MIN_COUNTDOWN_NUMBERS <= len(normalized) <= MAX_COUNTDOWN_NUMBERS:
        raise ValueError(
            "expected 3 or 4 countdown numbers: "
            f"{numbers!r}"
        )

    return tuple(sorted(normalized))


def count_countdown_solutions(
    numbers: Sequence[int],
    target: int | Fraction,
    *,
    cap: int | None = None,
) -> CountdownSolutionCount:
    """Counts unique exact final expressions for one target puzzle.

    The count is over verifier-style expressions, not over human-distinct
    derivation narratives. When ``cap`` is provided, search stops once that many
    unique expressions have been found.
    """

    if cap is not None and cap < 1:
        raise ValueError("cap must be at least 1")

    puzzle = normalize_countdown_numbers(numbers)
    target_fraction = Fraction(target)
    seen_expressions: set[str] = set()
    _search_count(
        state=tuple(_Term(Fraction(number), str(number)) for number in puzzle),
        target=target_fraction,
        seen_expressions=seen_expressions,
        cap=cap,
    )
    capped = cap is not None and len(seen_expressions) >= cap
    return CountdownSolutionCount(
        numbers=puzzle,
        target=target_fraction,
        count=len(seen_expressions),
        capped=capped,
    )


def solve_countdown_puzzle(
    numbers: Sequence[int],
    target: int | Fraction,
    *,
    max_solutions: int = 1,
) -> list[Solution]:
    """Finds deterministic exact solutions for a 3- or 4-number target puzzle."""

    if max_solutions < 1:
        raise ValueError("max_solutions must be at least 1")

    puzzle = normalize_countdown_numbers(numbers)
    target_fraction = Fraction(target)
    solutions: list[Solution] = []
    seen_expressions: set[str] = set()
    _search_solutions(
        state=tuple(_Term(Fraction(number), str(number)) for number in puzzle),
        trace=(),
        puzzle=puzzle,
        target=target_fraction,
        max_solutions=max_solutions,
        seen_expressions=seen_expressions,
        solutions=solutions,
    )
    return solutions


def _search_count(
    *,
    state: tuple[_Term, ...],
    target: Fraction,
    seen_expressions: set[str],
    cap: int | None,
) -> None:
    if cap is not None and len(seen_expressions) >= cap:
        return

    if len(state) == 1:
        expression = state[0].expression
        if state[0].value == target:
            seen_expressions.add(expression)
        return

    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            rest = tuple(state[k] for k in range(len(state)) if k not in {i, j})
            for left, operator, right, value in _candidate_operations(
                state[i], state[j]
            ):
                next_state = tuple(
                    sorted(
                        [
                            *rest,
                            _Term(
                                value,
                                (
                                    f"({left.expression} {operator} "
                                    f"{right.expression})"
                                ),
                            ),
                        ],
                        key=_term_sort_key,
                    )
                )
                _search_count(
                    state=next_state,
                    target=target,
                    seen_expressions=seen_expressions,
                    cap=cap,
                )
                if cap is not None and len(seen_expressions) >= cap:
                    return


def _search_solutions(
    *,
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
                    numbers=puzzle,
                    expression=expression,
                    trace=trace,
                    target=target,
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
                next_state = tuple(sorted([*rest, new_term], key=_term_sort_key))
                next_trace = trace + (
                    _format_trace_line(left, operator, right, value, rest),
                )
                _search_solutions(
                    state=next_state,
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
