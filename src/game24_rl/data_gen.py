"""SFT data generation for 24-point puzzles."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from game24_rl.datasets import puzzle_id
from game24_rl.solver import DEFAULT_TARGET, Solution, solve_puzzle
from game24_rl.verifier import verify_answer


def format_prompt(numbers: Sequence[int]) -> str:
    """Formats the SFT prompt for one standard 24-point puzzle."""

    joined = " ".join(str(number) for number in numbers)
    return (
        "Solve the 24-point game. Use each provided number exactly once: "
        f"{joined}. Return a short <think> trace and one <answer> expression."
    )


def format_completion(solution: Solution) -> str:
    """Formats a short-success-trace completion."""

    trace = "\n".join(solution.trace)
    return f"<think>\n{trace}\n</think>\n<answer>{solution.expression}</answer>"


def build_sft_records(
    puzzles: Iterable[Sequence[int]],
    traces_per_puzzle: int = 8,
    target: int = DEFAULT_TARGET,
) -> list[dict[str, Any]]:
    """Builds first-pass SFT records from solvable puzzles.

    Args:
        puzzles: Solvable puzzle numbers.
        traces_per_puzzle: Maximum unique solutions to emit per puzzle.
        target: Target value, normally 24.

    Raises:
        ValueError: If any input puzzle has no exact solution.
    """

    records: list[dict[str, Any]] = []
    for puzzle in puzzles:
        solutions = solve_puzzle(puzzle, target=target, max_solutions=traces_per_puzzle)
        if not solutions:
            raise ValueError(f"cannot build SFT record for unsolvable puzzle: {puzzle}")

        for index, solution in enumerate(solutions):
            completion = format_completion(solution)
            verification = verify_answer(completion, solution.numbers, target=target)
            if not verification.valid:
                raise ValueError(
                    f"generated invalid solution for {solution.numbers}: "
                    f"{verification.reason}"
                )

            records.append(
                {
                    "id": f"game24-{puzzle_id(solution.numbers)}-{index:02d}",
                    "numbers": list(solution.numbers),
                    "target": target,
                    "prompt": format_prompt(solution.numbers),
                    "completion": completion,
                    "answer": solution.expression,
                    "trace": list(solution.trace),
                    "source": "exact_solver_short_trace",
                }
            )
    return records


def records_from_split_manifest(
    manifest: dict[str, Any],
    split: str = "train",
    traces_per_puzzle: int = 8,
) -> list[dict[str, Any]]:
    """Builds SFT records from a named solvable split."""

    puzzles = [record["numbers"] for record in manifest["splits"][split]]
    return build_sft_records(
        puzzles,
        traces_per_puzzle=traces_per_puzzle,
        target=manifest["target"],
    )


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Writes records to JSON Lines."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
