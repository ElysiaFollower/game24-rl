"""SFT data generation for 24-point puzzles."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from game24_rl.datasets import puzzle_id
from game24_rl.solver import DEFAULT_TARGET, Solution, format_fraction, solve_puzzle
from game24_rl.verifier import verify_answer

PROMPT_STYLE_PLAIN = "plain"
PROMPT_STYLE_QWEN_CHAT = "qwen_chat"
PROMPT_STYLE_QWEN_CHAT_SEARCH = "qwen_chat_search_v1"
SUPPORTED_PROMPT_STYLES = frozenset(
    {PROMPT_STYLE_PLAIN, PROMPT_STYLE_QWEN_CHAT, PROMPT_STYLE_QWEN_CHAT_SEARCH}
)

TRACE_TYPE_SHORT_SUCCESS = "short_success"
TRACE_TYPE_CHECKED_SUCCESS = "checked_success"
SUPPORTED_TRACE_TYPES = frozenset(
    {TRACE_TYPE_SHORT_SUCCESS, TRACE_TYPE_CHECKED_SUCCESS}
)

_QWEN_CHAT_SYSTEM_PROMPT = (
    "Play the 24-point game. Given four numbers, reach 24 using +, -, *, "
    "and /, and use each provided number exactly once. Output concise "
    "reasoning steps. End with exactly one final expression inside "
    "<answer>...</answer>."
)

_QWEN_CHAT_SEARCH_SYSTEM_PROMPT = (
    "Play the 24-point game. Given four numbers, determine whether 24 is "
    "reachable using +, -, *, and /. Use each provided number exactly once. "
    "Output reasoning steps, one step per line. On the last line, output the "
    "final expression inside <answer>...</answer>.\n\n"
    "Example:\n"
    "Input numbers: 10 1 12 3\n"
    "Output:\n"
    "<think>\n"
    "(12) + (3) = 15, left: 15, 10, 1\n"
    "(10) - (1) = 9, left: 9, 15\n"
    "(15) + (9) = 24, left: 24\n"
    "</think>\n"
    "<answer>((12 + 3) + (10 - 1))</answer>"
)


def format_prompt(
    numbers: Sequence[int],
    *,
    prompt_style: str = PROMPT_STYLE_PLAIN,
) -> str:
    """Formats the SFT prompt for one standard 24-point puzzle."""

    _validate_prompt_style(prompt_style)
    joined = " ".join(str(number) for number in numbers)
    if prompt_style == PROMPT_STYLE_QWEN_CHAT:
        return (
            f"<|im_start|>system\n{_QWEN_CHAT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{joined}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if prompt_style == PROMPT_STYLE_QWEN_CHAT_SEARCH:
        return (
            f"<|im_start|>system\n{_QWEN_CHAT_SEARCH_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nInput numbers: {joined}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return (
        "Solve the 24-point game. Use each provided number exactly once: "
        f"{joined}. Return a short <think> trace and one <answer> expression.\n"
    )


def format_completion(
    solution: Solution,
    *,
    trace_type: str = TRACE_TYPE_SHORT_SUCCESS,
) -> str:
    """Formats a short-success-trace completion."""

    _validate_trace_type(trace_type)
    trace = "\n".join(solution.trace)
    if trace_type == TRACE_TYPE_CHECKED_SUCCESS:
        trace = "\n".join(
            [
                trace,
                (
                    "Check: the final expression "
                    f"{solution.expression} = {format_fraction(solution.target)}."
                ),
            ]
        )
    return f"<think>\n{trace}\n</think>\n<answer>{solution.expression}</answer>"


def build_sft_records(
    puzzles: Iterable[Sequence[int]],
    traces_per_puzzle: int = 8,
    target: int = DEFAULT_TARGET,
    *,
    trace_type: str = TRACE_TYPE_SHORT_SUCCESS,
    prompt_style: str = PROMPT_STYLE_PLAIN,
) -> list[dict[str, Any]]:
    """Builds first-pass SFT records from solvable puzzles.

    Args:
        puzzles: Solvable puzzle numbers.
        traces_per_puzzle: Maximum unique solutions to emit per puzzle.
        target: Target value, normally 24.
        trace_type: Completion trace format to generate.
        prompt_style: Prompt wrapper style to use.

    Raises:
        ValueError: If any input puzzle has no exact solution.
    """

    _validate_trace_type(trace_type)
    _validate_prompt_style(prompt_style)
    records: list[dict[str, Any]] = []
    for puzzle in puzzles:
        solutions = solve_puzzle(puzzle, target=target, max_solutions=traces_per_puzzle)
        if not solutions:
            raise ValueError(f"cannot build SFT record for unsolvable puzzle: {puzzle}")

        for index, solution in enumerate(solutions):
            completion = format_completion(solution, trace_type=trace_type)
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
                    "prompt": format_prompt(
                        solution.numbers,
                        prompt_style=prompt_style,
                    ),
                    "completion": completion,
                    "answer": solution.expression,
                    "trace": list(solution.trace),
                    "trace_type": trace_type,
                    "prompt_style": prompt_style,
                    "source": f"exact_solver_{trace_type}",
                }
            )
    return records


def records_from_split_manifest(
    manifest: dict[str, Any],
    split: str = "train",
    traces_per_puzzle: int = 8,
    *,
    trace_type: str = TRACE_TYPE_SHORT_SUCCESS,
    prompt_style: str = PROMPT_STYLE_PLAIN,
) -> list[dict[str, Any]]:
    """Builds SFT records from a named solvable split."""

    puzzles = [record["numbers"] for record in manifest["splits"][split]]
    return build_sft_records(
        puzzles,
        traces_per_puzzle=traces_per_puzzle,
        target=manifest["target"],
        trace_type=trace_type,
        prompt_style=prompt_style,
    )


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Writes records to JSON Lines."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _validate_trace_type(trace_type: str) -> None:
    if trace_type not in SUPPORTED_TRACE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_TRACE_TYPES))
        raise ValueError(f"unsupported trace_type {trace_type!r}; expected {supported}")


def _validate_prompt_style(prompt_style: str) -> None:
    if prompt_style not in SUPPORTED_PROMPT_STYLES:
        supported = ", ".join(sorted(SUPPORTED_PROMPT_STYLES))
        raise ValueError(
            f"unsupported prompt_style {prompt_style!r}; expected {supported}"
        )
