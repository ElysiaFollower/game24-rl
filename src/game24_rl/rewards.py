"""Verifier-backed rewards for Game24 GRPO experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from game24_rl.verifier import VerificationResult, verify_answer

GRPO_REWARD_VERSION = "strict-correctness-closure-v1"

CORRECT_REWARD = 1.0
MISSING_OR_INCOMPLETE_ANSWER_REWARD = -0.2
PARSEABLE_WRONG_ANSWER_REWARD = -0.1


@dataclass(frozen=True)
class AnswerClosureMetrics:
    """Token-position diagnostics for answer closure.

    The indices are whitespace-token indices. They are intentionally cheap and
    tokenizer-independent for local diagnostics; model-token indices can be
    added later in generation code when tokenizer outputs are available.
    """

    has_answer_open: bool
    has_answer_close: bool
    has_complete_answer: bool
    first_answer_token: int | None
    first_answer_close_token: int | None
    answer_close_token_index: int | None
    tokens_after_answer: int | None

    def as_dict(self) -> dict[str, Any]:
        """Returns JSON-serializable metrics."""

        return asdict(self)


@dataclass(frozen=True)
class GrpoRewardScore:
    """Reward and diagnostics for one completion."""

    reward: float
    reward_reason: str
    reward_version: str
    verifier_reason: str
    valid: bool
    expression: str | None
    has_complete_answer: bool
    answer_close_token_index: int | None
    tokens_after_answer: int | None

    @property
    def reason(self) -> str:
        """Compatibility alias for the verifier reason."""

        return self.verifier_reason

    def as_dict(self) -> dict[str, Any]:
        """Returns JSON-serializable reward diagnostics."""

        return asdict(self)


def score_completion(
    completion: str,
    *,
    numbers: Sequence[int],
    target: int = 24,
) -> GrpoRewardScore:
    """Scores one model completion with the GRPO reward v1 contract."""

    verification = verify_answer(completion, puzzle=numbers, target=target)
    closure = answer_closure_metrics(completion)
    reward, reward_reason = _reward_from_verification(verification, closure)
    return GrpoRewardScore(
        reward=reward,
        reward_reason=reward_reason,
        reward_version=GRPO_REWARD_VERSION,
        verifier_reason=verification.reason,
        valid=verification.valid,
        expression=verification.expression,
        has_complete_answer=closure.has_complete_answer,
        answer_close_token_index=closure.answer_close_token_index,
        tokens_after_answer=closure.tokens_after_answer,
    )


def reward_completions(
    *,
    completions: Sequence[str],
    numbers: Sequence[Sequence[int]],
    target: Sequence[int] | None = None,
    **_: Any,
) -> list[float]:
    """TRL-style reward function accepting dataset columns as keyword args."""

    if target is None:
        target = [24] * len(completions)
    if len(completions) != len(numbers) or len(completions) != len(target):
        raise ValueError("completions, numbers, and target must have equal length")

    return [
        score_completion(completion, numbers=puzzle, target=item_target).reward
        for completion, puzzle, item_target in zip(
            completions,
            numbers,
            target,
            strict=True,
        )
    ]


def answer_closure_metrics(completion: str) -> AnswerClosureMetrics:
    """Computes cheap answer-closure positions from raw completion text."""

    tokens = completion.split()
    first_open = _first_token_containing(tokens, "<answer>")
    first_close = _first_token_containing(tokens, "</answer>")
    has_open = first_open is not None
    has_close = first_close is not None
    has_complete = (
        has_open
        and has_close
        and first_open is not None
        and first_close is not None
        and first_open <= first_close
    )
    tokens_after = None
    if first_close is not None:
        tokens_after = max(0, len(tokens) - first_close - 1)

    return AnswerClosureMetrics(
        has_answer_open=has_open,
        has_answer_close=has_close,
        has_complete_answer=has_complete,
        first_answer_token=first_open,
        first_answer_close_token=first_close,
        answer_close_token_index=first_close,
        tokens_after_answer=tokens_after,
    )


def _reward_from_verification(
    verification: VerificationResult,
    closure: AnswerClosureMetrics,
) -> tuple[float, str]:
    if verification.valid:
        return CORRECT_REWARD, "strict_correct"
    if not closure.has_complete_answer or verification.reason.startswith(
        "answer_contract"
    ):
        return MISSING_OR_INCOMPLETE_ANSWER_REWARD, "missing_or_incomplete_answer"
    return PARSEABLE_WRONG_ANSWER_REWARD, "parseable_wrong_answer"


def _first_token_containing(tokens: Sequence[str], needle: str) -> int | None:
    for index, token in enumerate(tokens):
        if needle in token:
            return index
    return None
