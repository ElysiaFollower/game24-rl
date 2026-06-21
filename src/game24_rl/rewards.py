"""Verifier-backed rewards for Game24 GRPO experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

from game24_rl.verifier import VerificationResult, verify_answer

GRPO_REWARD_VERSION = "strict-correctness-closure-v1"
GRPO_CLOSE_BONUS_REWARD_VERSION = "strict-correctness-close-bonus-v1"
GRPO_CLOSURE_STRICT_REWARD_VERSION = "strict-correctness-closure-strict-v1"
GRPO_CLOSURE_CONTROL_SMOOTH_REWARD_VERSION = "closure-control-smooth-v1"
GRPO_TARGET_ALIGNMENT_REWARD_VERSION = "target-alignment-v1"
GRPO_TARGET_DISTANCE_REWARD_VERSION = "target-distance-v1"

CORRECT_REWARD = 1.0
MISSING_OR_INCOMPLETE_ANSWER_REWARD = -0.2
PARSEABLE_WRONG_ANSWER_REWARD = -0.1
STRICT_MISSING_OR_INCOMPLETE_ANSWER_REWARD = -0.5
STRICT_PARSEABLE_WRONG_ANSWER_REWARD = -0.2
SMOOTH_MISSING_OR_INCOMPLETE_ANSWER_REWARD = -0.3
SMOOTH_SYNTAX_OR_UNSUPPORTED_REWARD = -0.3
SMOOTH_MULTIPLE_OR_WRONG_ANSWER_REWARD = -0.35
SMOOTH_CLOSE_BONUS_WEIGHT = 0.25
SMOOTH_CLOSE_TOKEN_BUDGET = 4096
TARGET_ALIGNMENT_MISSING_OR_INCOMPLETE_REWARD = -0.5
TARGET_ALIGNMENT_WRONG_TARGET_REWARD = -1.0
TARGET_DISTANCE_MISSING_OR_INCOMPLETE_REWARD = -0.5
TARGET_DISTANCE_PARSE_ERROR_REWARD = -0.75
TARGET_DISTANCE_WRONG_NUMBERS_REWARD = -1.0
EARLY_CLOSE_BONUS = 0.1
ON_TIME_CLOSE_BONUS = 0.05


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
    answer_open_count: int
    answer_close_count: int

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
    reward_profile: str = "strict",
) -> GrpoRewardScore:
    """Scores one model completion with the GRPO reward v1 contract."""

    verification = verify_answer(completion, puzzle=numbers, target=target)
    closure = answer_closure_metrics(completion)
    reward, reward_reason = _reward_from_verification(
        verification,
        closure,
        target=target,
        reward_profile=reward_profile,
    )
    reward_version = GRPO_REWARD_VERSION
    if reward_profile == "close_bonus":
        reward_version = GRPO_CLOSE_BONUS_REWARD_VERSION
    if reward_profile == "closure_strict":
        reward_version = GRPO_CLOSURE_STRICT_REWARD_VERSION
    if reward_profile == "closure_control_smooth":
        reward_version = GRPO_CLOSURE_CONTROL_SMOOTH_REWARD_VERSION
    if reward_profile == "target_alignment":
        reward_version = GRPO_TARGET_ALIGNMENT_REWARD_VERSION
    if reward_profile == "target_distance":
        reward_version = GRPO_TARGET_DISTANCE_REWARD_VERSION
    return GrpoRewardScore(
        reward=reward,
        reward_reason=reward_reason,
        reward_version=reward_version,
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
    reward_profile: str = "strict",
    **_: Any,
) -> list[float]:
    """TRL-style reward function accepting dataset columns as keyword args."""

    if target is None:
        target = [24] * len(completions)
    if len(completions) != len(numbers) or len(completions) != len(target):
        raise ValueError("completions, numbers, and target must have equal length")

    return [
        score_completion(
            completion,
            numbers=puzzle,
            target=item_target,
            reward_profile=reward_profile,
        ).reward
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
    open_count = completion.count("<answer>")
    close_count = completion.count("</answer>")
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
        answer_open_count=open_count,
        answer_close_count=close_count,
    )


def _reward_from_verification(
    verification: VerificationResult,
    closure: AnswerClosureMetrics,
    *,
    target: int | Fraction,
    reward_profile: str,
) -> tuple[float, str]:
    if verification.valid:
        if reward_profile == "close_bonus":
            return _reward_valid_with_close_bonus(closure)
        if reward_profile == "closure_strict":
            return _reward_valid_with_closure_strict(closure)
        if reward_profile == "closure_control_smooth":
            return _reward_valid_with_closure_control_smooth(closure)
        if reward_profile == "target_alignment":
            return CORRECT_REWARD, "target_alignment_correct"
        if reward_profile == "target_distance":
            return CORRECT_REWARD, "target_distance_correct"
        if reward_profile != "strict":
            raise ValueError(f"unknown reward profile: {reward_profile}")
        return CORRECT_REWARD, "strict_correct"
    if reward_profile == "target_alignment":
        return _reward_invalid_with_target_alignment(verification, closure)
    if reward_profile == "target_distance":
        return _reward_invalid_with_target_distance(
            verification,
            closure,
            target=Fraction(target),
        )
    if reward_profile == "closure_control_smooth":
        return _reward_invalid_with_closure_control_smooth(verification, closure)
    if not closure.has_complete_answer or verification.reason.startswith(
        "answer_contract"
    ):
        if reward_profile == "closure_strict":
            return (
                STRICT_MISSING_OR_INCOMPLETE_ANSWER_REWARD,
                "closure_strict_missing_or_incomplete_answer",
            )
        return MISSING_OR_INCOMPLETE_ANSWER_REWARD, "missing_or_incomplete_answer"
    if reward_profile == "closure_strict":
        return (
            STRICT_PARSEABLE_WRONG_ANSWER_REWARD,
            "closure_strict_parseable_wrong_answer",
        )
    return PARSEABLE_WRONG_ANSWER_REWARD, "parseable_wrong_answer"


def _reward_invalid_with_target_alignment(
    verification: VerificationResult,
    closure: AnswerClosureMetrics,
) -> tuple[float, str]:
    """Strongly penalizes parseable expressions that miss the requested target."""

    if not closure.has_complete_answer or verification.reason.startswith(
        "answer_contract"
    ):
        return (
            TARGET_ALIGNMENT_MISSING_OR_INCOMPLETE_REWARD,
            "target_alignment_missing_or_incomplete_answer",
        )
    if verification.reason == "wrong_value":
        return (
            TARGET_ALIGNMENT_WRONG_TARGET_REWARD,
            "target_alignment_wrong_value",
        )
    if verification.reason == "wrong_numbers":
        return (
            TARGET_ALIGNMENT_WRONG_TARGET_REWARD,
            "target_alignment_wrong_numbers",
        )
    if verification.reason.startswith(("syntax_error", "unsupported_expression")):
        return (
            TARGET_ALIGNMENT_WRONG_TARGET_REWARD,
            "target_alignment_syntax_or_unsupported",
        )
    return (
        TARGET_ALIGNMENT_WRONG_TARGET_REWARD,
        "target_alignment_parseable_wrong_answer",
    )


def _reward_invalid_with_target_distance(
    verification: VerificationResult,
    closure: AnswerClosureMetrics,
    *,
    target: Fraction,
) -> tuple[float, str]:
    """Rewards wrong-value expressions by closeness to the requested target."""

    if not closure.has_complete_answer or verification.reason.startswith(
        "answer_contract"
    ):
        return (
            TARGET_DISTANCE_MISSING_OR_INCOMPLETE_REWARD,
            "target_distance_missing_or_incomplete_answer",
        )
    if verification.reason == "wrong_value" and verification.value is not None:
        distance = abs(verification.value - target)
        return _target_distance_reward(distance=distance, target=target)
    if verification.reason == "wrong_numbers":
        return TARGET_DISTANCE_WRONG_NUMBERS_REWARD, "target_distance_wrong_numbers"
    if verification.reason.startswith(("syntax_error", "unsupported_expression")):
        return TARGET_DISTANCE_PARSE_ERROR_REWARD, "target_distance_parse_error"
    return TARGET_DISTANCE_PARSE_ERROR_REWARD, "target_distance_parseable_wrong_answer"


def _target_distance_reward(
    *,
    distance: Fraction,
    target: Fraction,
) -> tuple[float, str]:
    scale = max(abs(target), Fraction(1))
    penalty = min(Fraction(1), distance / scale)
    return -float(penalty), "target_distance_wrong_value"


def _reward_valid_with_close_bonus(
    closure: AnswerClosureMetrics,
) -> tuple[float, str]:
    if closure.answer_close_token_index is None:
        return CORRECT_REWARD, "strict_correct"
    if closure.answer_close_token_index <= 256:
        return CORRECT_REWARD + EARLY_CLOSE_BONUS, "strict_correct_close_le_256"
    if closure.answer_close_token_index <= 512:
        return CORRECT_REWARD + ON_TIME_CLOSE_BONUS, "strict_correct_close_le_512"
    return CORRECT_REWARD, "strict_correct_late_close"


def _reward_valid_with_closure_strict(
    closure: AnswerClosureMetrics,
) -> tuple[float, str]:
    if closure.answer_close_token_index is None:
        return CORRECT_REWARD, "closure_strict_correct"
    if closure.answer_close_token_index <= 256:
        return CORRECT_REWARD + EARLY_CLOSE_BONUS, "closure_strict_correct_close_le_256"
    if closure.answer_close_token_index <= 512:
        return (
            CORRECT_REWARD + ON_TIME_CLOSE_BONUS,
            "closure_strict_correct_close_le_512",
        )
    return CORRECT_REWARD, "closure_strict_correct_late_close"


def _reward_valid_with_closure_control_smooth(
    closure: AnswerClosureMetrics,
) -> tuple[float, str]:
    if closure.answer_close_token_index is None:
        return CORRECT_REWARD, "closure_control_smooth_correct_no_close_index"
    close_token = closure.answer_close_token_index + 1
    reward = CORRECT_REWARD + SMOOTH_CLOSE_BONUS_WEIGHT * (
        1 - close_token / SMOOTH_CLOSE_TOKEN_BUDGET
    )
    return reward, "closure_control_smooth_correct"


def _reward_invalid_with_closure_control_smooth(
    verification: VerificationResult,
    closure: AnswerClosureMetrics,
) -> tuple[float, str]:
    if verification.reason.startswith("answer_contract"):
        if closure.answer_open_count > 1 or closure.answer_close_count > 1:
            return (
                SMOOTH_MULTIPLE_OR_WRONG_ANSWER_REWARD,
                "closure_control_smooth_multiple_answer_blocks",
            )
        return (
            SMOOTH_MISSING_OR_INCOMPLETE_ANSWER_REWARD,
            "closure_control_smooth_no_complete_answer",
        )
    if verification.reason.startswith(("syntax_error", "unsupported_expression")):
        return (
            SMOOTH_SYNTAX_OR_UNSUPPORTED_REWARD,
            "closure_control_smooth_syntax_or_unsupported",
        )
    if verification.reason in {"division_by_zero"}:
        return (
            SMOOTH_SYNTAX_OR_UNSUPPORTED_REWARD,
            "closure_control_smooth_syntax_or_unsupported",
        )
    if verification.reason in {"wrong_numbers", "wrong_value"}:
        return (
            SMOOTH_MULTIPLE_OR_WRONG_ANSWER_REWARD,
            f"closure_control_smooth_{verification.reason}",
        )
    return (
        SMOOTH_MULTIPLE_OR_WRONG_ANSWER_REWARD,
        "closure_control_smooth_parseable_wrong_answer",
    )


def _first_token_containing(tokens: Sequence[str], needle: str) -> int | None:
    for index, token in enumerate(tokens):
        if needle in token:
            return index
    return None
