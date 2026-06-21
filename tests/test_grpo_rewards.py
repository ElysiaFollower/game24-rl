"""Tests for GRPO reward and answer-closure diagnostics."""

from game24_rl.rewards import (
    GRPO_CLOSE_BONUS_REWARD_VERSION,
    GRPO_CLOSURE_CONTROL_SMOOTH_REWARD_VERSION,
    GRPO_CLOSURE_STRICT_REWARD_VERSION,
    GRPO_REWARD_VERSION,
    GRPO_TARGET_ALIGNMENT_REWARD_VERSION,
    GRPO_TARGET_DISTANCE_REWARD_VERSION,
    answer_closure_metrics,
    reward_completions,
    score_completion,
)


def test_score_completion_rewards_strict_success() -> None:
    score = score_completion(
        "<think>done</think><answer>((8 - 2) * (7 - 3))</answer>",
        numbers=[8, 2, 7, 3],
        target=24,
    )

    assert score.reward == 1.0
    assert score.valid is True
    assert score.reward_version == GRPO_REWARD_VERSION
    assert score.reason == "ok"
    assert score.has_complete_answer is True


def test_score_completion_penalizes_missing_answer_more_than_wrong_answer() -> None:
    missing = score_completion(
        "<think>still searching",
        numbers=[8, 2, 7, 3],
        target=24,
    )
    wrong = score_completion(
        "<answer>8 + 2 + 7 + 3</answer>",
        numbers=[8, 2, 7, 3],
        target=24,
    )

    assert missing.reward == -0.2
    assert missing.reward_reason == "missing_or_incomplete_answer"
    assert wrong.reward == -0.1
    assert wrong.reward_reason == "parseable_wrong_answer"


def test_close_bonus_profile_only_rewards_valid_early_closure() -> None:
    early = score_completion(
        "<answer>((8 - 2) * (7 - 3))</answer>",
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="close_bonus",
    )
    wrong = score_completion(
        "<answer>8 + 2 + 7 + 3</answer>",
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="close_bonus",
    )

    assert early.reward == 1.1
    assert early.reward_reason == "strict_correct_close_le_256"
    assert early.reward_version == GRPO_CLOSE_BONUS_REWARD_VERSION
    assert wrong.reward == -0.1
    assert wrong.reward_reason == "parseable_wrong_answer"


def test_closure_strict_profile_penalizes_unclosed_search_more() -> None:
    correct = score_completion(
        "<answer>((8 - 2) * (7 - 3))</answer>",
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="closure_strict",
    )
    missing = score_completion(
        "<think>still searching",
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="closure_strict",
    )
    wrong = score_completion(
        "<answer>8 + 2 + 7 + 3</answer>",
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="closure_strict",
    )

    assert correct.reward == 1.1
    assert correct.reward_reason == "closure_strict_correct_close_le_256"
    assert correct.reward_version == GRPO_CLOSURE_STRICT_REWARD_VERSION
    assert missing.reward == -0.5
    assert missing.reward_reason == "closure_strict_missing_or_incomplete_answer"
    assert wrong.reward == -0.2
    assert wrong.reward_reason == "closure_strict_parseable_wrong_answer"


def test_closure_control_smooth_rewards_correct_answer_with_bonus() -> None:
    early = score_completion(
        "<answer>((8-2)*(7-3))</answer>",
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="closure_control_smooth",
    )
    late_output = " ".join(
        ["scratch"] * 1024 + ["<answer>((8 - 2) * (7 - 3))</answer>"]
    )
    late = score_completion(
        late_output,
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="closure_control_smooth",
    )

    assert early.reward_version == GRPO_CLOSURE_CONTROL_SMOOTH_REWARD_VERSION
    assert early.reward_reason == "closure_control_smooth_correct"
    assert early.reward == 1.0 + 0.25 * (1 - 1 / 4096)
    assert late.reward == 1.0 + 0.25 * (1 - 1031 / 4096)
    assert late.reward > 1.0


def test_closure_control_smooth_uses_fixed_error_rewards() -> None:
    missing = score_completion(
        "<think>still searching",
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="closure_control_smooth",
    )
    multiple = score_completion(
        "<answer>8 + 2 + 7 + 3</answer><answer>8 + 2 + 7 + 3</answer>",
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="closure_control_smooth",
    )
    syntax = score_completion(
        "<answer>((8 - 2) * (7 - 3)</answer>",
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="closure_control_smooth",
    )
    wrong_numbers = score_completion(
        "<answer>(8 * 3)</answer>",
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="closure_control_smooth",
    )
    wrong_value = score_completion(
        "<answer>8 + 2 + 7 + 3</answer>",
        numbers=[8, 2, 7, 3],
        target=24,
        reward_profile="closure_control_smooth",
    )

    assert missing.reward == -0.3
    assert missing.reward_reason == "closure_control_smooth_no_complete_answer"
    assert multiple.reward == -0.35
    assert multiple.reward_reason == "closure_control_smooth_multiple_answer_blocks"
    assert syntax.reward == -0.3
    assert syntax.reward_reason == "closure_control_smooth_syntax_or_unsupported"
    assert wrong_numbers.reward == -0.35
    assert wrong_numbers.reward_reason == "closure_control_smooth_wrong_numbers"
    assert wrong_value.reward == -0.35
    assert wrong_value.reward_reason == "closure_control_smooth_wrong_value"


def test_target_alignment_profile_strongly_penalizes_wrong_target() -> None:
    correct = score_completion(
        "<answer>(91 - (70 / 70))</answer>",
        numbers=[70, 70, 91],
        target=90,
        reward_profile="target_alignment",
    )
    wrong_target = score_completion(
        "<answer>(70 / 70 + 91)</answer>",
        numbers=[70, 70, 91],
        target=90,
        reward_profile="target_alignment",
    )
    missing = score_completion(
        "<think>still searching",
        numbers=[70, 70, 91],
        target=90,
        reward_profile="target_alignment",
    )

    assert correct.reward == 1.0
    assert correct.reward_reason == "target_alignment_correct"
    assert correct.reward_version == GRPO_TARGET_ALIGNMENT_REWARD_VERSION
    assert wrong_target.reward == -1.0
    assert wrong_target.reward_reason == "target_alignment_wrong_value"
    assert missing.reward == -0.5
    assert missing.reward_reason == "target_alignment_missing_or_incomplete_answer"


def test_target_distance_profile_rewards_closeness_to_target() -> None:
    near = score_completion(
        "<answer>91 + (70 / 70)</answer>",
        numbers=[70, 70, 91],
        target=90,
        reward_profile="target_distance",
    )
    far = score_completion(
        "<answer>((70 + 70) - 91)</answer>",
        numbers=[70, 70, 91],
        target=90,
        reward_profile="target_distance",
    )
    correct = score_completion(
        "<answer>(91 - (70 / 70))</answer>",
        numbers=[70, 70, 91],
        target=90,
        reward_profile="target_distance",
    )

    assert correct.reward == 1.0
    assert correct.reward_version == GRPO_TARGET_DISTANCE_REWARD_VERSION
    assert near.reward == -(2 / 90)
    assert near.reward_reason == "target_distance_wrong_value"
    assert far.reward == -(41 / 90)
    assert far.reward < near.reward


def test_reward_completions_accepts_trl_style_extra_columns() -> None:
    rewards = reward_completions(
        completions=[
            "<answer>((8 - 2) * (7 - 3))</answer>",
            "<answer>8 + 2 + 7 + 3</answer>",
        ],
        numbers=[[8, 2, 7, 3], [8, 2, 7, 3]],
        target=[24, 24],
        id=["ok", "bad"],
    )

    assert rewards == [1.0, -0.1]


def test_answer_closure_metrics_tracks_close_and_trailing_text() -> None:
    metrics = answer_closure_metrics(
        "<think>x</think><answer>8 + 2 + 7 + 3</answer> trailing text"
    )

    assert metrics.has_answer_open is True
    assert metrics.has_answer_close is True
    assert metrics.has_complete_answer is True
    assert metrics.answer_close_token_index is not None
    assert metrics.tokens_after_answer == 2
    assert metrics.answer_open_count == 1
    assert metrics.answer_close_count == 1
