"""Tests for GRPO reward and answer-closure diagnostics."""

from game24_rl.rewards import (
    GRPO_REWARD_VERSION,
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
