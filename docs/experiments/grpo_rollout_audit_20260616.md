# GRPO Rollout Audit

## Status

This is a pre-GRPO readiness audit for the strong full fine-tuned SFT
checkpoint:
`outputs/experiments/baseline_format_v2_full_5000_from800/final`.

The audit does not train the model. It samples multiple completions per prompt
to check whether GRPO would have group-level reward variance and whether the
remaining greedy failures already have correct trajectories in the model's
sampling distribution.

## Method

- Checkpoint:
  `outputs/experiments/baseline_format_v2_full_5000_from800/final`
- Split: `validation`
- Prompt style: `qwen_chat`
- Sampling: `temperature=0.8`, `top_p=0.95`
- Reward: repository strict verifier, binary correctness for the summary.
- Script: `scripts/experiments/audit_rollout_distribution.py`
- Artifacts:
  `outputs/experiments/baseline_format_v2_full_5000_from800/rollout_audit/`

## Results

| Audit | Prompts | Generations | Max new tokens | Output solve rate | pass@k | Mixed groups | Zero-std groups | Truncation-like failures | Length p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation pilot | `32` | `4` | `512` | `53/128 = 41.41%` | `24/32 = 75.00%` | `21` | `11` | `75/128` | `512 / 512` |
| validation pilot | `32` | `4` | `1024` | `86/128 = 67.19%` | `30/32 = 93.75%` | `16` | `16` | `41/128` | `1024 / 1024` |
| greedy failures only | `26` | `8` | `1024` | `94/208 = 45.19%` | `22/26 = 84.62%` | `19` | `7` | `113/208` | `1024 / 1024` |

For the `26` greedy failures from the final SFT validation run, sampled
rollouts solved `22` of them at least once with `G=8`. This means the model's
distribution already contains correct trajectories for most current greedy
failures, even though greedy decoding fails by continuing long rollback traces
until truncation.

## Interpretation

GRPO is justified as the next stage. The audit found substantial group-level
reward variance:

- In the `1024`-token validation pilot, `16/32` prompts had mixed correct and
  incorrect sampled completions.
- In the greedy-failure targeted audit, `19/26` prompts had mixed rewards.

This is exactly the condition GRPO needs: for the same prompt, some sampled
trajectories receive higher verifier reward than others. Training can therefore
raise the probability of the successful trajectories instead of relying on a
new supervised teacher.

The main risk is length/search-control. Many sampled completions still run to
the generation budget:

- `completion_len_p50 = 1024` and `completion_len_p95 = 1024` in the
  `1024`-token audits.
- `113/208` targeted greedy-failure samples still failed the answer contract,
  usually by not reaching `<answer>` before truncation.

Therefore the first GRPO run should not chase reward alone. It must monitor
completion length, truncation rate, answer-contract failures, and greedy
validation solve rate.

## Recommended Next Decision

Proceed to a conservative GRPO pilot from the strong SFT checkpoint.

Recommended reward shape:

- `+1.0` for strict verifier success.
- `0.0` for parseable but wrong answer.
- `-0.2` for missing/incomplete `<answer>` or truncation-like answer-contract
  failure.
- Add no independent format bonus in the first pilot.
- Add no length bonus initially, or only a tiny correctness-gated length bonus
  after the first pilot shows reward can improve without solve-rate regression.

Recommended training pool:

- Start with prompts that showed mixed rewards in rollout audit.
- Include a low-rate replay sample from all-correct prompts to avoid forgetting.
- Keep all-wrong prompts out of the first pilot or sample them at very low rate.

Recommended validation:

- Greedy validation with `max_new_tokens=1024` remains the primary score.
- Also report sampled pass@k and failure mix to detect distribution collapse.
- Track `completion_len_mean`, `completion_len_p95`, truncation rate,
  `reward_mean`, `reward_std`, and zero-std group rate.
