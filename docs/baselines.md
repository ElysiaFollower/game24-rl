# Reference Baselines

Scores in this document are reference points, not direct proof of superiority unless the model, split, answer format, and verifier are aligned.

## Tree of Thoughts

Source: [Tree of Thoughts](https://arxiv.org/abs/2305.10601)

Setting:

- Model: GPT-4
- Evaluation: 100 hard games from 4nums.com, indexed 901-1000
- Metric: success rate, valid equation equals 24 and uses each input number once
- Method: inference-time search, not model training

Reported scores:

| Method | Success |
|---|---:|
| IO prompt | 7.3% |
| CoT prompt | 4.0% |
| CoT-SC, k=100 | 9.0% |
| ToT, b=1 | 45% |
| ToT, b=5 | 74% |
| IO + Refine, k=10 | 27% |
| IO best-of-100 | 33% |
| CoT best-of-100 | 49% |

Takeaway: 24-point solving is a search problem; single-path CoT is weak even with GPT-4.

## LLM4Game24

Source: [LLM4Game24](https://github.com/LiaoMengqi/LLM4Game24)

Role in this project: **Reference baseline**, not target implementation.

Setting:

- Model: Qwen2.5-0.5B
- Dataset: 1,362 solvable 24-point instances
- Split: 1,262 train / 100 test
- Training: solver-generated long CoT / rollback traces, then GRPO
- Answer format: `reach 24! expression: ...`

Reported SFT scores:

| Training data | Test accuracy |
|---|---:|
| Short | 57% |
| Medium | 72% |
| Long | 50% |
| Format v1 | 66% |
| Format v2 | 84% |
| Format v3 | 74% |

Reported RL observation:

- GRPO accuracy rises early and peaks around step 30 near 80%, then degrades.

Local strict-verifier recount of saved outputs:

| Saved output | Strict success |
|---|---:|
| `format_v2.json` | 84/100 |
| `res_mt02_1.json` | 73/100 |
| `res_mt02_2.json` | 69/100 |
| `res_mt02_3.json` | 75/100 |
| `res_mt02_4.json` | 74/100 |
| `res_mt02_5.json` | 61/100 |

Takeaways:

- Solver-generated traces are a strong SFT baseline.
- Medium-length / format-v2-like traces are more reliable than long traces.
- GRPO can become unstable and may not beat the best SFT checkpoint.
- This repository should borrow the idea, not the code or output contract.

Comparability limits:

- Different model size.
- Different split.
- Different answer format.
- Their verifier/reward path uses Python `eval`; this repository will use an AST + `Fraction` verifier.
