# 给队友的简短交接：repo-local split 实验

一句话版：

> 我们先在自己划分的 train / validation / test 上跑通了完整路线：
> `Qwen2.5-1.5B-Instruct -> full SFT final -> GRPO LoRA adapter -> decoding/eval`。
> 这批结果主要说明：SFT 已经很强，GRPO 有信号，token budget 对最终分数影响很大。

Hugging Face 留档：

- SFT-final：<https://huggingface.co/Prometheus17/game24-rl/tree/main/sft-final>
- handoff1 GRPO LoRA：<https://huggingface.co/Prometheus17/game24-rl/tree/main/grpo-lora-final>
- 仓库总览：<https://huggingface.co/Prometheus17/game24-rl/tree/main>

## 先说数据集

题目里提到两个数据集：`nlile/24-game` 和
`test-time-compute/game-of-24`。一开始看起来像是“一个训练，一个测试”。

但我们实际审计后发现：这两个数据集包含同一批 `1362` 道 24 点题目，只是排序和元数据不同。
而且当前下载到的 `nlile/24-game` 里，所有题目都是 solvable，没有不可解题。

所以探索阶段，为了先看清算法本身效果，我们没有直接使用 ToT 论文式 hard100 切法，
而是把这批标准 24 点题目划成：

- train：1089 题
- validation：136 题
- test：137 题

这份交接只讲这个 repo-local split 下的旧实验结果。

后面正在补另一个实验：按 Tree of Thoughts 论文 `arXiv:2305.10601` 的思路，
把 `test-time-compute/game-of-24` 的 indices `900-999` 当 hard100 测试题，
其余题训练。那个实验仍然有意义，但不属于这次交接。

## 补充说明(助教版)

我们问助教：是否必须严格按题目里暗示的数据集训练 / 测试方式做。助教回复：

> 哦哦你们可以自己切分下，只要说清楚数据是怎么切分的，保证test和train没有overlap，然后也可以把test分一下in-domain and out-of-domain test，只要说清楚你们是怎么划分这个逻辑的就行
>
> 没有严格地说一定要用什么数据训练，以及一定要用什么数据test
>
> 只要你们阐述出来的逻辑是合理的都可以

所以这份 repo-local split 结果是可用的：关键是说清楚切分逻辑、保证 train/test 无 overlap、
解释指标口径。official ToT-style hard100 实验属于补充对齐，不是否定这批旧结果。

## 我们做了什么

主线就是：

`Qwen2.5-1.5B-Instruct -> full SFT final -> GRPO LoRA adapter -> decoding/eval`

基座模型没有经过任何训练，直接在全量 1362 道题上测试。结果很低：
`16/1362 = 1.17%`，见 `docs/experiments/official_tot_results_20260618.md`。
这个分数可以作为“基座几乎不会做”的直观参考。

注意：base eval 是全量 1362 题、4096 token budget；下面的强 SFT baseline 是
repo-local validation 136 题、1024 token budget。两者都能说明训练前后差距很大，
但严格对比时要写清 eval set 和 token budget。

## 第一阶段：先拿到强 SFT baseline

看这个文档：

`docs/experiments/sft_full_finetune_search_trace_20260616.md`

它在说：我们用 rollback/search-trace 数据做 full-parameter SFT，把 validation 做到
`110/136 = 80.88%`。这已经是一个可用的强 SFT baseline。

关键点：

- 数据是 search / rollback trace，不是普通短答案。
- full fine-tuning 明显比之前短 LoRA probe 强。
- checkpoint 曲线从 `51/136`、`70/136`，最后到 `110/136`。
- 剩下 26 道失败基本不是算错，而是模型一直写搜索过程，1024 token 内没来得及输出 `<answer>`。

所以当时的判断是：SFT 路线成立。下一步问题不是“模型完全不会算”，而是“搜索过程太长，答案闭合太晚”。

## 第二阶段：判断 GRPO 值不值得做

看这个文档：

`docs/experiments/grpo_rollout_audit_20260616.md`

它在说：我们对 SFT checkpoint 做采样审计，发现很多 greedy 做错/没答完的题，
其实采样时能采到正确轨迹。

最关键数字：

- SFT greedy 失败题有 26 道。
- 对这 26 道题，每题采样 8 个输出。
- pass@8 能解出 `22/26`。
- `19/26` 道题有 mixed reward，也就是同一道题下有的输出对、有的输出错。

这说明 GRPO 有信号。因为 GRPO 需要的正是这种情况：同一个 prompt 下有好轨迹和坏轨迹，
模型才知道应该提高哪类输出概率。

## 第三阶段：实际做 GRPO

在 1024 token budget 下，GRPO 有可见提升：

- SFT 1024 validation：`110/136 = 80.88%`
- best GRPO 1024 validation：`116/136 = 85.29%`

所以不能说 GRPO 没用。它确实把一部分题推对了，而且基本保住了原来 SFT 会做的题。

但它没有彻底解决问题。失败仍然主要是 no-answer / answer-contract，也就是模型还在长搜索，
没及时闭合答案。

## 后来发现：token budget 是大因素

看这个文档：

`docs/experiments/direct_long_token_greedy_20260617.md`

它在说：我们把 eval 的 `max_new_tokens` 从 1024 放到 4096 后，结果明显变了。

SFT 自己在 4096 下就很强：

- SFT 1024 validation：`110/136 = 80.88%`
- SFT 4096 validation：`123/136 = 90.44%`
- SFT 4096 test：`128/137 = 93.43%`

GRPO 在同样 4096 下还有小幅提升：

- GRPO 4096 validation：`126/136 = 92.65%`
- GRPO 4096 test：`129/137 = 94.16%`

所以当前更准确的解释是：

> 90%+ 的主要跃升来自更长 token budget。GRPO 仍有同预算增益，但不大。

这不是坏消息。它说明很多失败题不是模型不会，而是 1024 下没来得及走到答案。

### 一个后续假设

当前 GRPO 主要是在 1024 token limit 下做出来的。在这个口径下，它从
`110/136` 提到 `116/136`，说明方法有信号。

但 4096 实验说明 token limit 是强混杂因素。因此一个合理猜想是：

> 现在这个 GRPO adapter 是按 1024 预算下的问题形态训练和选择出来的，
> 未必最适合 4096 这个最终展示口径。

这只是后续探索方向，不是已经证明的结论。

如果要继续补实验，建议方向是：

- 长 token budget 下重新做 GRPO
- 或做 closure-aware GRPO
- reward / logging 里重点看 answer closure
- 所有结果都同口径比较：SFT 4096 vs GRPO 4096

## 建议阅读顺序

1. `docs/experiments/sft_full_finetune_search_trace_20260616.md`
   先看强 SFT baseline 怎么来的。

2. `docs/experiments/grpo_rollout_audit_20260616.md`
   再看为什么我们认为 GRPO 值得做。

3. `docs/experiments/direct_long_token_greedy_20260617.md`
   最后看为什么 token budget 改变了结果解释。
