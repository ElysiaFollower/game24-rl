# Game24 RL

This context defines the language for a course project that trains and evaluates language-model-based solvers for the 24-point game.

## Language

**High-score objective**:
The project goal of obtaining the strongest reliable course result within the available time.
_Avoid_: Pure research objective, idea-first objective

**Fallback strong result**:
A reproducible method expected to produce a competitive visible score even if later RL exploration fails.
_Avoid_: Backup only, throwaway baseline

**Exploration route**:
An experimental extension attempted after the fallback strong result is secured.
_Avoid_: Main promise, required path

**Baseline**:
A result or method used for comparison under an explicitly stated evaluation setting.
_Avoid_: Competitor, reference without split

**In-distribution split**:
A train, validation, or test split drawn from the same 1..13 solvable 24-point multiset space, with no multiset overlap across splits.
_Avoid_: Random row split, overlapping puzzle split

**Out-of-distribution split**:
An evaluation split that differs from the primary training distribution, such as hard ToT puzzles, Countdown target puzzles, or unsolvable puzzles.
_Avoid_: Regular test split, leaked heldout

**Reference baseline**:
A baseline whose ideas and scores are studied for comparison, without treating its implementation as the project target.
_Avoid_: Target method, code template

**Owned implementation**:
A project implementation written and maintained under this repository's control, even when informed by external baselines.
_Avoid_: Forked copy, black-box reuse

**Target method**:
The method this project ultimately claims as its own best approach under the chosen evaluation setting.
_Avoid_: Baseline, external reference

**Short success trace**:
A concise reasoning trace that contains only operations on a successful path to the final 24-point expression.
_Avoid_: Full search tree, rollback-heavy trace

**Rollback trace**:
A reasoning trace that may include explicit rollback steps before reaching a final valid expression.
_Avoid_: Long CoT, exhaustive search log

**Answer expression**:
The final arithmetic expression claimed to solve a puzzle.
_Avoid_: Final line, output string

**Answer contract**:
The rule that only the expression inside `<answer>...</answer>` is evaluated for this repository's models.
_Avoid_: Multiple accepted final-answer formats, external baseline format

**R1 wrapper**:
The required outer response structure with a reasoning section and final answer section.
_Avoid_: Raw trace, plain expression only

**Visible score**:
The reported task success rate that can be shown in the course report or presentation.
_Avoid_: Internal reward, training reward

**SFT success gate**:
The metric threshold used to decide whether the supervised warm start is ready for GRPO.
_Avoid_: Subjective good enough, training loss only

**First-pass SFT set**:
The initial short-success-trace SFT dataset sized for fast result discovery, not the maximum planned dataset.
_Avoid_: Full training corpus, final data scale

**SFT warm start**:
A supervised training stage used to establish format following and basic solver imitation before reinforcement learning.
_Avoid_: Final optimization stage, main research endpoint

**GRPO frontier stage**:
The reinforcement-learning stage intended to push beyond the supervised warm start using verifier rewards.
_Avoid_: Optional afterthought, pure baseline

## Relationships

- The **High-score objective** takes priority over any **Exploration route**.
- A **Fallback strong result** must be established before relying on an **Exploration route**.
- Every **Baseline** must state its evaluation setting before its **Visible score** is compared.
- **LLM4Game24** is a **Reference baseline**, not the **Target method**.
- The **Target method** must be an **Owned implementation**.
- A **Short success trace** is the default SFT v1 reasoning content inside the **R1 wrapper**.
- A **Rollback trace** is an **Exploration route**, not part of the first **Fallback strong result**.
- An **Answer expression** is the only content evaluated as the final answer.
- The **Answer contract** is intentionally narrower than external baseline formats.
- An **In-distribution split** must isolate puzzles by multiset, not by generated trace row.
- An **Out-of-distribution split** is used only for evaluation unless a later experiment explicitly declares otherwise.
- The **First-pass SFT set** is a milestone dataset; larger trace counts remain a valid later scaling path.
- The **SFT warm start** prepares the model for the **GRPO frontier stage**.
- The **GRPO frontier stage** is the main route for trying to exceed the fallback supervised result.
- The **SFT success gate** determines whether to proceed to the **GRPO frontier stage** or debug the pipeline.

## Example dialogue

> **Dev:** "Should we start from pure GRPO because it better matches the RL idea?"
> **Domain expert:** "No. The **High-score objective** comes first; secure a **Fallback strong result**, then use GRPO as an **Exploration route**."

> **Dev:** "LLM4Game24 reports a strong score; should we copy its code?"
> **Domain expert:** "No. Treat it as a **Reference baseline** and build an **Owned implementation** we can control and extend."

> **Dev:** "Should the first SFT milestone include rollback examples?"
> **Domain expert:** "No. Use a **Short success trace** inside the **R1 wrapper** first; evaluate **Rollback trace** later as an **Exploration route**."

> **Dev:** "Can we train on every solvable 24-point puzzle and still report a heldout score?"
> **Domain expert:** "No. The **In-distribution split** must have multiset-level separation, and **Out-of-distribution split** results should be reported separately."

> **Dev:** "Should our verifier accept LLM4Game24's `expression:` output as a normal model answer?"
> **Domain expert:** "No. The **Answer contract** only evaluates `<answer>...</answer>` for this repository's models."

> **Dev:** "Does the first 10k-trace SFT dataset represent all useful training data?"
> **Domain expert:** "No. It is the **First-pass SFT set** for fast result discovery; data scaling remains an expected improvement path."

> **Dev:** "Is SFT the final method?"
> **Domain expert:** "No. SFT is the **SFT warm start**; the **GRPO frontier stage** is where we try to push the limit."

> **Dev:** "If SFT v1 scores below 50%, should we still launch GRPO?"
> **Domain expert:** "No. Failing the **SFT success gate** at that level means debug the pipeline first."

## Flagged ambiguities

- "Our idea" was resolved to mean an **Exploration route**, not the primary success criterion.
- "LLM4Game24 route" was resolved to mean **Reference baseline**, not **Target method**.
