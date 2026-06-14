# Use a single answer contract

The repository's verifier and reward path will evaluate only the arithmetic expression inside `<answer>...</answer>` for our models. We will not add main-path compatibility for external baseline formats such as LLM4Game24's `expression:` line, because the project needs a narrow, maintainable output contract for SFT, evaluation, and GRPO rewards rather than proving scores on another repository's format.
