# Use a strict AST and Fraction verifier

The verifier will parse only the repository's `<answer>...</answer>` contract, evaluate arithmetic through an AST allowlist, and compute values with `Fraction`; it will not use Python `eval` or regex-only validation. This verifier is the foundation for SFT evaluation, GRPO rewards, and failure analysis, so shortcut validation would risk misleading scores and expensive retraining after later fixes.
