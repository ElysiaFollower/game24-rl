# Prioritize the SFT loop before GRPO implementation

The repository will reserve space for GRPO concepts and configuration, but the first implementation milestone is the solver, verifier, data generation, SFT training, evaluation, and failure analysis loop. GRPO implementation and tuning will start only after the SFT warm start has produced a verified visible score, because GRPO depends on the same verifier and prompt contract and would make upstream bugs harder to diagnose.
