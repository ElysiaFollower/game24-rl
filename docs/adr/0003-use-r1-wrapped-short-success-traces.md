# Use R1-wrapped short success traces for the first SFT milestone

We will use short success traces as the first SFT reasoning content and wrap them in the course-required `<think>...</think><answer>...</answer>` interface. Rollback traces remain a useful idea from the LLM4Game24 reference baseline, but they are deferred to a later ablation because the first fallback route should minimize output length, truncation risk, and avoid teaching the model to emit long exploratory logs before a strong visible score is secured.
