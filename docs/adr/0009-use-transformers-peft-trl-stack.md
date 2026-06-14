# Use Transformers, PEFT, and TRL as the main training stack

We will build the repository around Transformers, PEFT, and TRL rather than using LLaMA-Factory or Unsloth as the primary training entrypoint. This keeps data generation, verifier behavior, evaluation, and later GRPO rewards under repository control while still using standard libraries for model loading, LoRA SFT, and GRPO.
