# Use LoRA for the first SFT milestone

The first SFT milestone will use LoRA on `Qwen2.5-1.5B-Instruct` rather than full-parameter fine-tuning. LoRA is faster to iterate on a 4090-class setup, reduces checkpoint and recovery cost, and is sufficient for establishing the supervised warm start needed before the GRPO frontier stage; full fine-tuning remains a later upgrade only if LoRA becomes the observed bottleneck.
