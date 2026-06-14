# Experiment Log

## 2026-06-14 - SFT v1 throughput observation

- Remote SFT on AutoDL4090 is progressing normally.
- At roughly 384 / 2331 training steps, GPU utilization was about 21% with about 5.9GB VRAM used.
- CPU was active on the training process, so this does not look like a CPU stall.
- Current configuration is small-batch LoRA SFT with gradient accumulation, so low GPU utilization appears to be a configuration-level throughput limit rather than a failure.
- Follow-up optimization idea: profile batch size, packing, and input pipeline once the first result is secured.
