# Use explicit SFT success gates

The first SFT checkpoint must pass explicit gates before GRPO starts: verifier and solver tests must pass, format following should be high, and in-distribution validation solve rate should reach at least 70% to count as a visible fallback result. If SFT v1 scores below 50%, we treat it as a likely pipeline defect in data generation, prompt formatting, label masking, truncation, or evaluation rather than proceeding to GRPO.
