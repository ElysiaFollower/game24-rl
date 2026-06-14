# Contributing

This repository follows a lightweight version of the Google Python Style Guide.

## Python Style

- Use clear module, function, and variable names.
- Prefer small functions with explicit inputs and outputs.
- Use Google-style docstrings for public functions and classes.
- Keep comments sparse and useful; explain non-obvious reasoning, not syntax.
- Avoid hidden global state in solver, verifier, data generation, and evaluation code.

## Testing

- Add unit tests for solver, verifier, data generation, and reward behavior.
- Any verifier bug fix must include a regression test.
- Training scripts may be integration-tested with tiny fixtures rather than real model runs.

## Reproducibility

- Every reported score must name its split, model checkpoint, decoding settings, and verifier version.
- Generated data should be reproducible from a seed and manifest.
- Do not compare baselines without stating their model, split, answer format, and verifier assumptions.

## Formatting

Use `ruff` for linting and formatting once dependencies are installed:

```bash
ruff check .
ruff format .
pytest
```
