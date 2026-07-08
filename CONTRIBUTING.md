# Contributing

Thanks for your interest in this project. It is the reference implementation for
the paper *Trust-Aware Query Optimization for AI-Native Data Systems*, so the
priorities are correctness, reproducibility, and a small, auditable codebase.

## Getting started

```bash
git clone https://github.com/jorge-martinez-gil/trust-aware.git
cd trust-aware
pip install -e .[federated]
```

Python 3.10 or newer is required. The core optimizer has no third-party runtime
dependencies; only the federated retrieval study needs numpy, scipy, and
matplotlib.

## Running the tests

Please make sure the full suite passes before opening a pull request:

```bash
python -m unittest discover -s tests -v
```

The same suite runs in CI on Python 3.10, 3.11, and 3.12.

## Guidelines

- Keep the core optimizer (`trust_aware/`, excluding `trust_aware/federated/`)
  free of third-party runtime dependencies.
- Preserve determinism: experiments and benchmarks must produce identical output
  for a given code revision and seed. Add or update tests when behavior changes.
- Match the existing code style and keep changes focused and well-described.
- When you change optimizer logic, regenerate any affected figures, tables, or
  cached results so the artifact stays internally consistent.

## Reporting issues

Please open a GitHub issue with a minimal reproduction, the Python version, and
the exact command you ran. See `docs/reproducibility.md` for the canonical
commands.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
