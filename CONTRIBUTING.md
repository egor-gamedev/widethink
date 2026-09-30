# Contributing

Thank you for helping. The most valuable contributions right now are
**benchmark tasks** (see [docs/benchmark.md](docs/benchmark.md) for how to write
a good one), bug reports with a reproducing recording, and providers for more
models.

## Development setup

Python 3.11 or newer.

```bash
git clone https://github.com/egor-gamedev/widethink && cd widethink
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

With [uv](https://docs.astral.sh/uv/): `uv venv && uv pip install -e ".[dev]"`.

## Checks

Every pull request must pass what CI runs:

```bash
ruff format --check .          # formatting
ruff check .                   # lint
mypy                           # strict type checking of src/
pytest --cov                   # tests with coverage (at least 85%)
widethink bench validate       # benchmark task files
```

`pre-commit` runs formatting and linting on each commit. The tests need no
network and no API keys: models are replaced by `ScriptedLLM` (see the fake mind
in `tests/conftest.py`). Tests that call real APIs are marked `live` and run only
with `WIDETHINK_LIVE_TESTS=1`.

## Conventions

- **Style.** Ruff formatting, line length 100, type hints everywhere (`mypy
  --strict`), Google-style docstrings where a docstring helps. Keep the style of
  the surrounding code.
- **Mechanisms.** A mechanism is a pure function or small class in
  `src/widethink/mechanisms/`, with its origin in the brain documented next to it,
  a switch in `ThinkConfig`, an ablation name in `ABLATIONS`, and unit tests
  against the documented behaviour.
- **Prompts.** Any change to the text in `prompts.py` bumps `PROMPTS_VERSION`;
  results are only comparable under identical prompts.
- **Decisions.** A change to architecture or methodology comes with an ADR in
  `docs/adr/`.
- **Changelog.** Add a line under *Unreleased* in `CHANGELOG.md`.
- **Commits.** Imperative, present tense ("Add critic retries"), one logical
  change per commit.

## Benchmark tasks

1. Copy an existing file in `benchmark/tasks/`, keep the canary line.
2. Follow the checklist in [docs/benchmark.md](docs/benchmark.md#writing-a-good-task).
3. Run `widethink bench validate`.
4. In the pull request, say which models you checked the standard solution with.

Tasks are published under CC BY 4.0; by contributing one you agree to that.

## Reporting bugs

Open an issue with the version, the provider and model, the command or code, and
what you expected. If you can, attach a recording (`--record`) with private data
removed — it lets us replay your run exactly.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
