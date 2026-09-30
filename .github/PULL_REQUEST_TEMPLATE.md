## What and why

<!-- What does this change do, and why is it needed? Link the issue if there is one. -->

## Checklist

- [ ] Tests cover the change; `pytest --cov` passes locally
- [ ] `ruff format --check .`, `ruff check .` and `mypy` pass
- [ ] `CHANGELOG.md` has a line under *Unreleased*
- [ ] Docs updated where behaviour changed
- [ ] If prompts changed: `PROMPTS_VERSION` bumped
- [ ] If a mechanism changed: documented origin, `ThinkConfig` switch, ablation name, unit tests
- [ ] If architecture or methodology changed: an ADR in `docs/adr/`
- [ ] If benchmark tasks changed: `widethink bench validate` passes; canary kept
