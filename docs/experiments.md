# Running experiments with real models

A practical guide to Phase 1 of the [roadmap](../ROADMAP.md): make the harness
work on a real model, look at what it does, and compare it with plain answers.
The examples use DeepSeek, which is cheap enough to iterate freely; every command
works the same with `--provider anthropic` or `--provider openai`.

## 1. Setup

```bash
pip install -e ".[dev]"          # includes the OpenAI SDK used for DeepSeek
cp .env.example .env             # then put your key into .env: DEEPSEEK_API_KEY=sk-...
```

`.env` is git-ignored; the `widethink` command loads it from the current
directory (variables already set in the environment win). Never commit a key.

**Models.** `deepseek-flash` is the default: fast and cheap, right for the many
short steps of the harness. `deepseek-v4-pro` is stronger: use it as the judge
(the default for `--judge-provider deepseek`) and as a strong baseline.
DeepSeek's thinking mode is off for the harness; `--thinking` (or the
`direct-high` solver) turns it on.

**Embeddings.** DeepSeek has no embeddings endpoint. The default hashing embedder
needs nothing but only sees surface form; for real experiments install a local
one and pass `--embedder local`:

```bash
pip install -e ".[local]"        # sentence-transformers (downloads a small model once)
```

## 2. Smoke test: one task, direct answer vs widethink

```bash
widethink bench run --tasks benchmark/tasks/auth-offline-field-app.yaml --solver direct --solver widethink --provider deepseek --judge-provider deepseek --budget 150000 --out results/smoke.jsonl --save-trees results/trees
```

Then look at it:

```bash
widethink bench show results/smoke.jsonl --answers
widethink render results/trees/auth-offline-field-app__widethink__seed0.json
```

`bench show` prints, per requirement, `N` (noticed), `A` (addressed), `Q`
(asked the user) and the judge's reason, plus invented requirements. The tree
shows *why*: which context detail cued which branch, what captured attention,
which branches got boring, what was pruned as a repeat.

What to check first:

- **No failed thoughts** (`failed` in the tree, `failed_calls` in the usage). If
  JSON mode keeps failing, report it with the recording.
- **Branches start from the context** (`cued` nodes) and find evidence.
- **The standard solution is not re-derived** (its decisions are marked standard;
  near-duplicates are pruned).
- **Questions** appear where the context cannot settle something.

## 3. Your own project

```bash
widethink think "Add JWT-based authentication to our API" -c path/to/project --provider deepseek --budget 150000 --record runs/jwt.recording.jsonl --out runs/jwt.json
```

`--record` stores every model call, so the run can be replayed for free with
`--replay runs/jwt.recording.jsonl` (same seed, same config) — useful when you
change rendering or analysis code, or to show someone exactly what happened.

## 4. Mini-benchmark

All exemplar tasks, four solvers, three repeats, with the cost report:

```bash
widethink bench run --solver direct --solver broad --solver best-of-n --solver widethink --provider deepseek --judge-provider deepseek --budget 150000 --repeats 3 --out results/mini.jsonl --save-trees results/trees --price-input 0.30 --price-output 1.20 --price-cache-read 0.006
widethink bench report results/mini.jsonl --price-input 0.30 --price-output 1.20 --price-cache-read 0.006
```

The prices are DeepSeek's peak rates for `deepseek-flash` in USD per million
tokens as published in September 2026 (off-peak is half); check the current
pricing page. An interrupted run resumes from the output file, and failed jobs
are retried on the next run.

**Reading the numbers.** Five tasks are an anecdote, not evidence: use them to
find failure modes and to tune on, not to claim anything. The judge here is also
a DeepSeek model, so it may favour DeepSeek-style answers; that is acceptable
while developing and not acceptable in the paper (see
[benchmark.md](benchmark.md#judge-validation)).

## 5. Weak model with the harness vs strong model without

A question labs care about: does thinking widely let a cheap model match an
expensive one?

```bash
widethink bench run --solver widethink --provider deepseek --model deepseek-flash --judge-provider deepseek --budget 150000 --repeats 3 --out results/flash-wide.jsonl
widethink bench run --solver direct --solver direct-high --provider deepseek --model deepseek-v4-pro --judge-provider deepseek --budget 150000 --repeats 3 --out results/pro-direct.jsonl
```

Compare recall *and* cost in the two reports.

## 6. Ablations

Switch one mechanism off and compare trees and answers on the same task and
seed:

```bash
widethink think "..." -c path/to/project --provider deepseek --seed 1 --out runs/full.json
widethink think "..." -c path/to/project --provider deepseek --seed 1 --without cueing --out runs/no-cueing.json
```

Names: `baseline`, `capture`, `cueing`, `fade`, `fatigue`, `grounding`,
`inhibition`, `lottery`, `parallel`, `value`.

## Cost at a glance

A widethink run of 24 thoughts typically uses 100–150 thousand tokens, most of
them input; with the system prompt served from DeepSeek's cache that is a few US
cents on `deepseek-flash`. The mini-benchmark above (5 tasks × 4 solvers × 3
repeats, plus judging) stays around one or two dollars. Measure your own with the
price flags — the numbers depend on the size of the context.
