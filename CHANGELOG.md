# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) (while in `0.x`, minor versions may
break the API).

## [Unreleased]

### Added

- Harness engine (`Thinker`) with deterministic rounds of parallel thought
  streams over a shared tree, a baseline anchor and a synthesis that states every
  deviation from the standard solution and the questions for the user.
- Mechanisms transplanted from the digital brain's thought stream: lottery
  selection, satiation, inhibition of return, fading, value pull, capture by
  surprise, emergence cued by the user's context or recalled from memory.
- `ThinkConfig` with every default derived from the brain, and `without()` for
  ablation studies.
- Providers: Claude via the Anthropic SDK (structured output, prompt caching,
  server-side fallback), OpenAI and OpenAI-compatible servers, a scripted model,
  and record/replay of model calls.
- Embedders: dependency-free hashing (default), OpenAI-compatible,
  sentence-transformers.
- Equal-budget accounting: normalized usage across providers, failed calls
  billed, synthesis reserve, cost reports.
- Memory of findings across sessions with offline consolidation ("sleep").
- Hidden Requirements benchmark: task schema with controls and canary, five
  exemplar tasks, solvers (direct, reasoning effort, broad prompt, best-of-N,
  widethink), LLM judge, resumable runner, report with bootstrap intervals.
- `widethink` command line: `think`, `render`, `bench validate|run|report|show`.
- Documentation: architecture, mechanisms, benchmark methodology, related work,
  architecture decision records, roadmap.
- DeepSeek provider (`DeepSeekLLM`, `--provider deepseek`): `deepseek-flash` by
  default, thinking mode off for harness steps and on for the reasoning baseline,
  JSON mode with the schema and a generated example in the prompt.
- Retries of transient provider failures (`TransientLLMError`); `content_filter`
  finishes are refusals; cache hits reported by DeepSeek are counted.
- `bench run --save-trees` keeps the full result of every widethink run;
  `bench show` prints verdicts and answers per task; `--price-*` options add cost
  to `think`, `bench run` and `bench report`.
- The command line loads keys from a `.env` file in the current directory.
- Guide to testing on real models: `docs/experiments.md`.

### Changed

- Baseline solvers answer in plain text instead of JSON.
- The benchmark runner records any failure of a job instead of stopping, and
  failed jobs are run again when a run is resumed.
