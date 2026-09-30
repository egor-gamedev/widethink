# 7. The benchmark lives in this repository for now

Date: 2026-09-30 · Status: Accepted

## Context

The Hidden Requirements benchmark may be as valuable as the harness, and
others should be able to use it without the harness. A separate repository
would decouple them, but while the task format and the judge are still changing,
two repositories double the coordination.

## Decision

- Task data lives in `benchmark/` (CC BY 4.0); the evaluation code in
  `widethink.bench` (Apache 2.0). The evaluation code depends on the harness only
  in `WideThinkSolver`; solvers, judge and runner work with any `Solver`.
- At v1.0 of the benchmark, the data is published on Zenodo and Hugging Face;
  the code may move to its own package if outside use warrants it.

## Consequences

- One pull request can change the format, the tasks and the tests together.
- Users of the benchmark install `widethink[bench]` for now.
