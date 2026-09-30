# 4. Equal-budget accounting as a first-class concern

Date: 2026-09-30 · Status: Accepted

## Context

A harness that makes many calls will beat a single answer simply by spending
more. Any claim is meaningless unless methods are compared at the same budget,
and the budget is counted the same way for all of them.

## Decision

- Usage is normalized across providers: `input_tokens` is every prompt token the
  model processed, cached ones included; cache reads and writes and reasoning
  tokens are reported as subsets.
- Every call is recorded in a `Ledger`, including failed and retried attempts,
  with its purpose and the model that actually answered (fallbacks are visible).
- A `Budget` limits total tokens, output tokens and calls; the engine stops
  thinking when the next round would not fit together with a reserve for the
  synthesis. Enforcement is per call, so a single call can overshoot its
  estimate; actual spend is always reported.
- The primary budget metric is total tokens (conservative for a harness that
  re-reads context); output tokens and cost are reported next to it.
- Embedding tokens are reported but not counted against the budget: they are
  orders of magnitude cheaper and not generative computation. Papers must state
  this.

## Consequences

- The harness is penalized for re-reading context, which motivates focused
  reinstatement and caching rather than hiding the cost.
- Baselines implemented in `widethink.bench` share the same accounting, so
  comparisons are like for like.
