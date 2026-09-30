# 2. An external harness over a frozen model

Date: 2026-09-30 · Status: Accepted

## Context

Wide thinking can be built into a model by training (reinforcement learning on
reasoning traces) or imposed from outside by a program that decides what the
model thinks about. Training needs data, compute and access to model weights we
do not have, and it would make results depend on one model.

## Decision

widethink is a harness: the model is used as is through its API; the harness
owns the tree, the choice of the next thought and the stopping rules.

## Consequences

- Works with any model behind an API, closed or open, so claims can be tested
  across model families.
- Every decision of the harness is observable and can be ablated.
- Each thought is a separate call, so the context is re-read at every step; this
  costs input tokens and must be counted (ADR 0004) and mitigated (focused
  reinstatement, prompt caching).
- Labs would adopt the idea through training; harness traces are a possible
  source of such training data (roadmap, "Distillation").
