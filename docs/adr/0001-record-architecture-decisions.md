# 1. Record architecture decisions

Date: 2026-09-30 · Status: Accepted

## Context

widethink is a research project that will be read by people who were not there
when it was designed: reviewers of a paper, contributors, researchers deciding
whether to trust a result. Many choices (how budgets are counted, how parallel
streams interleave) directly affect the validity of experiments.

## Decision

Record every significant architectural or methodological decision as an ADR in
`docs/adr`, in the same pull request as the change.

## Consequences

The reasoning behind the design stays reviewable. Changing a recorded decision
requires writing down why, which is the point.
