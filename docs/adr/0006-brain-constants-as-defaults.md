# 6. Brain constants, converted per thought, as defaults

Date: 2026-09-30 · Status: Accepted

## Context

The mechanisms come from a digital brain whose constants were found by
experiment there (fatigue gain, fade factor, capture threshold...). The brain
counts time in ticks; the harness counts it in thoughts. Tuning all parameters
for the harness before measuring anything would hide the origin of the method
and invite overfitting.

## Decision

- Every default is the brain's constant, converted per thought with
  `TICKS_PER_THOUGHT = 5.5` (the mean dwell of a thought) where the constant is
  per tick. The derivation is documented next to each parameter and in
  `docs/mechanisms.md`.
- The only deliberate departure: inhibition of return lasts the whole run by
  default, because a bounded run gains nothing from re-thinking an idea.
- Calibration happens on development tasks only (roadmap, Phase 1), and changed
  defaults are recorded in a new ADR.

## Consequences

- The method can be described faithfully as a transplant of the brain's thought
  stream, and ablations measure each mechanism at its native setting.
- Some defaults may be far from optimal for language models until calibrated.
