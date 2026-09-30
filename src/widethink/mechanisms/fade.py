"""Thoughts weaken with depth.

Brain (``ThoughtStream._advance``): each step deeper multiplies the strength of
the thought by ``FADE = 0.86``; when the next step would fall below
``MIN_STRENGTH = 0.22`` the branch is "exhausted", and the stack of thoughts
within thoughts never exceeds ``MAX_DEPTH = 7``. The branching mock-up
(``runtime/thinking.py``) also scales the step by ``min(1, 0.4 + link)``, so a
weakly linked tangent exhausts sooner.

Harness: strength also sets how much a thought may spend - weaker thoughts
propose fewer continuations and see less context.
"""

from __future__ import annotations

from widethink.config import ExpansionConfig, FadeConfig, ReinstatementConfig


def child_strength(parent: float, link: float, cfg: FadeConfig) -> float:
    """Strength of a thought reached from a parent of the given strength."""
    factor = cfg.factor
    if cfg.link_modulation is not None:
        factor *= min(1.0, cfg.link_modulation + max(0.0, link))
    return parent * factor


def exhausted(strength: float, cfg: FadeConfig) -> bool:
    """Whether one more step would be too weak to think."""
    return strength * cfg.factor < cfg.min_strength


def too_deep(depth: int, cfg: FadeConfig) -> bool:
    """Whether the stack of thoughts within thoughts is full."""
    return depth >= cfg.max_depth


def proposal_count(strength: float, cfg: ExpansionConfig) -> int:
    """How many continuations a thought of this strength proposes."""
    span = cfg.max_candidates - cfg.min_candidates
    return cfg.min_candidates + round(span * min(1.0, max(0.0, strength)))


def reinstated_items(strength: float, cfg: ReinstatementConfig) -> int:
    """How many context items accompany a thought of this strength."""
    return cfg.items if strength >= cfg.weak_below else cfg.weak_items
