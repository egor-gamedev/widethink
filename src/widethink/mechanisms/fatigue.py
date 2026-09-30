"""Satiation: a branch that keeps saying the same thing gets boring and closes.

Brain (``Neuron._update_fatigue``): each tick the same input pattern repeats,
the neuron's fatigue grows by 0.12 (up to 0.8) and raises its threshold; each
tick without repetition it recovers by 0.06. At fatigue 0.45 the thought
stream finds the focus "boring" and returns to the level above.

Harness: "the same input" becomes "ideas that are semantically close to ones
already in the tree". The repetition of one thought is measured on the ideas it
proposes, and fatigue accumulates along the branch.
"""

from __future__ import annotations

from collections.abc import Iterable

from widethink.config import FatigueConfig


def repetition(similarity: float, *, related: float, duplicate: float) -> float:
    """Map a cosine similarity to a repetition degree in ``[0, 1]``.

    At or below ``related`` the idea is new (0); at or above ``duplicate`` it is
    the same idea again (1); in between the degree grows linearly.
    """
    if duplicate <= related:
        raise ValueError("duplicate threshold must be above related threshold")
    return min(1.0, max(0.0, (similarity - related) / (duplicate - related)))


def mean_repetition(degrees: Iterable[float]) -> float:
    """Repetition of a whole thought: the mean over its proposals (0 if it proposed nothing)."""
    values = list(degrees)
    return sum(values) / len(values) if values else 0.0


def update_fatigue(previous: float, degree: float, cfg: FatigueConfig) -> float:
    """Fatigue after one more thought with the given repetition degree."""
    if not cfg.enabled:
        return 0.0
    fatigue = previous + cfg.gain * degree - cfg.recovery * (1.0 - degree)
    return min(cfg.max_fatigue, max(0.0, fatigue))


def satiated(fatigue: float, cfg: FatigueConfig) -> bool:
    """Whether the branch is boring enough to close."""
    return cfg.enabled and fatigue >= cfg.satiated
