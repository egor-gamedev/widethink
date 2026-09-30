"""Choosing the next thought: a weighted lottery, not argmax.

Brain (``ThoughtStream._associate``): every neighbour of the focus is scored as
``link * (1 - fatigue) * pull(value)`` (x1.2 for memory traces) and the next
focus is drawn with weights ``score ** 2`` - a stronger link wins more often,
"but not always the strongest: thoughts differ". Squaring sharpens the draw
without making it deterministic.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: Scores at or below this are treated as zero (brain: ``score > 1e-3``).
SCORE_FLOOR = 1e-3


@dataclass(frozen=True, slots=True)
class Candidate:
    """A possible next thought, as the selection sees it."""

    key: str
    link: float
    """Association strength with the current focus, 0..1."""
    redundancy: float = 0.0
    """How much it repeats already-considered ideas, 0..1 (brain: candidate fatigue)."""
    value: float | None = None
    """Critic's estimate of how much the idea matters to this user, -1..1."""
    grounded: bool = False
    """Whether the idea is anchored in a concrete item of the user's context."""


def score(
    candidate: Candidate,
    *,
    pull: Callable[[float | None], float],
    grounded_bonus: float = 1.2,
) -> float:
    """Selection score of one candidate (before squaring)."""
    link = min(1.0, max(0.0, candidate.link))
    freshness = 1.0 - min(1.0, max(0.0, candidate.redundancy))
    bonus = grounded_bonus if candidate.grounded else 1.0
    return link * freshness * pull(candidate.value) * bonus


def draw(
    candidates: Sequence[Candidate],
    rng: random.Random,
    *,
    pull: Callable[[float | None], float],
    grounded_bonus: float = 1.2,
    square: bool = True,
    greedy: bool = False,
) -> Candidate | None:
    """Draw the next thought, or ``None`` if no candidate is worth thinking about."""
    scored = [
        (candidate, s)
        for candidate in candidates
        if (s := score(candidate, pull=pull, grounded_bonus=grounded_bonus)) > SCORE_FLOOR
    ]
    if not scored:
        return None
    if greedy:
        return max(scored, key=lambda pair: pair[1])[0]
    weights = [s * s if square else s for _, s in scored]
    return rng.choices([candidate for candidate, _ in scored], weights=weights, k=1)[0]
