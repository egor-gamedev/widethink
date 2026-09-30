"""Where a new thought comes from when the current line of thought has ended.

Brain (``ThoughtStream._emerge``): with probability ``CUED = 0.5`` the new
thought is prompted by what is being perceived; otherwise a memory trace
surfaces with weight ``(0.05 + salience + 1 / (1 + age / 30)) * pull(value)``.

Harness: "what is perceived" is the user's context - a file, a sentence of the
request, a constraint. A cued thought asks what that detail implies for the
task, which is how branches about *this user's* specifics are born. Recall
draws from unexplored ideas of this session and from findings of past ones.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class Source(StrEnum):
    CONTEXT = "context"
    RECALL = "recall"


@dataclass(frozen=True, slots=True)
class Option:
    """Something that could surface as a new thought, with its weight."""

    key: str
    weight: float


def recall_weight(salience: float, age: float, pull: float, *, recency_scale: float) -> float:
    """Weight of a memory surfacing unprompted (brain formula)."""
    return (0.05 + max(0.0, salience) + 1.0 / (1.0 + max(0.0, age) / recency_scale)) * pull


def choose_emergence(
    rng: random.Random,
    *,
    cued_share: float,
    cues: Sequence[Option],
    recalls: Sequence[Option],
) -> tuple[Source, str] | None:
    """Pick the source and key of a new thought, or ``None`` if nothing can surface."""
    if cues and (not recalls or rng.random() < cued_share):
        return Source.CONTEXT, _weighted(rng, cues)
    if recalls:
        return Source.RECALL, _weighted(rng, recalls)
    return None


def _weighted(rng: random.Random, options: Sequence[Option]) -> str:
    weights = [max(0.0, option.weight) for option in options]
    if sum(weights) <= 0.0:
        return rng.choice(options).key
    return rng.choices([option.key for option in options], weights=weights, k=1)[0]
