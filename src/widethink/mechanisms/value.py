"""Pull towards valuable ideas (brain milestone M3).

Brain (``ThoughtStream._pull``): if the brain knows what the content is worth,
a thought is drawn to it with weight ``exp(VALUE_PULL * value)``, the value
clipped to ``[-0.5, 1.0]``. Combined with the squared lottery weights this
gives about 20:1 in favour of good over bad at equal links: worse ideas come
up less often, but not never (``k = 2`` gave 600:1 and bad ideas were never
thought, which the brain explicitly rejects).
"""

from __future__ import annotations

import math

from widethink.config import ValueConfig


def value_pull(value: float | None, cfg: ValueConfig) -> float:
    """Multiplier on a candidate's score for the given value (1 when unknown or disabled)."""
    if value is None or not cfg.enabled:
        return 1.0
    return math.exp(cfg.pull * min(cfg.ceiling, max(cfg.floor, value)))
