"""Capture by the unexpected.

Brain (``ThoughtStream.step``): surprise is judged against the *habitual*
background, an exponential moving average, because "surprise in general" is
always high. A spike of at least ``CAPTURE = 0.2`` above it hijacks the stream,
but not more often than once in ``CAPTURE_GAP = 15`` ticks. Without the
baseline the stream was captured on 166 of 168 emergences and never went
anywhere.

Harness: each step reports how surprising its findings were (a contradiction
with the standard approach, a surprising fact in the code, a failed check);
a spike makes the surprise itself the next thought.
"""

from __future__ import annotations

from widethink.config import CaptureConfig


class SurpriseMonitor:
    """Tracks the surprise baseline of one stream and decides on captures."""

    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self.baseline = 0.0
        self.last_capture: int | None = None
        self.captures = 0

    def observe(self, level: float, step: int) -> bool:
        """Feed the surprise of one thought; return ``True`` if it captures attention."""
        level = min(1.0, max(0.0, level))
        spike = level - self.baseline
        self.baseline += self.cfg.baseline_rate * (level - self.baseline)
        if not self.cfg.enabled or spike < self.cfg.threshold:
            return False
        if self.last_capture is not None and step - self.last_capture < self.cfg.refractory:
            return False
        self.last_capture = step
        self.captures += 1
        return True
