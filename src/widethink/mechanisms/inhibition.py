"""Inhibition of return: what was just thought about is not thought about again.

Brain (``ThoughtStream._recent``): a neuron visited in the last
``RETURN_BLOCK = 60`` ticks cannot become the focus. Harness: ideas considered
during the window are shown to the model as "already considered", and
proposals that duplicate them are pruned instead of entering the tree.
"""

from __future__ import annotations


class InhibitionOfReturn:
    """Remembers when each idea was thought about, in thought steps."""

    def __init__(self, window: int | None = None, *, enabled: bool = True) -> None:
        if window is not None and window < 1:
            raise ValueError("window must be positive or None")
        self.window = window
        self.enabled = enabled
        self._visited: dict[str, int] = {}

    def visit(self, key: str, step: int) -> None:
        """Record that ``key`` was thought about at ``step``."""
        self._visited.pop(key, None)  # keep insertion order == recency order
        self._visited[key] = step

    def was_visited(self, key: str) -> bool:
        return key in self._visited

    def is_recent(self, key: str, step: int) -> bool:
        """Whether ``key`` still blocks its duplicates at ``step``."""
        if not self.enabled:
            return False
        visited_at = self._visited.get(key)
        if visited_at is None:
            return False
        return self.window is None or step - visited_at < self.window

    def recent(self, step: int) -> list[str]:
        """Keys that still inhibit at ``step``, oldest first."""
        if not self.enabled:
            return []
        return [key for key in self._visited if self.is_recent(key, step)]

    def visited(self) -> list[str]:
        """All keys ever visited, oldest first."""
        return list(self._visited)
