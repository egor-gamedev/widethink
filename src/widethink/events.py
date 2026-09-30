"""Events emitted while thinking: for live display, logging and later analysis."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("widethink")

EventType = Literal[
    "run_started",
    "baseline",
    "selected",
    "expanded",
    "failed",
    "pruned",
    "closed",
    "captured",
    "question",
    "stopped",
    "synthesized",
    "run_finished",
]


class Event(BaseModel):
    """Something that happened during a run."""

    model_config = ConfigDict(frozen=True)

    type: EventType
    message: str
    step: int | None = None
    stream: int | None = None
    node_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


Hook = Callable[[Event], None]


class EventLog:
    """Keeps every event of a run and forwards it to hooks.

    A failing hook is logged and skipped: observers must never break a run.
    """

    def __init__(self, hooks: Iterable[Hook] = ()) -> None:
        self.events: list[Event] = []
        self._hooks = list(hooks)

    def emit(self, type_: EventType, message: str, **fields: Any) -> Event:
        event = Event(type=type_, message=message, **fields)
        self.events.append(event)
        logger.debug("%s: %s", type_, message)
        for hook in self._hooks:
            try:
                hook(event)
            except Exception:
                logger.exception("event hook %r failed", hook)
        return event
