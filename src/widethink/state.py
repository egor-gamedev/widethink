"""Mutable state shared by the streams of one run."""

from __future__ import annotations

from dataclasses import dataclass, field

from widethink.budget import Ledger
from widethink.config import ThinkConfig
from widethink.context import Context
from widethink.events import EventLog
from widethink.mechanisms.inhibition import InhibitionOfReturn
from widethink.memory.store import Finding
from widethink.result import Baseline
from widethink.semantic import SemanticIndex
from widethink.tree import ThoughtNode, ThoughtTree


def context_key(item_id: str) -> str:
    """Key of a context item in the semantic index (node ids never contain ':')."""
    return f"ctx:{item_id}"


def memory_key(finding_id: str) -> str:
    """Key of a recalled finding in the semantic index."""
    return f"mem:{finding_id}"


@dataclass
class RunState:
    """Everything the streams of one run share.

    The streams run in one event loop and mutate this state only between
    awaits, in a fixed order, so no locking is needed.
    """

    task: str
    context: Context
    config: ThinkConfig
    tree: ThoughtTree
    index: SemanticIndex
    inhibition: InhibitionOfReturn
    ledger: Ledger
    log: EventLog
    system_prompt: str = ""
    step: int = 0
    """Thoughts started so far, over all streams."""
    cue_weights: dict[str, float] = field(default_factory=dict)
    """Context items that may still cue a new thought, with their weights."""
    recalled: dict[str, Finding] = field(default_factory=dict)
    """Findings from past sessions that may still surface."""
    questions: list[str] = field(default_factory=list)
    baseline: Baseline | None = None
    stop_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def next_step(self) -> int:
        self.step += 1
        return self.step

    def stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
            self.log.emit("stopped", f"stopped: {reason}", step=self.step, data={"reason": reason})

    def thought_ids(self) -> list[str]:
        """Ideas already thought about or being thought about, oldest first."""
        return self.inhibition.visited()

    def recent_thought_ids(self) -> list[str]:
        """Ideas that still inhibit their duplicates."""
        return self.inhibition.recent(self.step)

    def comparable_ids(self) -> list[str]:
        """Every non-pruned idea with a vector: what new proposals are compared against."""
        return [
            node.id
            for node in self.tree.nodes()
            if node.status != "pruned" and node.id in self.index and node.kind != "task"
        ]

    def blocks(self, node: ThoughtNode) -> bool:
        """Whether a proposal duplicating ``node`` should be pruned.

        A duplicate of an idea not yet thought about is always merged into it;
        a duplicate of a thought is pruned while the thought is recent.
        """
        if not self.config.inhibition.enabled:
            return False
        if node.status == "open":
            return True
        return self.inhibition.is_recent(node.id, self.step)
