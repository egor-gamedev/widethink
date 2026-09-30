"""The thought tree: every idea considered, how it came up, and what became of it.

A node is an idea or a question formulated by the model (brain: a concept
neuron). Children are thoughts that arose *within* a thought. The tree records
provenance - which step proposed an idea, why a stream turned to it, why a
branch ended - so that the final answer can be traced back to the evidence that
made it deviate from the template.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NodeKind = Literal["task", "solution", "context", "risk", "surprise", "recalled"]
"""``context`` ideas ask what is special about this user; ``solution`` ideas are ways to
do the task; ``risk`` ideas are ways it could go wrong; ``surprise`` nodes are
unexpected findings that captured attention; ``recalled`` nodes come from memory."""

Origin = Literal["task", "baseline", "proposal", "cued", "captured", "recalled"]
"""How the node came into existence."""

NodeStatus = Literal["open", "thinking", "expanded", "failed", "pruned"]
"""``open``: proposed, not yet thought about. ``thinking``: claimed by a stream.
``expanded``: thought about. ``failed``: the model call failed. ``pruned``: never
to be thought about (a duplicate of an idea already in the tree)."""

Resolution = Literal["supported", "contradicted", "unknown", "not_applicable"]
"""What the user's context says about the idea."""

ClosedReason = Literal["bored", "exhausted", "too_deep", "nowhere"]
"""Why a stream stopped going deeper from a node (brain: надоело, выдохлась,
слишком глубоко, некуда)."""

_LABEL_LIMIT = 300


class Evidence(BaseModel):
    """A quote from the user's context that supports a claim."""

    model_config = ConfigDict(frozen=True)

    source: str
    """Id of the context item."""
    quote: str


class ThoughtNode(BaseModel):
    """One idea in the tree."""

    id: str
    parent_id: str | None
    kind: NodeKind
    origin: Origin
    label: str
    detail: str = ""
    link: float = 1.0
    """Association strength with the parent, as proposed (0..1)."""
    value: float | None = None
    """How much the idea matters to this user, as judged by the critic (-1..1)."""
    grounding: list[str] = Field(default_factory=list)
    """Context item ids the idea is anchored in."""
    depth: int = 0
    created_step: int = 0

    status: NodeStatus = "open"
    why: str | None = None
    """Why a stream turned to this idea: task, association, return, cued, recalled, captured."""
    step: int | None = None
    """Index of the thought in which the idea was thought about."""
    stream: int | None = None
    strength: float | None = None

    elaboration: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    resolution: Resolution | None = None
    question: str = ""
    """A question for the user when the context cannot resolve the idea."""
    surprise: float = 0.0
    fatigue: float = 0.0
    closed_reason: ClosedReason | None = None
    duplicate_of: str | None = None
    error: str | None = None
    children: list[str] = Field(default_factory=list)

    @property
    def thought(self) -> bool:
        """Whether the idea has been thought about (successfully or not)."""
        return self.status in ("expanded", "failed")

    def text(self) -> str:
        """Short text used for similarity comparisons."""
        return f"{self.label}. {self.detail}".strip() if self.detail else self.label


class ThoughtTree:
    """A mutable tree of :class:`ThoughtNode` with sequential, deterministic ids."""

    def __init__(self, task: str) -> None:
        self._nodes: dict[str, ThoughtNode] = {}
        self.root = self._insert(
            ThoughtNode(id="n0", parent_id=None, kind="task", origin="task", label=_clip(task))
        )

    # ------------------------------------------------------------------ building

    def add(
        self,
        parent_id: str,
        *,
        kind: NodeKind,
        origin: Origin,
        label: str,
        detail: str = "",
        link: float = 1.0,
        value: float | None = None,
        grounding: Iterable[str] = (),
        created_step: int = 0,
        status: NodeStatus = "open",
    ) -> ThoughtNode:
        """Add a child of ``parent_id`` and return it."""
        parent = self[parent_id]
        node = ThoughtNode(
            id=f"n{len(self._nodes)}",
            parent_id=parent_id,
            kind=kind,
            origin=origin,
            label=_clip(label),
            detail=detail.strip(),
            link=min(1.0, max(0.0, link)),
            value=None if value is None else min(1.0, max(-1.0, value)),
            grounding=list(dict.fromkeys(grounding)),
            depth=parent.depth + 1,
            created_step=created_step,
            status=status,
        )
        parent.children.append(node.id)
        return self._insert(node)

    def _insert(self, node: ThoughtNode) -> ThoughtNode:
        self._nodes[node.id] = node
        return node

    # ------------------------------------------------------------------ access

    def __getitem__(self, node_id: str) -> ThoughtNode:
        return self._nodes[node_id]

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def nodes(self) -> list[ThoughtNode]:
        """All nodes in creation order."""
        return list(self._nodes.values())

    def children(self, node_id: str) -> list[ThoughtNode]:
        return [self._nodes[child] for child in self._nodes[node_id].children]

    def parent(self, node_id: str) -> ThoughtNode | None:
        parent_id = self._nodes[node_id].parent_id
        return None if parent_id is None else self._nodes[parent_id]

    def path(self, node_id: str) -> list[ThoughtNode]:
        """Nodes from the root down to ``node_id`` inclusive."""
        chain: list[ThoughtNode] = []
        current: str | None = node_id
        while current is not None:
            node = self._nodes[current]
            chain.append(node)
            current = node.parent_id
        return chain[::-1]

    def walk(self, start: str | None = None) -> Iterator[ThoughtNode]:
        """Depth-first, pre-order traversal (iterative, so deep trees are safe)."""
        stack = [start or self.root.id]
        while stack:
            node = self._nodes[stack.pop()]
            yield node
            stack.extend(reversed(node.children))

    def frontier(self) -> list[ThoughtNode]:
        """Ideas proposed but not yet claimed by any stream."""
        return [node for node in self._nodes.values() if node.status == "open"]

    @property
    def max_depth(self) -> int:
        return max(node.depth for node in self._nodes.values())

    # ------------------------------------------------------------------ serialization

    @classmethod
    def from_nodes(cls, nodes: Iterable[ThoughtNode]) -> ThoughtTree:
        """Rebuild a tree from serialized nodes (root first)."""
        ordered = [node.model_copy(deep=True) for node in nodes]
        if not ordered or ordered[0].parent_id is not None:
            raise ValueError("the first node must be the root")
        tree = cls.__new__(cls)
        tree._nodes = {node.id: node for node in ordered}
        tree.root = ordered[0]
        return tree


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _LABEL_LIMIT else text[: _LABEL_LIMIT - 1] + "…"
