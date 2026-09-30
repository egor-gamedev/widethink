"""The outcome of a run: the answer, why it deviates from the template, and the full trace."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from widethink.budget import UsageSummary
from widethink.config import ThinkConfig
from widethink.events import Event
from widethink.tree import ThoughtNode, ThoughtTree


class Baseline(BaseModel):
    """The standard answer the model gives without widening."""

    model_config = ConfigDict(frozen=True)

    approach: str
    key_decisions: list[str]
    answer: str


class Deviation(BaseModel):
    """One place where the answer departs from the standard solution, and why."""

    model_config = ConfigDict(frozen=True)

    standard: str
    instead: str
    because: str
    evidence_nodes: list[str] = Field(default_factory=list)


class Question(BaseModel):
    """Something only the user can answer, and how it affects the solution."""

    model_config = ConfigDict(frozen=True)

    question: str
    why_it_matters: str = ""
    node_id: str | None = None


class RunMeta(BaseModel):
    """Everything needed to reproduce or attribute a run."""

    model_config = ConfigDict(frozen=True)

    widethink_version: str
    prompts_version: str
    llm: str
    critic: str
    embedder: str
    seed: int
    started_at: str
    duration_s: float
    stop_reason: str | None


class ThinkResult(BaseModel):
    """What :meth:`Thinker.think` returns."""

    task: str
    answer: str
    deviations: list[Deviation]
    questions: list[Question]
    confidence: Literal["low", "medium", "high"] | None
    baseline: Baseline | None
    nodes: list[ThoughtNode]
    stats: dict[str, int]
    usage: UsageSummary
    events: list[Event]
    config: ThinkConfig
    meta: RunMeta
    warnings: list[str] = Field(default_factory=list)

    def tree(self) -> ThoughtTree:
        """The thought tree, rebuilt for traversal and rendering."""
        return ThoughtTree.from_nodes(self.nodes)

    def render(self, fmt: Literal["text", "mermaid"] = "text", *, ascii_only: bool = False) -> str:
        """Render the thought tree as indented text or as a Mermaid flowchart."""
        from widethink.render import render_mermaid, render_text

        if fmt == "mermaid":
            return render_mermaid(self.tree())
        return render_text(self.tree(), ascii_only=ascii_only)

    def save(self, path: str | Path) -> Path:
        """Write the result as JSON (UTF-8)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> ThinkResult:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
