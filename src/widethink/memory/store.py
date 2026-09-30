"""Findings that outlive a session, and "sleep" that consolidates them.

Brain: a trace that is thought about counts as rehearsed and grows stronger;
during sleep the hippocampus replays traces without sensory input, and that is
when separate episodes get linked to each other (``DigitalBrain.sleep``).

Harness: non-standard findings of a run are stored with their embeddings.
In later runs, findings close to the new task can surface as thoughts of their
own (see ``EmergenceConfig.memory_recall``). :meth:`FindingStore.sleep` replays
the store offline: it merges duplicates, links related findings and lets
findings that are never used fade away. Feedback on whether a finding helped
(:meth:`FindingStore.reinforce`) changes how readily it surfaces.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from widethink.embeddings.base import Embedder, Vector
from widethink.embeddings.hashing import HashingEmbedder

#: Salience multiplier applied at each sleep to findings unused since the last one.
FADE_PER_SLEEP = 0.97
#: Findings below this salience are forgotten during sleep.
FORGET_BELOW = 0.05


class Finding(BaseModel):
    """A remembered, non-standard insight."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    label: str
    detail: str = ""
    task: str = ""
    kind: str = "context"
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    reward: float = 0.0
    uses: int = 0
    created: float = 0.0
    last_used: float = 0.0
    links: list[str] = Field(default_factory=list)
    vector: list[float] | None = Field(default=None, repr=False)

    def text(self) -> str:
        return f"{self.label}. {self.detail}".strip() if self.detail else self.label


class SleepReport(BaseModel):
    merged: int
    linked: int
    forgotten: int
    remaining: int


class FindingStore:
    """A small persistent store of findings (JSON Lines on disk, or in memory)."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        embedder: Embedder | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = None if path is None else Path(path)
        self.embedder: Embedder = embedder or HashingEmbedder()
        self._clock = clock
        self._last_sleep = 0.0
        self.findings: dict[str, Finding] = {}
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    finding = Finding.model_validate_json(line)
                    self.findings[finding.id] = finding

    def __len__(self) -> int:
        return len(self.findings)

    async def remember(self, findings: Sequence[Finding]) -> None:
        """Add findings (embedding them if needed) and persist."""
        now = self._clock()
        await self._embed([f for f in findings if f.vector is None])
        for finding in findings:
            if not finding.created:
                finding.created = now
            self.findings[finding.id] = finding
        self.save()

    async def recall(self, query: str, k: int = 5) -> list[Finding]:
        """The ``k`` findings most worth surfacing for ``query``.

        Relevance is similarity to the query; salience and reward make a
        finding surface more readily, as significance does for memory traces.
        """
        if not self.findings or k <= 0:
            return []
        batch = await self.embedder.embed([query])
        query_vector = batch.vectors[0]
        scored: list[tuple[float, Finding]] = []
        for finding in self.findings.values():
            similarity = float(_vector(finding) @ query_vector)
            if similarity <= self.embedder.related_threshold:
                continue
            weight = similarity * (0.5 + finding.salience) * (1.0 + 0.5 * np.tanh(finding.reward))
            scored.append((weight, finding))
        scored.sort(key=lambda pair: -pair[0])
        chosen = [finding for _, finding in scored[:k]]
        now = self._clock()
        for finding in chosen:
            finding.uses += 1
            finding.last_used = now
        if chosen:
            self.save()
        return chosen

    def reinforce(self, finding_id: str, reward: float) -> None:
        """Report whether a finding helped (+) or misled (-)."""
        finding = self.findings[finding_id]
        finding.reward += reward
        finding.salience = min(1.0, max(0.0, finding.salience + 0.2 * reward))
        self.save()

    async def sleep(self) -> SleepReport:
        """Offline consolidation: merge duplicates, link related findings, forget unused ones."""
        await self._embed([f for f in self.findings.values() if f.vector is None])
        merged = linked = forgotten = 0
        ordered = sorted(self.findings.values(), key=lambda f: (-f.salience, f.created, f.id))
        kept: list[Finding] = []
        for finding in ordered:
            twin = next(
                (k for k in kept if _similarity(k, finding) >= self.embedder.duplicate_threshold),
                None,
            )
            if twin is not None:
                twin.uses += finding.uses
                twin.reward += finding.reward
                twin.links = sorted(set(twin.links) | set(finding.links) - {twin.id})
                merged += 1
                continue
            kept.append(finding)
        for index, left in enumerate(kept):
            for right in kept[index + 1 :]:
                if _similarity(left, right) > self.embedder.related_threshold and (
                    right.id not in left.links
                ):
                    left.links.append(right.id)
                    right.links.append(left.id)
                    linked += 1
        survivors: dict[str, Finding] = {}
        for finding in kept:
            if finding.last_used <= self._last_sleep:
                finding.salience *= FADE_PER_SLEEP
            if finding.salience < FORGET_BELOW:
                forgotten += 1
                continue
            survivors[finding.id] = finding
        for finding in survivors.values():
            finding.links = [link for link in finding.links if link in survivors]
        self.findings = survivors
        self._last_sleep = self._clock()
        self.save()
        return SleepReport(
            merged=merged, linked=linked, forgotten=forgotten, remaining=len(self.findings)
        )

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for finding in self.findings.values():
                handle.write(json.dumps(finding.model_dump(), ensure_ascii=False) + "\n")
        temporary.replace(self.path)

    async def _embed(self, findings: Sequence[Finding]) -> None:
        if not findings:
            return
        batch = await self.embedder.embed([finding.text() for finding in findings])
        for finding, vector in zip(findings, batch.vectors, strict=True):
            finding.vector = [float(x) for x in vector]


def _vector(finding: Finding) -> Vector:
    if finding.vector is None:  # pragma: no cover - vectors are filled before use
        raise ValueError(f"finding {finding.id} has no vector")
    return np.asarray(finding.vector, dtype=np.float32)


def _similarity(left: Finding, right: Finding) -> float:
    return float(_vector(left) @ _vector(right))
