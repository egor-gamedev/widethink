from __future__ import annotations

from pathlib import Path

import pytest

from widethink import Finding, FindingStore


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def finding(label: str, salience: float = 0.5, **extra: object) -> Finding:
    return Finding(label=label, task="Add JWT auth", salience=salience, **extra)  # type: ignore[arg-type]


async def test_remember_and_recall_relevant_findings(tmp_path: Path) -> None:
    store = FindingStore(tmp_path / "memory.jsonl")
    await store.remember(
        [
            finding("Offline technicians need long-lived device tokens", 0.9),
            finding("Quarterly invoices are exported to CSV", 0.9),
        ]
    )
    recalled = await store.recall("tokens for offline technicians", k=5)
    assert [f.label for f in recalled] == ["Offline technicians need long-lived device tokens"]
    assert recalled[0].uses == 1
    assert await store.recall("anything", k=0) == []


async def test_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    store = FindingStore(path)
    await store.remember([finding("Revoke access within an hour")])
    reloaded = FindingStore(path)
    assert len(reloaded) == 1
    [item] = reloaded.findings.values()
    assert item.vector is not None
    assert item.created > 0


async def test_reinforcement_changes_salience() -> None:
    store = FindingStore()
    item = finding("Device binding", 0.5)
    await store.remember([item])
    store.reinforce(item.id, 1.0)
    assert store.findings[item.id].salience == pytest.approx(0.7)
    store.reinforce(item.id, -10.0)
    assert store.findings[item.id].salience == 0.0


async def test_sleep_merges_duplicates_links_related_and_forgets() -> None:
    clock = Clock()
    store = FindingStore(clock=clock)
    keep = finding("Device-bound refresh tokens", 0.9, uses=2)
    twin = finding("Device bound refresh tokens", 0.4, uses=3)
    related = finding("Rotate refresh tokens after each use", 0.6)  # similarity ~0.46
    faint = finding("Unrelated trivia about fonts", 0.04)
    await store.remember([keep, twin, related, faint])

    report = await store.sleep()

    assert report.merged == 1
    assert report.forgotten == 1
    assert report.linked >= 1
    assert report.remaining == 2
    assert twin.id not in store.findings
    assert store.findings[keep.id].uses == 5
    assert related.id in store.findings[keep.id].links
    assert keep.id in store.findings[related.id].links


async def test_unused_findings_fade_across_sleeps() -> None:
    clock = Clock()
    store = FindingStore(clock=clock)
    item = finding("Something", 0.5)
    await store.remember([item])
    await store.sleep()
    clock.now += 10
    await store.sleep()
    assert store.findings[item.id].salience == pytest.approx(0.5 * 0.97 * 0.97)
