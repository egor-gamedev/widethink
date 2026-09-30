"""The control flow of a stream of thought on hand-built trees, without any model."""

from __future__ import annotations

import random

import pytest

from widethink import Context, ContextItem, Finding, ThinkConfig
from widethink.budget import Ledger
from widethink.config import EmergenceConfig, FadeConfig, InhibitionConfig
from widethink.embeddings import HashingEmbedder
from widethink.events import EventLog
from widethink.mechanisms import InhibitionOfReturn
from widethink.semantic import SemanticIndex
from widethink.state import RunState, context_key, memory_key
from widethink.stream import ThoughtStream
from widethink.tree import ThoughtNode, ThoughtTree

EMBEDDER = HashingEmbedder()


def make_state(config: ThinkConfig | None = None, context: Context | None = None) -> RunState:
    cfg = config or ThinkConfig()
    state = RunState(
        task="Add JWT authentication",
        context=context or Context(),
        config=cfg,
        tree=ThoughtTree("Add JWT authentication"),
        index=SemanticIndex(EMBEDDER),
        inhibition=InhibitionOfReturn(cfg.inhibition.window, enabled=cfg.inhibition.enabled),
        ledger=Ledger(),
        log=EventLog(),
    )
    for item in state.context.items:
        state.index.put(context_key(item.id), EMBEDDER.vector(item.content))
        state.cue_weights[item.id] = item.salience
    return state


def add(state: RunState, parent: str, label: str, *, link: float = 0.9) -> ThoughtNode:
    node = state.tree.add(parent, kind="solution", origin="proposal", label=label, link=link)
    state.index.put(node.id, EMBEDDER.vector(label))
    return node


def thought(node: ThoughtNode | None, *, fatigue: float = 0.0) -> ThoughtNode:
    """Pretend the model has thought about ``node``."""
    assert node is not None
    node.status = "expanded"
    node.fatigue = fatigue
    return node


def start(state: RunState, seed: int = 0) -> ThoughtStream:
    stream = ThoughtStream(state, 0, random.Random(seed))
    stream.start(state.tree.root.id, "the task itself")
    thought(stream.decide())
    return stream


def test_goes_deeper_by_association_and_keeps_the_stack() -> None:
    state = make_state()
    stream = start(state)
    child = add(state, "n0", "Device-bound tokens")
    first = thought(stream.decide())
    assert first is child
    assert first.why == "association"
    assert stream.stack == [("n0", 1.0)]
    assert first.strength == pytest.approx(0.86)
    grandchild = add(state, child.id, "Remote wipe on reconnect")
    assert stream.decide() is grandchild
    assert len(stream.stack) == 2


def test_a_boring_branch_returns_to_the_parent_for_another_child() -> None:
    state = make_state()
    stream = start(state)
    one = add(state, "n0", "Offline grace period")
    two = add(state, "n0", "Revocation list sync")
    first = thought(stream.decide(), fatigue=0.9)  # bored at once
    second = thought(stream.decide())
    assert {first.id, second.id} == {one.id, two.id}
    assert first.closed_reason == "bored"
    assert second.why == "return (bored)"
    assert stream.stats["returns"] == 1
    assert stream.stack == [("n0", 1.0)]
    # the second branch has nowhere to go, the parent has no more children: nothing emerges
    assert stream.decide() is None
    assert second.closed_reason == "nowhere"
    assert stream.focus is None


def test_too_deep_returns_up() -> None:
    state = make_state(ThinkConfig(fade=FadeConfig(max_depth=1)))
    stream = start(state)
    deep = thought(stream.decide() if add(state, "n0", "Encrypted local storage") else None)
    add(state, deep.id, "Key rotation for local storage")
    sibling = add(state, "n0", "Audit log of token use")
    assert stream.decide() is sibling
    assert deep.closed_reason == "too_deep"


def test_an_exhausted_thought_returns_up() -> None:
    state = make_state(ThinkConfig(fade=FadeConfig(min_strength=0.8, link_modulation=None)))
    stream = start(state)
    weak = thought(stream.decide() if add(state, "n0", "Signed offline sessions") else None)
    add(state, weak.id, "Session expiry banner")
    sibling = add(state, "n0", "Per-technician keys")
    assert stream.decide() is sibling  # 0.86 * 0.86 < 0.8: the child would be too weak
    assert weak.closed_reason == "exhausted"


def test_duplicates_of_recent_thoughts_are_pruned_at_selection() -> None:
    state = make_state()
    stream = start(state)
    earlier = add(state, "n0", "Refresh token rotation")
    earlier.status = "expanded"
    state.inhibition.visit(earlier.id, 1)
    other = add(state, "n0", "Tablet pool check-out flow")
    duplicate = add(state, earlier.id, "Refresh-token rotation")
    stream.stack = [("n0", 1.0)]
    stream.focus, stream.strength = earlier.id, 0.86
    assert stream.decide() is other  # the duplicate child of `earlier` is skipped...
    assert duplicate.status == "pruned"  # ...and pruned
    assert duplicate.duplicate_of == earlier.id


def test_without_inhibition_duplicates_can_be_chosen() -> None:
    state = make_state(ThinkConfig(inhibition=InhibitionConfig(enabled=False)))
    stream = start(state)
    earlier = add(state, "n0", "Refresh token rotation")
    earlier.status = "expanded"
    state.inhibition.visit(earlier.id, 1)
    duplicate = add(state, earlier.id, "Refresh-token rotation")
    stream.focus, stream.strength = earlier.id, 0.86
    assert stream.decide() is duplicate


def test_capture_jumps_and_drops_the_stack() -> None:
    state = make_state()
    stream = start(state)
    child = thought(stream.decide() if add(state, "n0", "Short-lived tokens") else None)
    surprise = state.tree.add(child.id, kind="surprise", origin="captured", label="Offline!")
    stream.capture(surprise.id)
    assert stream.decide() is surprise
    assert surprise.why == "captured by a surprise"
    assert stream.stack == []
    assert surprise.strength == 1.0
    assert stream.stats["captured"] == 1


def test_a_context_detail_cues_a_grounded_thought() -> None:
    context = Context.of(
        ContextItem(id="README.md", title="README.md", content="Offline for days.", kind="doc")
    )
    state = make_state(context=context)
    state.tree.root.status = "thinking"  # stream 0 has already claimed the task itself
    stream = ThoughtStream(state, 1, random.Random(0))
    cued = stream.decide()
    assert cued is not None
    assert cued.origin == "cued"
    assert cued.grounding == ["README.md"]
    assert cued.label == 'What does "README.md" imply for this task?'
    assert state.index.get(cued.id) is state.index.get(context_key("README.md"))
    assert "README.md" not in state.cue_weights  # each detail cues once
    assert stream.stats["cued"] == 1


def test_unexplored_ideas_and_memories_resurface() -> None:
    config = ThinkConfig(emergence=EmergenceConfig(cued=0.0))
    state = make_state(config)
    idea = add(state, "n0", "Idea left open by another stream")
    state.tree.root.status = "expanded"
    stream = ThoughtStream(state, 0, random.Random(0))
    assert stream.decide() is idea
    assert idea.why == "an unexplored idea resurfaced"
    thought(idea)

    finding = Finding(id="f1", label="Past finding about offline auth", salience=0.9)
    state.recalled[finding.id] = finding
    state.index.put(memory_key("f1"), EMBEDDER.vector(finding.label))
    stream.focus = None
    recalled = stream.decide()
    assert recalled is not None
    assert recalled.origin == "recalled"
    assert recalled.label == finding.label
    assert state.recalled == {}
    assert stream.stats["recalled"] == 1


def test_a_stale_pending_start_is_ignored() -> None:
    state = make_state()
    stream = ThoughtStream(state, 0, random.Random(0))
    state.tree.root.status = "expanded"
    stream.start(state.tree.root.id, "the task itself")
    assert stream.decide() is None  # already thought by someone else, nothing else to do
