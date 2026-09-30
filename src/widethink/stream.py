"""A stream of thought over the shared tree.

This is the digital brain's ``ThoughtStream`` moved from neurons to ideas. At
each step the stream decides one thing only - what to think about next:

* **deeper**, by association: a child of the current thought, drawn by lottery;
* **back up** when the branch ends - it got boring (satiation), exhausted (fade),
  too deep, or there is nowhere left to go - to the parent, taking another child;
* **anew** when there is nowhere up either: a thought cued by a detail of the
  user's context, or an unexplored idea (or a past finding) surfacing;
* **captured** by a surprise the previous thought ran into.

The model call that actually thinks the chosen thought is made by the engine;
the stream only decides. Decisions are synchronous and draw from the stream's
own random generator, so a run is reproducible for a given seed.
"""

from __future__ import annotations

import random
from collections import Counter

from widethink.mechanisms.capture import SurpriseMonitor
from widethink.mechanisms.emergence import Option, Source, choose_emergence, recall_weight
from widethink.mechanisms.fade import child_strength, exhausted, too_deep
from widethink.mechanisms.fatigue import repetition, satiated
from widethink.mechanisms.selection import Candidate, draw
from widethink.mechanisms.value import value_pull
from widethink.state import RunState, context_key, memory_key
from widethink.tree import ClosedReason, ThoughtNode


class ThoughtStream:
    """One line of thought. Several streams share a tree and see each other's thoughts."""

    def __init__(self, state: RunState, index: int, rng: random.Random) -> None:
        self.state = state
        self.index = index
        self.rng = rng
        self.focus: str | None = None
        self.strength = 0.0
        self.stack: list[tuple[str, float]] = []
        """Enclosing thoughts and their strengths: the thought within a thought."""
        self.monitor = SurpriseMonitor(state.config.capture)
        self.thoughts = 0
        self.stats: Counter[str] = Counter()
        self._pending: tuple[str, str] | None = None

    # ------------------------------------------------------------------ control

    def start(self, node_id: str, why: str) -> None:
        """Make ``node_id`` the next thought (used for the task itself)."""
        self._pending = (node_id, why)

    def capture(self, node_id: str) -> None:
        """A surprise takes over: it becomes the next thought, dropping the current stack."""
        self._pending = (node_id, "captured by a surprise")
        self.stats["captured"] += 1

    @property
    def branch_fatigue(self) -> float:
        """Fatigue inherited by the current focus from the thought it arose within."""
        if not self.stack:
            return 0.0
        return self.state.tree[self.stack[-1][0]].fatigue

    def decide(self) -> ThoughtNode | None:
        """Choose the next thought, or ``None`` if nothing is left to think about."""
        if self._pending is not None:
            node_id, why = self._pending
            self._pending = None
            node = self.state.tree[node_id]
            if node.status == "open":
                self._jump(node, why)
                return node
        if self.focus is None:
            return self._emerge()
        return self._advance()

    # ------------------------------------------------------------------ where to next

    def _advance(self) -> ThoughtNode | None:
        """The current thought is done: go deeper, back up, or let something new emerge."""
        cfg = self.state.config
        assert self.focus is not None
        node = self.state.tree[self.focus]
        tired = satiated(node.fatigue, cfg.fatigue)
        faded = exhausted(self.strength, cfg.fade)
        deep = too_deep(len(self.stack), cfg.fade)
        end: ClosedReason
        if not (tired or faded or deep):
            following = self._associate(node.id)
            if following is not None:
                self.stack.append((node.id, self.strength))
                self._set(
                    following,
                    child_strength(self.strength, following.link, cfg.fade),
                    "association",
                )
                self.stats["associations"] += 1
                return following
            end = "nowhere"
        else:
            end = "bored" if tired else ("exhausted" if faded else "too_deep")
        self._close(node, end)
        while self.stack:
            parent_id, strength = self.stack.pop()
            alternative = self._associate(parent_id)
            if alternative is not None:
                self.stack.append((parent_id, strength))
                self._set(
                    alternative,
                    child_strength(strength, alternative.link, cfg.fade),
                    f"return ({end})",
                )
                self.stats["returns"] += 1
                return alternative
        self.focus = None
        return self._emerge()

    def _associate(self, node_id: str) -> ThoughtNode | None:
        """Draw a not-yet-thought child of ``node_id`` (brain: ``_associate``)."""
        state = self.state
        cfg = state.config
        recent = state.recent_thought_ids()
        thought = state.thought_ids()
        candidates: list[Candidate] = []
        for child in state.tree.children(node_id):
            if child.status != "open" or self._inhibited(child, recent):
                continue
            candidates.append(
                Candidate(
                    key=child.id,
                    link=child.link,
                    redundancy=self._redundancy(child, thought),
                    value=child.value,
                    grounded=bool(child.grounding),
                )
            )
        choice = draw(
            candidates,
            self.rng,
            pull=lambda value: value_pull(value, cfg.value),
            grounded_bonus=cfg.selection.grounded_bonus,
            square=cfg.selection.square_weights,
            greedy=cfg.selection.greedy,
        )
        return None if choice is None else state.tree[choice.key]

    def _emerge(self) -> ThoughtNode | None:
        """Nothing to continue: a context detail cues a thought, or a memory surfaces."""
        state = self.state
        cfg = state.config
        cues = (
            [Option(item_id, weight) for item_id, weight in state.cue_weights.items()]
            if cfg.emergence.cued > 0
            else []
        )
        recent = state.recent_thought_ids()
        thought = state.thought_ids()
        recalls = [
            Option(
                node.id,
                recall_weight(
                    node.link,
                    state.step - node.created_step,
                    value_pull(node.value, cfg.value),
                    recency_scale=cfg.emergence.recency_scale,
                )
                # an idea that mostly repeats what was thought surfaces less readily
                * (1.0 - self._redundancy(node, thought)),
            )
            for node in state.tree.frontier()
            if not self._inhibited(node, recent)
        ]
        recalls += [
            Option(
                memory_key(finding_id),
                recall_weight(
                    finding.salience, 0.0, 1.0, recency_scale=cfg.emergence.recency_scale
                ),
            )
            for finding_id, finding in state.recalled.items()
        ]
        choice = choose_emergence(
            self.rng, cued_share=cfg.emergence.cued, cues=cues, recalls=recalls
        )
        if choice is None:
            return None
        source, key = choice
        if source is Source.CONTEXT:
            node = self._cue(key)
        elif key.startswith("mem:"):
            node = self._recall_finding(key.removeprefix("mem:"))
        else:
            node = state.tree[key]
            self.stats["resumed"] += 1
            self._jump(node, "an unexplored idea resurfaced")
        return node

    def _cue(self, item_id: str) -> ThoughtNode:
        state = self.state
        state.cue_weights.pop(item_id, None)
        item = state.context.get(item_id)
        assert item is not None
        node = state.tree.add(
            state.tree.root.id,
            kind="context",
            origin="cued",
            label=f'What does "{item.title}" imply for this task?',
            detail=f"Look for anything non-standard about this user in [{item.id}].",
            grounding=[item.id],
            created_step=state.step,
        )
        vector = state.index.get(context_key(item.id))
        if vector is not None:
            state.index.put(node.id, vector)
        self.stats["cued"] += 1
        self._jump(node, f"cued by the context item [{item.id}]")
        return node

    def _recall_finding(self, finding_id: str) -> ThoughtNode:
        state = self.state
        finding = state.recalled.pop(finding_id)
        node = state.tree.add(
            state.tree.root.id,
            kind="recalled",
            origin="recalled",
            label=finding.label,
            detail=finding.detail,
            created_step=state.step,
        )
        vector = state.index.get(memory_key(finding_id))
        if vector is not None:
            state.index.put(node.id, vector)
        self.stats["recalled"] += 1
        self._jump(node, "recalled from a past session")
        return node

    # ------------------------------------------------------------------ bookkeeping

    def _jump(self, node: ThoughtNode, why: str) -> None:
        """A new line of thought: the stack of enclosing thoughts is dropped."""
        self.stack = []
        self._set(node, 1.0, why)
        self.stats["jumps"] += 1

    def _set(self, node: ThoughtNode, strength: float, why: str) -> None:
        state = self.state
        step = state.next_step()
        self.thoughts += 1
        self.focus = node.id
        self.strength = strength
        node.status = "thinking"
        node.why = why
        node.step = step
        node.stream = self.index
        node.strength = round(strength, 4)
        state.inhibition.visit(node.id, step)
        state.log.emit(
            "selected",
            f"{why}: {node.label}",
            step=step,
            stream=self.index,
            node_id=node.id,
            data={"strength": round(strength, 3), "depth": len(self.stack)},
        )

    def _redundancy(self, node: ThoughtNode, thought: list[str]) -> float:
        """How much ``node`` repeats ideas already thought (brain: fatigue of a candidate)."""
        index = self.state.index
        vector = index.get(node.id)
        if vector is None or not self.state.config.fatigue.enabled:
            return 0.0
        similarity, _ = index.nearest(vector, thought)
        return repetition(similarity, related=index.related, duplicate=index.duplicate)

    def _inhibited(self, node: ThoughtNode, recent: list[str]) -> bool:
        """Prune ``node`` if it duplicates a recent thought (it may have been thought meanwhile)."""
        index = self.state.index
        vector = index.get(node.id)
        if vector is None or not self.state.config.inhibition.enabled:
            return False
        similarity, nearest = index.nearest(vector, recent)
        if nearest is None or similarity < index.duplicate:
            return False
        self._prune(node, nearest)
        return True

    def _close(self, node: ThoughtNode, reason: ClosedReason) -> None:
        if node.closed_reason is None:
            node.closed_reason = reason
            self.state.log.emit(
                "closed",
                f"branch ended ({reason}): {node.label}",
                step=self.state.step,
                stream=self.index,
                node_id=node.id,
                data={"reason": reason, "fatigue": round(node.fatigue, 3)},
            )

    def _prune(self, node: ThoughtNode, duplicate_of: str) -> None:
        node.status = "pruned"
        node.duplicate_of = duplicate_of
        self.state.log.emit(
            "pruned",
            f"duplicate of {duplicate_of}: {node.label}",
            step=self.state.step,
            stream=self.index,
            node_id=node.id,
            data={"duplicate_of": duplicate_of},
        )
