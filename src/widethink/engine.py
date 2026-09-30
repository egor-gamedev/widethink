"""The harness: runs streams of thought over a model and turns the tree into an answer.

A run has four phases:

1. **Prepare** - embed the task and the context; recall past findings.
2. **Baseline** - ask for the standard answer once. Its key decisions enter the
   tree as already considered, so the streams spend their budget elsewhere.
3. **Think** - in rounds. In each round every stream (in a fixed order) decides
   what to think about; the model calls of all streams then run concurrently;
   their results are integrated in the same fixed order. Streams therefore see
   each other's thoughts, and a run is reproducible for a given seed and
   deterministic model.
4. **Synthesize** - one call turns the most important findings into the answer,
   stating every deviation from the standard and the questions for the user.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from widethink import prompts
from widethink.__about__ import __version__
from widethink.budget import Budget, Ledger, Pricing
from widethink.config import ThinkConfig
from widethink.context import Context, ContextItem, ContextLike
from widethink.embeddings.base import Embedder
from widethink.embeddings.hashing import HashingEmbedder
from widethink.errors import LLMError
from widethink.events import EventLog, Hook
from widethink.llm.base import LLM, LLMRequest, Usage
from widethink.llm.structured import AttemptHook, generate_structured
from widethink.mechanisms.fade import proposal_count, reinstated_items
from widethink.mechanisms.fatigue import mean_repetition, repetition, update_fatigue
from widethink.mechanisms.inhibition import InhibitionOfReturn
from widethink.memory.store import Finding, FindingStore
from widethink.result import Baseline, Deviation, Question, RunMeta, ThinkResult
from widethink.schemas import BaselineOut, CriticOut, ExpansionOut, SynthesisOut
from widethink.semantic import SemanticIndex
from widethink.state import RunState, context_key, memory_key
from widethink.stream import ThoughtStream
from widethink.tree import Evidence, ThoughtNode, ThoughtTree

#: Assumed cost of a thinking step before any has been observed.
FIRST_STEP_ESTIMATE = Usage(input_tokens=3000, output_tokens=1200)
FIRST_CRITIC_ESTIMATE = Usage(input_tokens=1500, output_tokens=400)
BASELINE_ESTIMATE = Usage(input_tokens=3000, output_tokens=2500)
#: Characters of a context item used to embed it.
EMBED_CHARS = 6000
CRITIC_MAX_TOKENS = 2000


@dataclass
class _Expansion:
    output: ExpansionOut | None
    error: LLMError | None
    requested: int


class Thinker:
    """Wide thinking over any :class:`~widethink.llm.base.LLM`.

    Args:
        llm: The model that thinks, and writes the baseline and the final answer.
        embedder: Measures similarity between ideas. The default hashing embedder
            works offline but only sees surface form; use a neural embedder for
            real runs.
        config: Mechanism parameters; defaults come from the digital brain.
        critic: Model for the separate critic (``config.value.critic == "separate"``);
            defaults to ``llm``.
        memory: Store of findings shared across runs; ``None`` disables memory.
        hooks: Callables receiving every :class:`~widethink.events.Event` live.

    Example:
        >>> from widethink import Thinker
        >>> from widethink.llm import AnthropicLLM
        >>> thinker = Thinker(AnthropicLLM())  # doctest: +SKIP
        >>> result = thinker.think(
        ...     "Add JWT auth to our API", "./my-project", budget=60_000
        ... )  # doctest: +SKIP
        >>> print(result.answer)  # doctest: +SKIP
    """

    def __init__(
        self,
        llm: LLM,
        *,
        embedder: Embedder | None = None,
        config: ThinkConfig | None = None,
        critic: LLM | None = None,
        memory: FindingStore | None = None,
        hooks: Sequence[Hook] = (),
    ) -> None:
        self.llm = llm
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.config = config or ThinkConfig()
        self.critic = critic or llm
        self.memory = memory
        self.hooks = list(hooks)

    # ------------------------------------------------------------------ public API

    def think(
        self,
        task: str,
        context: ContextLike = None,
        *,
        budget: Budget | int | None = None,
        seed: int | None = None,
        pricing: Pricing | None = None,
    ) -> ThinkResult:
        """Synchronous wrapper around :meth:`athink`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.athink(task, context, budget=budget, seed=seed, pricing=pricing)
            )
        raise RuntimeError(
            "Thinker.think() cannot run inside an event loop; use `await thinker.athink(...)`"
        )

    async def athink(
        self,
        task: str,
        context: ContextLike = None,
        *,
        budget: Budget | int | None = None,
        seed: int | None = None,
        pricing: Pricing | None = None,
    ) -> ThinkResult:
        """Think about ``task`` in the light of ``context`` within ``budget``.

        Args:
            task: What the user asked for.
            context: The user's context - a path to a project, free text, a
                mapping of titles to texts, or a :class:`~widethink.context.Context`.
            budget: A :class:`~widethink.budget.Budget`, or a total-token limit.
            seed: Seed of the streams' lotteries; drawn at random if omitted and
                recorded in the result either way.
            pricing: Prices for the cost report.
        """
        if not task.strip():
            raise ValueError("task must not be empty")
        started = time.perf_counter()
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
        seed = random.SystemRandom().randrange(2**31) if seed is None else seed
        cfg = self.config
        ledger = Ledger(Budget.coerce(budget))
        log = EventLog(self.hooks)
        state = RunState(
            task=task.strip(),
            context=Context.coerce(context),
            config=cfg,
            tree=ThoughtTree(task.strip()),
            index=SemanticIndex(self.embedder, on_tokens=ledger.add_embedding),
            inhibition=InhibitionOfReturn(cfg.inhibition.window, enabled=cfg.inhibition.enabled),
            ledger=ledger,
            log=log,
        )
        state.system_prompt = prompts.expansion_system(state.task, state.context, cfg.reinstatement)
        log.emit(
            "run_started",
            f"thinking about: {state.task[:120]}",
            data={"seed": seed, "context_items": len(state.context), "llm": self.llm.name},
        )

        await self._prepare(state)
        if cfg.baseline:
            await self._baseline(state)
        streams = [
            ThoughtStream(state, number, random.Random(f"{seed}:{number}"))
            for number in range(cfg.parallel)
        ]
        streams[0].start(state.tree.root.id, "the task itself")
        while state.stop_reason is None and await self._round(state, streams):
            pass
        answer = await self._synthesize(state)
        if self.memory is not None:
            await self._remember(state)
        log.emit("run_finished", "done", step=state.step, data={"reason": state.stop_reason})
        return self._result(
            state,
            streams,
            answer,
            seed=seed,
            started=started,
            started_at=started_at,
            pricing=pricing,
        )

    # ------------------------------------------------------------------ phases

    async def _prepare(self, state: RunState) -> None:
        items = state.context.items
        texts = [state.task] + [f"{item.title}\n{item.content[:EMBED_CHARS]}" for item in items]
        vectors = await state.index.embed(texts)
        task_vector = vectors[0]
        state.index.put(state.tree.root.id, task_vector)
        for item, vector in zip(items, vectors[1:], strict=True):
            state.index.put(context_key(item.id), vector)
            relevance = max(0.0, float(vector @ task_vector))
            state.cue_weights[item.id] = item.salience * (0.5 + relevance)
        wanted = state.config.emergence.memory_recall
        if self.memory is not None and wanted:
            findings = await self.memory.recall(state.task, wanted)
            recalled_vectors = await state.index.embed([finding.text() for finding in findings])
            for finding, vector in zip(findings, recalled_vectors, strict=True):
                state.recalled[finding.id] = finding
                state.index.put(memory_key(finding.id), vector)

    async def _baseline(self, state: RunState) -> None:
        cfg = state.config
        if not state.ledger.affordable(BASELINE_ESTIMATE, reserve=self._synthesis_reserve()):
            state.warnings.append("budget too small for a baseline; skipped it")
            return
        request = LLMRequest.single(
            prompts.BASELINE_SYSTEM,
            prompts.baseline_user(state.task, state.context, cfg.reinstatement),
            max_tokens=cfg.baseline_max_tokens,
            purpose="baseline",
            cache_system=False,
        )
        try:
            out, _ = await generate_structured(
                self.llm,
                request,
                BaselineOut,
                retries=cfg.expansion.retries,
                on_attempt=self._recorder(state, "baseline"),
            )
        except LLMError as error:
            state.warnings.append(f"baseline failed: {error}")
            return
        state.baseline = Baseline(
            approach=out.approach.strip(),
            key_decisions=[d.strip() for d in out.key_decisions if d.strip()][:7],
            answer=out.answer.strip(),
        )
        tree = state.tree
        anchor = tree.add(
            tree.root.id,
            kind="solution",
            origin="baseline",
            label=f"Standard solution: {state.baseline.approach}",
            status="expanded",
        )
        anchor.why = "baseline"
        anchor.elaboration = state.baseline.approach
        nodes = [anchor] + [
            tree.add(
                anchor.id, kind="solution", origin="baseline", label=decision, status="expanded"
            )
            for decision in state.baseline.key_decisions
        ]
        vectors = await state.index.embed([node.text() for node in nodes])
        for node, vector in zip(nodes, vectors, strict=True):
            state.index.put(node.id, vector)
            state.inhibition.visit(node.id, 0)
        state.log.emit(
            "baseline",
            f"standard solution: {state.baseline.approach[:120]}",
            node_id=anchor.id,
            data={"decisions": len(state.baseline.key_decisions)},
        )

    async def _round(self, state: RunState, streams: list[ThoughtStream]) -> bool:
        """One round: decide in order, call concurrently, integrate in order."""
        cfg = state.config
        plans: list[tuple[ThoughtStream, ThoughtNode]] = []
        for stream in streams:
            if cfg.max_thoughts is not None and state.step >= cfg.max_thoughts:
                state.stop("max_thoughts")
                break
            upcoming = [self._step_estimate(state)] * (len(plans) + 1)
            if not state.ledger.affordable(
                *upcoming,
                calls=self._calls_per_step() * len(upcoming),
                reserve=self._synthesis_reserve(),
            ):
                state.stop("budget")
                break
            node = stream.decide()
            if node is not None:
                plans.append((stream, node))
        if not plans:
            state.stop("exhausted")
            return False

        async with asyncio.TaskGroup() as group:
            calls = [group.create_task(self._expand(state, s, n)) for s, n in plans]
        for (stream, node), call in zip(plans, calls, strict=True):
            await self._integrate(state, stream, node, call.result())

        if cfg.value.critic == "separate":
            expanded = [node for _, node in plans if node.status == "expanded"]
            await self._critique(state, expanded)
        return True

    async def _expand(
        self, state: RunState, stream: ThoughtStream, node: ThoughtNode
    ) -> _Expansion:
        cfg = state.config
        strength = stream.strength
        proposals = proposal_count(strength, cfg.expansion)
        request = LLMRequest.single(
            state.system_prompt,
            prompts.expansion_user(
                path=state.tree.path(node.id),
                focus=node,
                relevant=self._reinstate(
                    state, node, reinstated_items(strength, cfg.reinstatement)
                ),
                cfg=cfg.reinstatement,
                considered=self._considered(state, node),
                proposals=proposals,
                context_proposals=min(cfg.expansion.context_candidates, proposals),
                detailed=strength >= 0.6,
            ),
            max_tokens=cfg.expansion.max_tokens,
            effort=cfg.expansion.effort,
            purpose="expand",
        )
        try:
            out, _ = await generate_structured(
                self.llm,
                request,
                ExpansionOut,
                retries=cfg.expansion.retries,
                on_attempt=self._recorder(state, "expand", node.step, stream.index),
            )
        except LLMError as error:
            return _Expansion(None, error, proposals)
        return _Expansion(out, None, proposals)

    async def _integrate(
        self, state: RunState, stream: ThoughtStream, node: ThoughtNode, expansion: _Expansion
    ) -> None:
        """Write one thought into the tree: evidence, questions, proposals, fatigue, surprise."""
        cfg = state.config
        tree, index, log = state.tree, state.index, state.log
        if expansion.output is None:
            node.status = "failed"
            node.error = str(expansion.error)
            log.emit(
                "failed",
                f"thought failed: {expansion.error}",
                step=node.step,
                stream=stream.index,
                node_id=node.id,
            )
            return
        out = expansion.output
        known = state.context.ids()
        node.status = "expanded"
        node.elaboration = out.elaboration.strip()
        node.evidence = [
            Evidence(source=e.source, quote=e.quote.strip())
            for e in out.evidence
            if e.source in known and e.quote.strip()
        ][:5]
        node.resolution = out.resolution
        node.surprise = out.surprise.level
        if out.resolution == "unknown" and out.question.strip():
            node.question = out.question.strip()
            state.questions.append(node.id)
            log.emit(
                "question", node.question, step=node.step, stream=stream.index, node_id=node.id
            )

        proposals = [p for p in out.next if p.idea.strip()][: expansion.requested]
        vectors = await index.embed([f"{p.idea}. {p.detail}" for p in proposals])
        comparable = state.comparable_ids()
        degrees: list[float] = []
        for proposal, vector in zip(proposals, vectors, strict=True):
            similarity, nearest = index.nearest(vector, comparable)
            degrees.append(repetition(similarity, related=index.related, duplicate=index.duplicate))
            duplicate = (
                nearest is not None
                and similarity >= index.duplicate
                and state.blocks(tree[nearest])
            )
            child = tree.add(
                node.id,
                kind=proposal.kind,
                origin="proposal",
                label=proposal.idea,
                detail=proposal.detail,
                link=proposal.link,
                value=proposal.value if cfg.value.critic == "inline" else None,
                grounding=[g for g in proposal.grounding if g in known],
                created_step=state.step,
                status="pruned" if duplicate else "open",
            )
            if duplicate:
                child.duplicate_of = nearest
                log.emit(
                    "pruned",
                    f"duplicate of {nearest}: {child.label}",
                    step=node.step,
                    stream=stream.index,
                    node_id=child.id,
                )
            else:
                index.put(child.id, vector)
                comparable.append(child.id)
        node.fatigue = update_fatigue(stream.branch_fatigue, mean_repetition(degrees), cfg.fatigue)

        about = out.surprise.about.strip()
        if stream.monitor.observe(out.surprise.level, stream.thoughts) and about:
            [vector] = await index.embed([about])
            similarity, nearest = index.nearest(vector, state.comparable_ids())
            if nearest is None or similarity < index.duplicate:
                surprise = tree.add(
                    node.id,
                    kind="surprise",
                    origin="captured",
                    label=about,
                    created_step=state.step,
                )
                index.put(surprise.id, vector)
                stream.capture(surprise.id)
                log.emit(
                    "captured",
                    f"surprise: {about}",
                    step=node.step,
                    stream=stream.index,
                    node_id=surprise.id,
                    data={"level": out.surprise.level},
                )
        log.emit(
            "expanded",
            node.label,
            step=node.step,
            stream=stream.index,
            node_id=node.id,
            data={
                "proposals": len(proposals),
                "fatigue": round(node.fatigue, 3),
                "resolution": node.resolution,
            },
        )

    async def _critique(self, state: RunState, nodes: list[ThoughtNode]) -> None:
        """Separate critic: score the new proposals of each thought against the user's needs."""
        jobs = [
            (node, children)
            for node in nodes
            if (
                children := [
                    c
                    for c in state.tree.children(node.id)
                    if c.status == "open" and c.origin == "proposal"
                ]
            )
        ]
        if not jobs:
            return
        async with asyncio.TaskGroup() as group:
            calls = [group.create_task(self._critic_call(state, n, c)) for n, c in jobs]
        for (_, children), call in zip(jobs, calls, strict=True):
            scores = call.result()
            if scores is None:
                continue
            for score in scores.scores:
                if 1 <= score.index <= len(children):
                    children[score.index - 1].value = score.value

    async def _critic_call(
        self, state: RunState, node: ThoughtNode, children: list[ThoughtNode]
    ) -> CriticOut | None:
        cfg = state.config
        relevant = self._reinstate(state, node, cfg.reinstatement.items)
        request = LLMRequest.single(
            prompts.CRITIC_SYSTEM,
            prompts.critic_user(state.task, node, relevant, children, cfg.reinstatement),
            max_tokens=CRITIC_MAX_TOKENS,
            purpose="critic",
            cache_system=False,
        )
        try:
            out, _ = await generate_structured(
                self.critic,
                request,
                CriticOut,
                retries=cfg.expansion.retries,
                on_attempt=self._recorder(state, "critic", node.step, node.stream),
            )
        except LLMError as error:
            state.warnings.append(f"critic failed on {node.id}: {error}")
            return None
        return out

    async def _synthesize(self, state: RunState) -> SynthesisOut | None:
        cfg = state.config
        findings = self._findings(state)
        questions = [state.tree[node_id] for node_id in state.questions]
        cited = [e.source for node in findings for e in node.evidence]
        cited += [g for node in findings for g in node.grounding]
        relevant = [
            item
            for item_id in dict.fromkeys(cited)
            if (item := state.context.get(item_id)) is not None
        ][:6]
        baseline = state.baseline
        request = LLMRequest.single(
            prompts.SYNTHESIS_SYSTEM,
            prompts.synthesis_user(
                task=state.task,
                approach=None if baseline is None else baseline.approach,
                decisions=[] if baseline is None else baseline.key_decisions,
                findings=findings,
                questions=questions,
                relevant=relevant,
                cfg=cfg.reinstatement,
            ),
            max_tokens=cfg.synthesis.max_tokens,
            effort=cfg.synthesis.effort,
            purpose="synthesis",
            cache_system=False,
        )
        try:
            out, _ = await generate_structured(
                self.llm,
                request,
                SynthesisOut,
                retries=cfg.expansion.retries,
                on_attempt=self._recorder(state, "synthesis"),
            )
        except LLMError as error:
            state.warnings.append(f"synthesis failed: {error}")
            return None
        state.log.emit(
            "synthesized",
            f"{len(out.deviations)} deviations, {len(out.questions)} questions",
            step=state.step,
        )
        return out

    async def _remember(self, state: RunState) -> None:
        assert self.memory is not None
        keep = [
            node
            for node in self._findings(state)
            if (node.value or 0.0) >= 0.3
            and (node.resolution in ("supported", "contradicted") or node.kind == "surprise")
        ]
        await self.memory.remember(
            [
                Finding(
                    label=node.label,
                    detail=node.elaboration[:500],
                    task=state.task,
                    kind=node.kind,
                    salience=min(1.0, max(0.0, node.value or 0.0)),
                )
                for node in keep
            ]
        )

    # ------------------------------------------------------------------ helpers

    def _reinstate(self, state: RunState, node: ThoughtNode, limit: int) -> list[ContextItem]:
        """Context items that accompany a thought: its grounding first, then the nearest."""
        if limit <= 0 or not state.context.items:
            return []
        chosen: list[str] = []
        parent = state.tree.parent(node.id)
        preferred = list(node.grounding)
        if parent is not None:
            preferred += [e.source for e in parent.evidence] + list(parent.grounding)
        for item_id in preferred:
            if item_id not in chosen and state.context.get(item_id) is not None:
                chosen.append(item_id)
        vector = state.index.get(node.id)
        if vector is not None:
            keys = [context_key(item.id) for item in state.context.items]
            for key, _ in state.index.rank(vector, keys):
                item_id = key.removeprefix("ctx:")
                if item_id not in chosen:
                    chosen.append(item_id)
                if len(chosen) >= limit:
                    break
        return [item for item_id in chosen[:limit] if (item := state.context.get(item_id))]

    def _considered(self, state: RunState, focus: ThoughtNode) -> list[str]:
        """Labels of recent thoughts the model must not repeat (inhibition of return)."""
        shown = state.config.inhibition.shown
        if not state.config.inhibition.enabled or shown == 0:
            return []
        labels = [
            state.tree[node_id].label
            for node_id in state.recent_thought_ids()
            if node_id != focus.id and state.tree[node_id].kind != "task"
        ]
        return labels[-shown:]

    def _findings(self, state: RunState) -> list[ThoughtNode]:
        """The thoughts most worth showing to the synthesis, in the order they were thought."""

        def importance(node: ThoughtNode) -> float:
            score = node.value if node.value is not None else 0.0
            score += 0.3 if node.resolution in ("supported", "contradicted") else 0.0
            score += 0.3 if node.kind == "surprise" else 0.0
            score += 0.2 if node.evidence else 0.0
            score += 0.2 if node.question else 0.0
            return score

        thought = [
            node
            for node in state.tree.nodes()
            if node.status == "expanded" and node.origin not in ("task", "baseline")
        ]
        top = sorted(thought, key=importance, reverse=True)[: state.config.synthesis.top_nodes]
        return sorted(top, key=lambda node: node.step or 0)

    def _recorder(
        self, state: RunState, purpose: str, step: int | None = None, stream: int | None = None
    ) -> AttemptHook:
        def record(usage: Usage, model: str, ok: bool) -> None:
            state.ledger.add(purpose, usage, model=model, ok=ok, step=step, stream=stream)

        return record

    def _calls_per_step(self) -> int:
        return 2 if self.config.value.critic == "separate" else 1

    def _step_estimate(self, state: RunState) -> Usage:
        estimate = state.ledger.estimate("expand", FIRST_STEP_ESTIMATE)
        if self.config.value.critic == "separate":
            estimate = estimate + state.ledger.estimate("critic", FIRST_CRITIC_ESTIMATE)
        return estimate

    def _synthesis_reserve(self) -> Usage:
        synthesis = self.config.synthesis
        return Usage(
            input_tokens=synthesis.reserve_input_tokens,
            output_tokens=synthesis.reserve_output_tokens,
        )

    def _result(
        self,
        state: RunState,
        streams: list[ThoughtStream],
        synthesis: SynthesisOut | None,
        *,
        seed: int,
        started: float,
        started_at: str,
        pricing: Pricing | None,
    ) -> ThinkResult:
        stats: Counter[str] = Counter()
        for stream in streams:
            stats.update(stream.stats)
        nodes = state.tree.nodes()
        stats["thoughts"] = state.step
        stats["nodes"] = len(nodes)
        stats["questions"] = len(state.questions)
        for node in nodes:
            if node.status in ("pruned", "failed", "open"):
                stats[
                    {"pruned": "pruned", "failed": "failed", "open": "unexplored"}[node.status]
                ] += 1
            if node.closed_reason is not None:
                stats[f"closed_{node.closed_reason}"] += 1
        if synthesis is None:
            answer = "" if state.baseline is None else state.baseline.answer
            if state.baseline is not None:
                state.warnings.append("the answer is the baseline: synthesis did not run")
            deviations: list[Deviation] = []
            questions = [
                Question(question=state.tree[q].question, node_id=q) for q in state.questions
            ]
            confidence = None
        else:
            answer = synthesis.answer
            deviations = [Deviation(**d.model_dump()) for d in synthesis.deviations]
            questions = [Question(**q.model_dump()) for q in synthesis.questions]
            confidence = synthesis.confidence
        return ThinkResult(
            task=state.task,
            answer=answer,
            deviations=deviations,
            questions=questions,
            confidence=confidence,
            baseline=state.baseline,
            nodes=nodes,
            stats=dict(sorted(stats.items())),
            usage=state.ledger.summary(pricing),
            events=state.log.events,
            config=state.config,
            meta=RunMeta(
                widethink_version=__version__,
                prompts_version=prompts.PROMPTS_VERSION,
                llm=self.llm.name,
                critic=self.critic.name,
                embedder=self.embedder.name,
                seed=seed,
                started_at=started_at,
                duration_s=round(time.perf_counter() - started, 3),
                stop_reason=state.stop_reason,
            ),
            warnings=state.warnings,
        )
