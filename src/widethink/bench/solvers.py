"""Solvers compared on the benchmark, all under the same token budget.

* ``direct`` - the model answers once (what users get today).
* ``broad`` - the same with an instruction to consider what is special about the user.
* ``best-of-n`` - N independent answers (as many as the budget allows) and a
  final call that picks the one fitting this user best.
* ``widethink`` - the harness.

Reasoning modes are covered by passing an effort to ``direct``; Tree of
Thoughts and parallel-reasoning baselines are on the roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, Field

from widethink.bench.task import BenchTask
from widethink.budget import Budget, Ledger, UsageSummary
from widethink.config import Effort, ThinkConfig
from widethink.context import Context
from widethink.embeddings.base import Embedder
from widethink.engine import Thinker
from widethink.llm.base import LLM, LLMRequest, Usage
from widethink.llm.structured import AttemptHook, generate_structured
from widethink.prompts import render_context
from widethink.result import ThinkResult

DIRECT_SYSTEM = "You are an expert assistant. Answer the user's task."
BROAD_SYSTEM = (
    "You are an expert assistant. Before answering, think about what is special about "
    "this particular user and project: requirements implied by their code, documents, "
    "users and constraints that the standard solution would miss. Answer the task in "
    "a way that fits them, state where you depart from the usual approach and why, and "
    "ask about anything you cannot determine."
)
SELECT_SYSTEM = (
    "You compare candidate answers to the same task. Pick the one that best fits this "
    "particular user's needs as shown by their context, not the most generic one."
)
CONTEXT_CHARS = 200_000


@dataclass
class SolverOutput:
    """An answer plus what it cost."""

    answer: str
    usage: UsageSummary
    extra: dict[str, object] = field(default_factory=dict)


class Solver(Protocol):
    @property
    def name(self) -> str: ...

    async def solve(self, task: BenchTask, budget: Budget, seed: int) -> SolverOutput: ...


def task_prompt(task: BenchTask) -> str:
    context = render_context(task.context, CONTEXT_CHARS) if task.context else "(none)"
    return f"TASK:\n{task.task}\n\nCONTEXT:\n{context}"


class _Plain(BaseModel):
    answer: str = Field(description="The complete answer.")


class _Choice(BaseModel):
    best: int = Field(description="Number of the best candidate.")
    reason: str


class DirectSolver:
    """One call, no widening; ``effort`` turns it into a reasoning-mode baseline."""

    def __init__(
        self, llm: LLM, *, effort: Effort | None = None, max_tokens: int = 8000, name: str = ""
    ) -> None:
        self.llm = llm
        self.effort = effort
        self.max_tokens = max_tokens
        self._name = name or ("direct" if effort is None else f"direct-{effort}")
        self.system = DIRECT_SYSTEM

    @property
    def name(self) -> str:
        return self._name

    async def solve(self, task: BenchTask, budget: Budget, seed: int) -> SolverOutput:
        ledger = Ledger(budget)
        answer = await _answer(
            self.llm,
            ledger,
            self.system,
            task_prompt(task),
            max_tokens=self.max_tokens,
            effort=self.effort,
            purpose="answer",
        )
        return SolverOutput(answer=answer, usage=ledger.summary())


class BroadPromptSolver(DirectSolver):
    """One call with an instruction to think about the user's specifics."""

    def __init__(self, llm: LLM, *, max_tokens: int = 8000) -> None:
        super().__init__(llm, max_tokens=max_tokens, name="broad")
        self.system = BROAD_SYSTEM


class BestOfNSolver:
    """As many independent answers as the budget allows, then a pick for this user."""

    def __init__(self, llm: LLM, *, max_n: int = 8, max_tokens: int = 8000) -> None:
        self.llm = llm
        self.max_n = max_n
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "best-of-n"

    async def solve(self, task: BenchTask, budget: Budget, seed: int) -> SolverOutput:
        ledger = Ledger(budget)
        prompt = task_prompt(task)
        answers = [
            await _answer(
                self.llm,
                ledger,
                DIRECT_SYSTEM,
                prompt,
                max_tokens=self.max_tokens,
                purpose="sample",
            )
        ]
        per_sample = ledger.estimate("sample", Usage())
        selection = Usage(
            input_tokens=per_sample.input_tokens + per_sample.output_tokens * (self.max_n + 1),
            output_tokens=300,
        )
        while len(answers) < self.max_n and ledger.affordable(per_sample, reserve=selection):
            variant = f"{prompt}\n\n(Independent attempt #{len(answers) + 1}.)"
            answers.append(
                await _answer(
                    self.llm,
                    ledger,
                    DIRECT_SYSTEM,
                    variant,
                    max_tokens=self.max_tokens,
                    purpose="sample",
                )
            )
        if len(answers) == 1:
            return SolverOutput(answer=answers[0], usage=ledger.summary(), extra={"n": 1})
        listed = "\n\n".join(f"CANDIDATE {i}:\n{a}" for i, a in enumerate(answers, 1))
        request = LLMRequest.single(
            SELECT_SYSTEM,
            f"{prompt}\n\n{listed}\n\nWhich candidate is best for this user?",
            max_tokens=2000,
            purpose="select",
            cache_system=False,
        )
        choice, _ = await generate_structured(
            self.llm, request, _Choice, on_attempt=_record(ledger, "select")
        )
        best = min(max(choice.best, 1), len(answers))
        return SolverOutput(
            answer=answers[best - 1], usage=ledger.summary(), extra={"n": len(answers)}
        )


class WideThinkSolver:
    """The harness itself; the answer includes its deviations and questions."""

    def __init__(
        self,
        llm: LLM,
        *,
        embedder: Embedder | None = None,
        config: ThinkConfig | None = None,
        name: str = "widethink",
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.config = config
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def solve(self, task: BenchTask, budget: Budget, seed: int) -> SolverOutput:
        thinker = Thinker(self.llm, embedder=self.embedder, config=self.config)
        result = await thinker.athink(
            task.task, Context.of(*task.context), budget=budget, seed=seed
        )
        return SolverOutput(
            answer=format_result(result),
            usage=result.usage,
            extra={"thoughts": result.stats.get("thoughts", 0), "stop": result.meta.stop_reason},
        )


def format_result(result: ThinkResult) -> str:
    """The harness's answer as plain text, deviations and questions included."""
    parts = [result.answer.strip()]
    if result.deviations:
        parts.append(
            "Departures from the standard solution:\n"
            + "\n".join(
                f"- Instead of {d.standard}: {d.instead}. Because {d.because}"
                for d in result.deviations
            )
        )
    if result.questions:
        parts.append(
            "Questions for you:\n"
            + "\n".join(
                f"- {q.question}" + (f" ({q.why_it_matters})" if q.why_it_matters else "")
                for q in result.questions
            )
        )
    return "\n\n".join(part for part in parts if part)


async def _answer(
    llm: LLM,
    ledger: Ledger,
    system: str,
    prompt: str,
    *,
    max_tokens: int,
    purpose: str,
    effort: Effort | None = None,
) -> str:
    request = LLMRequest.single(
        system, prompt, max_tokens=max_tokens, effort=effort, purpose=purpose, cache_system=False
    )
    out, _ = await generate_structured(llm, request, _Plain, on_attempt=_record(ledger, purpose))
    return out.answer


def _record(ledger: Ledger, purpose: str) -> AttemptHook:
    def record(usage: Usage, model: str, ok: bool) -> None:
        ledger.add(purpose, usage, model=model, ok=ok)

    return record
