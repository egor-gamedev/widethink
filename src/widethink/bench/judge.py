"""Grading answers against the hidden requirements of a task.

The judge is a model with a per-requirement rubric. Its verdicts are only as
good as its agreement with people, so the benchmark protocol (docs/benchmark.md)
requires validating it against human labels and reporting the agreement. Use a
judge from a different model family than the solvers to limit self-preference.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from widethink.bench.task import BenchTask
from widethink.budget import Ledger, UsageSummary
from widethink.llm.base import LLM, LLMRequest, Usage
from widethink.llm.structured import generate_structured
from widethink.prompts import render_context

JUDGE_PROMPT_VERSION = "2026-09-30.1"

JUDGE_SYSTEM = """\
You grade answers for a benchmark of hidden requirements. A task has a standard \
solution that a typical model gives, and a context (code, documents, statements of \
the user) that may imply requirements the standard solution misses. You receive the \
hidden requirements with a rubric for each, and one answer to grade.

For every hidden requirement decide, literally and strictly:
- noticed: the answer explicitly recognizes the requirement (not just something vaguely \
related);
- addressed: the proposed solution actually handles it, not merely mentions it;
- asked: the answer asks the user a question that targets it.

Then list unsupported_claims: special requirements the answer asserts about this user \
that nothing in the context supports (inventions, not reasonable questions). Finally, \
decide kept_standard: whether the answer keeps the standard solution wherever the \
context does not justify a change.

Judge only what is written in the answer."""


class RequirementVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    requirement_id: str
    # A field the judge forgets to return counts as "no" instead of failing the grading.
    noticed: bool = False
    addressed: bool = False
    asked: bool = False
    rationale: str = Field(default="", description="One or two sentences citing the answer.")


class JudgeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdicts: list[RequirementVerdict]
    unsupported_claims: list[str]
    kept_standard: bool


class JudgedResult(BaseModel):
    """A verdict together with what the judging cost."""

    model_config = ConfigDict(frozen=True)

    result: JudgeResult
    usage: UsageSummary
    judge: str
    prompt_version: str = JUDGE_PROMPT_VERSION


class LLMJudge:
    """Grades one answer per call."""

    def __init__(self, llm: LLM, *, max_tokens: int = 4000) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    async def grade(self, task: BenchTask, answer: str) -> JudgedResult:
        ledger = Ledger()
        request = LLMRequest.single(
            JUDGE_SYSTEM,
            judge_prompt(task, answer),
            max_tokens=self.max_tokens,
            purpose="judge",
            cache_system=False,
        )

        def record(usage: Usage, model: str, ok: bool) -> None:
            ledger.add("judge", usage, model=model, ok=ok)

        result, _ = await generate_structured(self.llm, request, JudgeResult, on_attempt=record)
        known = {r.id for r in task.hidden_requirements}
        verdicts = {v.requirement_id: v for v in result.verdicts if v.requirement_id in known}
        complete = [
            verdicts.get(r.id)
            or RequirementVerdict(
                requirement_id=r.id,
                noticed=False,
                addressed=False,
                asked=False,
                rationale="(no verdict returned; counted as missed)",
            )
            for r in task.hidden_requirements
        ]
        return JudgedResult(
            result=result.model_copy(update={"verdicts": complete}),
            usage=ledger.summary(),
            judge=self.llm.name,
        )


def judge_prompt(task: BenchTask, answer: str) -> str:
    context = render_context(task.context, 200_000) if task.context else "(none)"
    if task.hidden_requirements:
        rubric = "\n\n".join(
            f"[{r.id}] {r.description}\n  noticed? {r.noticed}\n  addressed? {r.addressed}"
            for r in task.hidden_requirements
        )
    else:
        rubric = "(none - the standard solution is right for this user)"
    return (
        f"TASK:\n{task.task}\n\nCONTEXT:\n{context}\n\n"
        f"STANDARD SOLUTION:\n{task.standard_solution}\n\n"
        f"HIDDEN REQUIREMENTS:\n{rubric}\n\n"
        f"ANSWER TO GRADE:\n<answer>\n{answer}\n</answer>"
    )
