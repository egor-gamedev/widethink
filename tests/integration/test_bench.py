"""The Hidden Requirements benchmark: task files, solvers, judge, runner and report."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeMind
from widethink import Budget, Pricing, ThinkConfig
from widethink.bench import (
    CANARY,
    BenchTask,
    BestOfNSolver,
    BroadPromptSolver,
    DirectSolver,
    JudgedResult,
    JudgeResult,
    LLMJudge,
    RequirementVerdict,
    RunRecord,
    SolverOutput,
    WideThinkSolver,
    bootstrap_ci,
    load_records,
    load_tasks,
    run_benchmark,
    summarize,
    to_markdown,
)
from widethink.budget import UsageSummary
from widethink.errors import LLMError
from widethink.llm import LLMRequest, ScriptedLLM, Usage

TASKS_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "tasks"
_REQUIREMENT = re.compile(r"^\[([\w-]+)\] ", re.MULTILINE)


def make_task(*, control: bool = False, requirements: int = 2) -> BenchTask:
    return BenchTask.model_validate(
        {
            "id": "t-control" if control else "t-hidden",
            "title": "t",
            "domain": "test",
            "task": "Add caching",
            "context": [{"id": "doc", "title": "doc", "content": "prices differ per customer"}],
            "standard_solution": "cache by URL",
            "control": control,
            "hidden_requirements": []
            if control
            else [
                {
                    "id": f"r{i}",
                    "description": "d",
                    "evidence": ["doc"],
                    "noticed": "n?",
                    "addressed": "a?",
                }
                for i in range(requirements)
            ],
        }
    )


def judge_responder(request: LLMRequest) -> Any:
    """Marks every requirement as noticed if the answer mentions 'customer'."""
    prompt = request.messages[0].content
    rubric = prompt.split("HIDDEN REQUIREMENTS:")[1].split("ANSWER TO GRADE:")[0]
    answer = prompt.split("<answer>")[1]
    hit = "customer" in answer
    return {
        "verdicts": [
            {"requirement_id": rid, "noticed": hit, "addressed": hit, "asked": False,
             "rationale": "r"}
            for rid in _REQUIREMENT.findall(rubric)
        ],
        "unsupported_claims": [] if hit else ["invented thing"],
        "kept_standard": True,
    }  # fmt: skip


def answer_responder(text: str) -> Any:
    def respond(request: LLMRequest) -> Any:
        if request.purpose == "judge":
            return judge_responder(request)
        return text

    return respond


# ----------------------------------------------------------------------------- tasks


def test_repository_tasks_are_valid() -> None:
    tasks = load_tasks(TASKS_DIR)
    assert len(tasks) >= 5
    assert len({t.id for t in tasks}) == len(tasks)
    assert all(task.canary == CANARY for task in tasks)
    assert any(task.control for task in tasks)
    for task in tasks:
        assert task.context_obj().ids() >= {e for r in task.hidden_requirements for e in r.evidence}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"control": True}, "control task has no hidden requirements"),
        ({"hidden_requirements": []}, "needs hidden requirements"),
    ],
)
def test_inconsistent_tasks_are_rejected(change: dict[str, Any], message: str) -> None:
    data = make_task().model_dump() | change
    with pytest.raises(ValueError, match=message):
        BenchTask.model_validate(data)


def test_evidence_must_exist_in_context() -> None:
    data = make_task().model_dump()
    data["hidden_requirements"][0]["evidence"] = ["nowhere"]
    with pytest.raises(ValueError, match="not in context"):
        BenchTask.model_validate(data)


def test_json_tasks_and_duplicate_ids(tmp_path: Path) -> None:
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(make_task().model_dump_json(), encoding="utf-8")
    assert load_tasks(tmp_path / "a.json")[0].id == "t-hidden"
    with pytest.raises(ValueError, match="duplicate task id"):
        load_tasks(tmp_path)


# ----------------------------------------------------------------------------- solvers


async def test_direct_broad_and_effort_solvers() -> None:
    llm = ScriptedLLM(lambda request: request.system[-30:])
    task = make_task()
    direct = await DirectSolver(llm).solve(task, Budget(), seed=0)
    broad = await BroadPromptSolver(llm).solve(task, Budget(), seed=0)
    high = DirectSolver(llm, effort="high")
    await high.solve(task, Budget(), seed=0)
    assert direct.answer != broad.answer
    assert "special" in llm.requests[1].system
    assert llm.requests[2].effort == "high"
    assert high.name == "direct-high"
    assert direct.usage.calls == 1
    assert "prices differ per customer" in llm.requests[0].messages[0].content


async def test_best_of_n_samples_within_budget_then_selects() -> None:
    samples = iter(range(1, 100))

    def respond(request: LLMRequest) -> Any:
        if request.purpose == "select":
            return {"best": 2, "reason": "fits this customer"}
        return f"candidate {next(samples)}"

    llm = ScriptedLLM(respond)
    output = await BestOfNSolver(llm, max_n=3).solve(make_task(), Budget(), seed=0)
    assert output.extra == {"n": 3}
    assert output.answer == "candidate 2"
    assert [r.purpose for r in llm.requests] == ["sample", "sample", "sample", "select"]

    single = ScriptedLLM(respond)
    tight = await BestOfNSolver(single, max_n=3).solve(make_task(), Budget(max_calls=1), seed=0)
    assert tight.extra == {"n": 1}
    assert len(single.requests) == 1


async def test_widethink_solver_reports_deviations_and_questions() -> None:
    solver = WideThinkSolver(ScriptedLLM(FakeMind()), config=ThinkConfig(max_thoughts=3))
    output = await solver.solve(make_task(), Budget(max_tokens=200_000), seed=1)
    assert "Departures from the standard solution:" in output.answer
    assert "Questions for you:" in output.answer
    assert output.extra["thoughts"] == 3
    assert solver.name == "widethink"


# ----------------------------------------------------------------------------- judge


async def test_judge_fills_in_missing_verdicts() -> None:
    reply = {
        "verdicts": [
            {"requirement_id": "r0", "noticed": True, "addressed": False, "asked": False,
             "rationale": "mentions it"},
            {"requirement_id": "made-up", "noticed": True, "addressed": True, "asked": True,
             "rationale": "?"},
        ],
        "unsupported_claims": [],
        "kept_standard": False,
    }  # fmt: skip
    llm = ScriptedLLM(replies=[reply])
    judged = await LLMJudge(llm).grade(make_task(), "an answer")
    ids = [v.requirement_id for v in judged.result.verdicts]
    assert ids == ["r0", "r1"]
    assert not judged.result.verdicts[1].noticed
    assert "no verdict" in judged.result.verdicts[1].rationale
    assert judged.judge == "scripted"
    prompt = llm.requests[0].messages[0].content
    assert "<answer>\nan answer\n</answer>" in prompt
    assert "STANDARD SOLUTION:\ncache by URL" in prompt


async def test_control_tasks_are_judged_without_requirements() -> None:
    llm = ScriptedLLM(judge_responder)
    judged = await LLMJudge(llm).grade(make_task(control=True), "per customer")
    assert judged.result.verdicts == []
    assert "the standard solution is right" in llm.requests[0].messages[0].content


# ----------------------------------------------------------------------------- runner & report


class FailingSolver:
    name = "failing"

    async def solve(self, task: BenchTask, budget: Budget, seed: int) -> SolverOutput:
        raise LLMError("provider refused")


async def test_run_benchmark_resumes_and_records_failures(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    llm = ScriptedLLM(answer_responder("price per customer"))
    tasks = [make_task(), make_task(control=True)]
    solvers = [DirectSolver(llm), FailingSolver()]
    seen: list[RunRecord] = []
    first = await run_benchmark(
        tasks, solvers, judge=LLMJudge(llm), budget=Budget(max_tokens=10_000), out=out,
        on_record=seen.append,
    )  # fmt: skip
    assert [(r.task_id, r.solver) for r in first] == [
        ("t-hidden", "direct"), ("t-hidden", "failing"),
        ("t-control", "direct"), ("t-control", "failing"),
    ]  # fmt: skip
    assert len(seen) == 4
    failing = [r for r in first if r.solver == "failing"]
    assert all(r.error == "solver: provider refused" for r in failing)
    graded = next(r for r in first if r.solver == "direct" and not r.control)
    assert graded.judged is not None
    assert graded.budget.max_tokens == 10_000
    calls = len(llm.requests)

    again = await run_benchmark(tasks, solvers, judge=LLMJudge(llm), budget=Budget(), out=out)
    assert len(llm.requests) == calls  # nothing re-run
    assert [r.key for r in again] == [r.key for r in first]


async def test_judge_failure_is_recorded(tmp_path: Path) -> None:
    solver_llm = ScriptedLLM(lambda request: "x")
    judge_llm = ScriptedLLM(lambda request: "not json")
    [record] = await run_benchmark(
        [make_task()], [DirectSolver(solver_llm)], judge=LLMJudge(judge_llm), budget=Budget()
    )
    assert record.judged is None
    assert (record.error or "").startswith("judge:")
    assert record.answer == "x"


def record(
    solver: str, *, control: bool, flags: list[tuple[bool, bool, bool]], claims: int = 0,
    kept: bool = True, tokens: int = 100,
) -> RunRecord:  # fmt: skip
    verdicts = [
        RequirementVerdict(requirement_id=f"r{i}", noticed=n, addressed=a, asked=q, rationale="")
        for i, (n, a, q) in enumerate(flags)
    ]
    usage = UsageSummary(
        total=Usage(input_tokens=tokens - 10, output_tokens=10), calls=1, failed_calls=0,
        by_purpose={}, embedding_tokens=0, models=["m"], cost_usd=0.01,
    )  # fmt: skip
    return RunRecord(
        task_id=f"{solver}-{control}-{len(flags)}",
        solver=solver,
        repeat=0,
        control=control,
        budget=Budget(),
        answer="a",
        usage=usage,
        judged=JudgedResult(
            result=JudgeResult(
                verdicts=verdicts, unsupported_claims=["x"] * claims, kept_standard=kept
            ),
            usage=usage,
            judge="j",
        ),
    )


def test_summary_metrics() -> None:
    records = [
        record("wide", control=False, flags=[(True, True, False), (False, False, True)]),
        record("wide", control=False, flags=[(False, False, False), (True, False, False)]),
        record("wide", control=True, flags=[], claims=1),
        record("direct", control=False, flags=[(False, False, False)], tokens=50),
        record("direct", control=True, flags=[]),
    ]
    by_name = {s.solver: s for s in summarize(records, resamples=200)}
    wide = by_name["wide"]
    assert wide.noticed == 0.75  # noticed or asked: 3 of 4
    assert wide.addressed == 0.25
    assert wide.asked == 0.25
    assert wide.unsupported_per_task == pytest.approx(1 / 3, abs=1e-4)
    assert wide.control_kept == 0.0  # the control answer invented a requirement
    assert wide.mean_tokens == 100
    assert wide.mean_cost_usd == 0.01
    assert by_name["direct"].noticed == 0.0
    assert by_name["direct"].control_kept == 1.0
    table = to_markdown(list(by_name.values()))
    assert table.splitlines()[0].startswith("| solver |")
    assert "| wide | 3 / 3 | 0 | 75% (" in table


def test_bootstrap_interval() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)
    values = [0.0, 1.0] * 20
    low, high = bootstrap_ci(values, seed=3)
    assert low < 0.5 < high
    assert bootstrap_ci(values, seed=3) == (low, high)


def test_records_serialize_round_trip(tmp_path: Path) -> None:
    item = record("wide", control=False, flags=[(True, False, False)])
    line = item.model_dump_json()
    assert RunRecord.model_validate(json.loads(line)) == item


class FlakySolver:
    """Fails with a non-widethink exception first, then answers."""

    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    async def solve(self, task: BenchTask, budget: Budget, seed: int) -> SolverOutput:
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("network down")
        return SolverOutput(answer="per customer", usage=record("x", control=False, flags=[]).usage)  # type: ignore[arg-type]


async def test_failed_jobs_are_recorded_and_rerun_on_resume(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    judge = LLMJudge(ScriptedLLM(judge_responder))
    solver = FlakySolver()
    [first] = await run_benchmark([make_task()], [solver], judge=judge, budget=Budget(), out=out)
    assert first.error == "solver: ConnectionError: network down"

    [second] = await run_benchmark([make_task()], [solver], judge=judge, budget=Budget(), out=out)
    assert second.error is None
    assert solver.calls == 2
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2  # both attempts kept on disk
    [latest] = load_records(out)  # ...but the last one wins
    assert latest.error is None
    assert latest.judged is not None


def test_report_prices_recorded_usage() -> None:
    runs = [record("wide", control=False, flags=[(True, True, False)], tokens=1_000_000)]
    [summary] = summarize(runs, pricing=Pricing(input=1.0, output=10.0), resamples=10)
    assert summary.mean_cost_usd == pytest.approx((999_990 * 1.0 + 10 * 10.0) / 1_000_000)


class BrokeError(Exception):
    """Mimics an SDK error for "402 Insufficient funds"."""

    status_code = 402


class NoMoneySolver:
    name = "no-money"

    def __init__(self) -> None:
        self.calls = 0

    async def solve(self, task: BenchTask, budget: Budget, seed: int) -> SolverOutput:
        self.calls += 1
        raise ExceptionGroup("task group", [BrokeError("Insufficient funds")])


async def test_a_fatal_provider_error_stops_the_run(tmp_path: Path) -> None:
    from widethink.bench.runner import BenchmarkAbortedError

    solver = NoMoneySolver()
    out = tmp_path / "results.jsonl"
    tasks = [make_task(), make_task(control=True)]
    with pytest.raises(BenchmarkAbortedError, match="Insufficient funds") as caught:
        await run_benchmark(
            tasks, [solver], judge=LLMJudge(ScriptedLLM(judge_responder)), budget=Budget(),
            repeats=5, concurrency=1, out=out,
        )  # fmt: skip
    assert solver.calls == 1  # nothing more is spent after the first refusal
    assert "BrokeError: Insufficient funds" in str(caught.value)  # the group is unwrapped
    assert load_records(out)[0].error == "solver: BrokeError: Insufficient funds"


def test_ungraded_runs_are_not_reported_as_zero() -> None:
    failed = RunRecord(
        task_id="t", solver="wide", repeat=0, control=False, budget=Budget(), error="judge: x"
    )
    [summary] = summarize([failed], resamples=10)
    assert summary.graded == 0
    assert summary.noticed is None
    assert summary.noticed_ci is None
    assert "| wide | 0 / 1 | 1 | - | - | - | - | - |" in to_markdown([summary])


def test_judge_does_not_read_filler() -> None:
    from widethink.bench.judge import FILLER_KIND, judge_prompt

    task = make_task()
    noise = {"id": "legacy/x.py", "title": "x", "content": "NOISE" * 100, "kind": FILLER_KIND}
    padded = BenchTask.model_validate(
        task.model_dump() | {"context": [*task.model_dump()["context"], noise]}
    )
    prompt = judge_prompt(padded, "answer")
    assert "NOISE" not in prompt
    assert "prices differ per customer" in prompt
