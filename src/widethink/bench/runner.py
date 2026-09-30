"""Running solvers over tasks at an equal budget, resumably."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from widethink.bench.judge import JudgedResult, LLMJudge
from widethink.bench.solvers import Solver
from widethink.bench.task import BenchTask
from widethink.budget import Budget, UsageSummary
from widethink.errors import LLMError, WideThinkError

#: HTTP statuses after which every further call would fail the same way:
#: bad key, no money, no permission. The run stops instead of burning through jobs.
FATAL_STATUSES = frozenset({401, 402, 403})


class BenchmarkAbortedError(WideThinkError):
    """The provider refused in a way no retry can fix; finished jobs are kept in the output."""


class RunRecord(BaseModel):
    """One solver on one task, graded."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    solver: str
    repeat: int
    control: bool
    budget: Budget
    answer: str = ""
    usage: UsageSummary | None = None
    judged: JudgedResult | None = None
    extra: dict[str, object] = Field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.task_id, self.solver, self.repeat)


async def run_benchmark(
    tasks: Sequence[BenchTask],
    solvers: Sequence[Solver],
    *,
    judge: LLMJudge,
    budget: Budget,
    repeats: int = 1,
    seed: int = 0,
    concurrency: int = 4,
    out: str | Path | None = None,
    on_record: Callable[[RunRecord], None] | None = None,
) -> list[RunRecord]:
    """Solve and grade every (task, solver, repeat); append records to ``out`` as they finish.

    Successful records already present in ``out`` are skipped, so an interrupted
    run resumes where it stopped; failed ones are run again. Failures are
    recorded, not raised.
    """
    path = None if out is None else Path(out)
    done: dict[tuple[str, str, int], RunRecord] = {}
    if path is not None and path.exists():
        for record in load_records(path):
            if record.error is None:
                done[record.key] = record
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    records: list[RunRecord] = list(done.values())
    fatal: list[str] = []

    async def one(task: BenchTask, solver: Solver, repeat: int) -> None:
        async with semaphore:
            if fatal:  # the provider already refused for good: do not spend more calls
                return
            record, is_fatal = await _solve_and_grade(
                task, solver, repeat=repeat, judge=judge, budget=budget, seed=seed
            )
            if is_fatal and not fatal:
                fatal.append(record.error or "provider refused")
        async with lock:
            records.append(record)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(record.model_dump_json() + "\n")
        if on_record is not None:
            on_record(record)

    jobs = [
        (task, solver, repeat)
        for repeat in range(repeats)
        for task in tasks
        for solver in solvers
        if (task.id, solver.name, repeat) not in done
    ]
    async with asyncio.TaskGroup() as group:
        for task, solver, repeat in jobs:
            group.create_task(one(task, solver, repeat))
    if fatal:
        raise BenchmarkAbortedError(
            f"stopped after a fatal provider error ({fatal[0]}); finished jobs are kept"
            + (f" in {path} - fix the cause and run the same command to resume" if path else "")
        )
    order = {
        (t.id, s.name, r): i
        for i, (t, s, r) in enumerate(
            (t, s, r) for r in range(repeats) for t in tasks for s in solvers
        )
    }
    return sorted(records, key=lambda rec: order.get(rec.key, len(order)))


async def _solve_and_grade(
    task: BenchTask, solver: Solver, *, repeat: int, judge: LLMJudge, budget: Budget, seed: int
) -> tuple[RunRecord, bool]:
    """Returns the record and whether the failure (if any) is fatal for the whole run."""
    started = time.perf_counter()

    def record(**fields: Any) -> RunRecord:
        return RunRecord(
            task_id=task.id,
            solver=solver.name,
            repeat=repeat,
            control=task.control,
            budget=budget,
            seconds=round(time.perf_counter() - started, 3),
            **fields,
        )

    # Any failure of one job is recorded, never raised: a long benchmark run must
    # not die because one call failed. Failed records are re-run on resume.
    try:
        output = await solver.solve(task, budget, seed + repeat)
    except Exception as error:
        cause = _root(error)
        return record(error=f"solver: {_describe(cause)}"), _is_fatal(cause)
    try:
        judged = await judge.grade(task, output.answer)
    except Exception as error:
        cause = _root(error)
        failed = record(
            answer=output.answer,
            usage=output.usage,
            extra=output.extra,
            error=f"judge: {_describe(cause)}",
        )
        return failed, _is_fatal(cause)
    ok = record(answer=output.answer, usage=output.usage, judged=judged, extra=output.extra)
    return ok, False


def _root(error: BaseException) -> BaseException:
    """The first real exception inside (nested) exception groups from task groups."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return error


def _is_fatal(error: BaseException) -> bool:
    return getattr(error, "status_code", None) in FATAL_STATUSES


def _describe(error: BaseException) -> str:
    return str(error) if isinstance(error, LLMError) else f"{type(error).__name__}: {error}"


def load_records(path: str | Path) -> list[RunRecord]:
    """Records of a results file; when a job was re-run, its last record wins."""
    latest: dict[tuple[str, str, int], RunRecord] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = RunRecord.model_validate(json.loads(line))
            latest.pop(record.key, None)
            latest[record.key] = record
    return list(latest.values())
