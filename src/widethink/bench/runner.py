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
from widethink.errors import LLMError


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

    Records already present in ``out`` are loaded and skipped, so an interrupted
    run resumes where it stopped. Failures are recorded, not raised.
    """
    path = None if out is None else Path(out)
    done: dict[tuple[str, str, int], RunRecord] = {}
    if path is not None and path.exists():
        for record in load_records(path):
            done[record.key] = record
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    records: list[RunRecord] = list(done.values())

    async def one(task: BenchTask, solver: Solver, repeat: int) -> None:
        async with semaphore:
            record = await _solve_and_grade(
                task, solver, repeat=repeat, judge=judge, budget=budget, seed=seed
            )
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
    order = {
        (t.id, s.name, r): i
        for i, (t, s, r) in enumerate(
            (t, s, r) for r in range(repeats) for t in tasks for s in solvers
        )
    }
    return sorted(records, key=lambda rec: order.get(rec.key, len(order)))


async def _solve_and_grade(
    task: BenchTask, solver: Solver, *, repeat: int, judge: LLMJudge, budget: Budget, seed: int
) -> RunRecord:
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

    try:
        output = await solver.solve(task, budget, seed + repeat)
    except LLMError as error:
        return record(error=f"solver: {error}")
    try:
        judged = await judge.grade(task, output.answer)
    except LLMError as error:
        return record(
            answer=output.answer, usage=output.usage, extra=output.extra, error=f"judge: {error}"
        )
    return record(answer=output.answer, usage=output.usage, judged=judged, extra=output.extra)


def load_records(path: str | Path) -> list[RunRecord]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [RunRecord.model_validate(json.loads(line)) for line in lines if line.strip()]
