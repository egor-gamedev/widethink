"""The Hidden Requirements benchmark: tasks, solvers, judge, runner and report.

Tasks live in ``benchmark/tasks`` of the repository; see ``docs/benchmark.md``
for the methodology (equal budgets, controls, judge validation).
"""

from widethink.bench.judge import JudgedResult, JudgeResult, LLMJudge, RequirementVerdict
from widethink.bench.report import SolverSummary, bootstrap_ci, summarize, to_markdown
from widethink.bench.runner import RunRecord, load_records, run_benchmark
from widethink.bench.solvers import (
    BestOfNSolver,
    BroadPromptSolver,
    DirectSolver,
    Solver,
    SolverOutput,
    WideThinkSolver,
    format_result,
)
from widethink.bench.task import CANARY, BenchTask, HiddenRequirement, load_tasks

__all__ = [
    "CANARY",
    "BenchTask",
    "BestOfNSolver",
    "BroadPromptSolver",
    "DirectSolver",
    "HiddenRequirement",
    "JudgeResult",
    "JudgedResult",
    "LLMJudge",
    "RequirementVerdict",
    "RunRecord",
    "Solver",
    "SolverOutput",
    "SolverSummary",
    "WideThinkSolver",
    "bootstrap_ci",
    "format_result",
    "load_records",
    "load_tasks",
    "run_benchmark",
    "summarize",
    "to_markdown",
]
