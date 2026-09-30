"""Aggregating graded runs into the numbers a paper reports."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from statistics import mean

from pydantic import BaseModel, ConfigDict

from widethink.bench.runner import RunRecord
from widethink.budget import Pricing


class SolverSummary(BaseModel):
    """Headline metrics of one solver. Rates are in [0, 1]; ``ci`` are 95% bootstrap intervals."""

    model_config = ConfigDict(frozen=True)

    solver: str
    runs: int
    graded: int
    """Runs the judge actually graded; metrics are over these only."""
    errors: int
    noticed: float | None
    """Share of hidden requirements the answer noticed or asked about (None: nothing graded)."""
    noticed_ci: tuple[float, float] | None
    addressed: float | None
    """Share of hidden requirements the solution actually handles."""
    asked: float | None
    unsupported_per_task: float | None
    """Invented requirements per task (false alarms)."""
    control_kept: float | None
    """Share of control tasks answered with the standard solution and no inventions."""
    mean_tokens: float
    mean_output_tokens: float
    mean_cost_usd: float | None


def summarize(
    records: Sequence[RunRecord],
    *,
    pricing: Pricing | None = None,
    resamples: int = 2000,
    seed: int = 0,
) -> list[SolverSummary]:
    """Per-solver metrics. With ``pricing``, costs are computed from the recorded usage."""
    by_solver: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_solver[record.solver].append(record)
    return [
        _summarize(solver, runs, pricing, resamples, seed)
        for solver, runs in sorted(by_solver.items())
    ]


def _summarize(
    solver: str, runs: list[RunRecord], pricing: Pricing | None, resamples: int, seed: int
) -> SolverSummary:
    graded = [r for r in runs if r.judged is not None]
    hidden = [r for r in graded if not r.control]
    controls = [r for r in graded if r.control]

    per_task_noticed: list[float] = []
    noticed: list[float] = []
    addressed: list[float] = []
    asked: list[float] = []
    for record in hidden:
        assert record.judged is not None
        verdicts = record.judged.result.verdicts
        flags = [1.0 if v.noticed or v.asked else 0.0 for v in verdicts]
        noticed.extend(flags)
        addressed.extend(1.0 if v.addressed else 0.0 for v in verdicts)
        asked.extend(1.0 if v.asked else 0.0 for v in verdicts)
        if flags:
            per_task_noticed.append(mean(flags))

    control_kept = None
    if controls:
        control_kept = mean(
            1.0
            if r.judged and r.judged.result.kept_standard and not r.judged.result.unsupported_claims
            else 0.0
            for r in controls
        )
    usages = [r.usage for r in runs if r.usage is not None]
    if pricing is not None:
        costs = [pricing.cost(u.total) for u in usages]
    else:
        costs = [u.cost_usd for u in usages if u.cost_usd is not None]
    return SolverSummary(
        solver=solver,
        runs=len(runs),
        graded=len(graded),
        errors=sum(1 for r in runs if r.error),
        noticed=_rate(noticed),
        noticed_ci=(
            bootstrap_ci(per_task_noticed, resamples=resamples, seed=seed)
            if per_task_noticed
            else None
        ),
        addressed=_rate(addressed),
        asked=_rate(asked),
        unsupported_per_task=_rate(
            [float(len(r.judged.result.unsupported_claims)) for r in graded if r.judged]
        ),
        control_kept=control_kept,
        mean_tokens=_mean([float(u.total.total_tokens) for u in usages]),
        mean_output_tokens=_mean([float(u.total.output_tokens) for u in usages]),
        mean_cost_usd=round(mean(costs), 6) if costs else None,
    )


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean (over tasks, the unit of sampling)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(max(1, resamples)))
    low = means[int((1 - level) / 2 * (len(means) - 1))]
    high = means[int((1 + level) / 2 * (len(means) - 1))]
    return (round(low, 4), round(high, 4))


def to_markdown(summaries: Sequence[SolverSummary]) -> str:
    header = (
        "| solver | graded / runs | errors | noticed (95% CI) | addressed | asked "
        "| inventions/task | control kept | tokens | output tokens | cost |\n"
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for s in summaries:
        noticed = _pct(s.noticed)
        if s.noticed_ci is not None:
            noticed += f" ({s.noticed_ci[0]:.0%}-{s.noticed_ci[1]:.0%})"
        inventions = "-" if s.unsupported_per_task is None else f"{s.unsupported_per_task:.2f}"
        cost = "-" if s.mean_cost_usd is None else f"${s.mean_cost_usd:.4f}"
        rows.append(
            f"| {s.solver} | {s.graded} / {s.runs} | {s.errors} | {noticed}"
            f" | {_pct(s.addressed)} | {_pct(s.asked)} | {inventions} | {_pct(s.control_kept)}"
            f" | {s.mean_tokens:,.0f} | {s.mean_output_tokens:,.0f} | {cost} |"
        )
    return "\n".join([header, *rows])


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _rate(values: Sequence[float]) -> float | None:
    """Mean of graded values; None when nothing was graded (not the same as 0%)."""
    return round(mean(values), 4) if values else None


def _mean(values: Sequence[float]) -> float:
    return round(mean(values), 4) if values else 0.0
