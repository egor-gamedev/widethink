from __future__ import annotations

import pytest

from widethink import Budget, Pricing
from widethink.budget import Ledger
from widethink.llm import Usage


def usage(inp: int, out: int, **extra: int) -> Usage:
    return Usage(input_tokens=inp, output_tokens=out, **extra)


def test_budget_coercion() -> None:
    assert Budget.coerce(None).unlimited
    assert Budget.coerce(5000).max_tokens == 5000
    budget = Budget(max_calls=3)
    assert Budget.coerce(budget) is budget
    assert not budget.unlimited


def test_usage_adds_up() -> None:
    total = usage(1, 2, cache_read_tokens=1) + usage(10, 20, reasoning_tokens=5)
    assert total == Usage(
        input_tokens=11, output_tokens=22, cache_read_tokens=1, reasoning_tokens=5
    )
    assert total.total_tokens == 33


def test_affordability_checks_every_limit() -> None:
    ledger = Ledger(Budget(max_tokens=1000, max_output_tokens=300, max_calls=3))
    ledger.add("expand", usage(400, 100), model="m", ok=True)
    assert ledger.affordable(usage(300, 100))
    assert not ledger.affordable(usage(450, 100))  # total tokens
    assert not ledger.affordable(usage(10, 250))  # output tokens
    assert ledger.affordable(usage(1, 1), calls=2)
    assert not ledger.affordable(usage(1, 1), calls=2, reserve=usage(1, 1))  # reserve takes a call


def test_unlimited_budget_affords_anything() -> None:
    assert Ledger().affordable(usage(10**9, 10**9), calls=10**6)


def test_estimate_is_the_largest_successful_call() -> None:
    ledger = Ledger()
    fallback = usage(1, 1)
    assert ledger.estimate("expand", fallback) == fallback
    ledger.add("expand", usage(100, 50), model="m", ok=True)
    ledger.add("expand", usage(80, 90), model="m", ok=True)
    ledger.add("expand", usage(9999, 9999), model="m", ok=False)
    assert ledger.estimate("expand", fallback) == usage(100, 90)


def test_summary_counts_failures_models_and_embeddings() -> None:
    ledger = Ledger()
    ledger.add("expand", usage(100, 10), model="claude-opus-5", ok=True)
    ledger.add("expand", usage(50, 5), model="claude-opus-4-8", ok=False)
    ledger.add("synthesis", usage(200, 100), model="claude-opus-5", ok=True)
    ledger.add_embedding(42)
    summary = ledger.summary(Pricing(input=5.0, output=25.0))
    assert summary.calls == 3
    assert summary.failed_calls == 1
    assert summary.by_purpose["expand"] == usage(150, 15)
    assert summary.models == ["claude-opus-4-8", "claude-opus-5"]
    assert summary.embedding_tokens == 42
    assert summary.cost_usd == pytest.approx((350 * 5 + 115 * 25) / 1_000_000)


def test_pricing_distinguishes_cache_reads_and_writes() -> None:
    pricing = Pricing(input=10.0, output=0.0, cache_read=1.0, cache_write=12.5)
    cost = pricing.cost(usage(1_000_000, 0, cache_read_tokens=500_000, cache_write_tokens=100_000))
    assert cost == pytest.approx(0.4 * 10 + 0.5 * 1 + 0.1 * 12.5)
