"""Token budgets and the ledger that enforces them.

"If you give a plain model the same number of tokens, does the gain remain?"
is the first question any lab will ask, so every call - including failed and
retried ones - is recorded, and the same :class:`Budget` bounds the harness and
every baseline it is compared with.

Enforcement happens at call granularity: before a call the ledger checks that
the spent tokens plus a conservative estimate of the call plus the reserve for
the final synthesis stay within the limits. A single call can still overshoot
its estimate; the actual spend is always reported.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from widethink.llm.base import Usage


class Budget(BaseModel):
    """Limits for one run. ``None`` means unlimited in that dimension."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tokens: int | None = Field(default=None, ge=1)
    """Input plus output tokens over all model calls (cached input included)."""
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_calls: int | None = Field(default=None, ge=1)

    @classmethod
    def coerce(cls, value: Budget | int | None) -> Budget:
        """``None`` -> unlimited, an ``int`` -> a total-token limit."""
        if value is None:
            return cls()
        if isinstance(value, Budget):
            return value
        return cls(max_tokens=value)

    @property
    def unlimited(self) -> bool:
        return self.max_tokens is None and self.max_output_tokens is None and self.max_calls is None


class Pricing(BaseModel):
    """Prices in USD per million tokens, for cost reports. Supply current prices yourself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)
    cache_read: float | None = Field(default=None, ge=0.0)
    """Price of cache reads; defaults to the input price."""
    cache_write: float | None = Field(default=None, ge=0.0)
    """Price of cache writes; defaults to the input price."""

    def cost(self, usage: Usage) -> float:
        read = usage.cache_read_tokens
        written = usage.cache_write_tokens
        plain = usage.input_tokens - read - written
        total = (
            plain * self.input
            + read * (self.input if self.cache_read is None else self.cache_read)
            + written * (self.input if self.cache_write is None else self.cache_write)
            + usage.output_tokens * self.output
        )
        return total / 1_000_000


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """One model call (or one embedding batch) as the ledger saw it."""

    purpose: str
    model: str
    usage: Usage
    ok: bool
    step: int | None = None
    stream: int | None = None


class UsageSummary(BaseModel):
    """Totals of a run, overall and per purpose."""

    model_config = ConfigDict(frozen=True)

    total: Usage
    calls: int
    failed_calls: int
    by_purpose: dict[str, Usage]
    embedding_tokens: int
    models: list[str]
    """Every model that answered at least one call (fallbacks show up here)."""
    cost_usd: float | None = None


class Ledger:
    """Records usage and answers "can we afford one more call?"."""

    def __init__(self, budget: Budget | None = None) -> None:
        self.budget = budget or Budget()
        self.records: list[UsageRecord] = []
        self.embedding_tokens = 0
        self._estimates: dict[str, list[Usage]] = defaultdict(list)

    def add(
        self,
        purpose: str,
        usage: Usage,
        *,
        model: str,
        ok: bool,
        step: int | None = None,
        stream: int | None = None,
    ) -> None:
        self.records.append(UsageRecord(purpose, model, usage, ok, step, stream))
        if ok:
            self._estimates[purpose].append(usage)

    def add_embedding(self, tokens: int) -> None:
        """Embedding tokens are reported separately and do not count against the budget."""
        self.embedding_tokens += tokens

    @property
    def spent(self) -> Usage:
        total = Usage()
        for record in self.records:
            total = total + record.usage
        return total

    @property
    def calls(self) -> int:
        return len(self.records)

    def estimate(self, purpose: str, fallback: Usage) -> Usage:
        """Expected usage of the next call for ``purpose``: the largest seen so far.

        The maximum (not the mean) keeps the budget conservative, since
        prompts grow as the tree grows.
        """
        seen = self._estimates.get(purpose)
        if not seen:
            return fallback
        return Usage(
            input_tokens=max(u.input_tokens for u in seen),
            output_tokens=max(u.output_tokens for u in seen),
        )

    def affordable(self, *upcoming: Usage, calls: int = 1, reserve: Usage | None = None) -> bool:
        """Whether ``calls`` more calls using ``upcoming`` (plus ``reserve``) fit the budget."""
        planned = Usage()
        for usage in (*upcoming, reserve or Usage()):
            planned = planned + usage
        spent = self.spent
        budget = self.budget
        if budget.max_tokens is not None and (
            spent.total_tokens + planned.total_tokens > budget.max_tokens
        ):
            return False
        if budget.max_output_tokens is not None and (
            spent.output_tokens + planned.output_tokens > budget.max_output_tokens
        ):
            return False
        reserved_calls = 1 if reserve is not None and reserve.total_tokens > 0 else 0
        return budget.max_calls is None or self.calls + calls + reserved_calls <= budget.max_calls

    def summary(self, pricing: Pricing | None = None) -> UsageSummary:
        by_purpose: dict[str, Usage] = {}
        for record in self.records:
            by_purpose[record.purpose] = by_purpose.get(record.purpose, Usage()) + record.usage
        total = self.spent
        models = sorted({record.model for record in self.records if record.model})
        return UsageSummary(
            total=total,
            calls=self.calls,
            failed_calls=sum(1 for record in self.records if not record.ok),
            by_purpose=by_purpose,
            embedding_tokens=self.embedding_tokens,
            models=models,
            cost_usd=None if pricing is None else round(pricing.cost(total), 6),
        )
