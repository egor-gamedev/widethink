"""Smoke tests against real APIs. Skipped unless WIDETHINK_LIVE_TESTS=1 (they cost money).

    WIDETHINK_LIVE_TESTS=1 pytest tests/live -v

Claude needs ANTHROPIC_API_KEY (or an ``ant auth login`` profile); the
OpenAI-compatible test needs WIDETHINK_LIVE_OPENAI_MODEL and, for a local
server, OPENAI_BASE_URL.
"""

from __future__ import annotations

import os

import pytest

from widethink import Budget, Context, ThinkConfig, Thinker
from widethink.llm import AnthropicLLM, OpenAICompatibleLLM

pytestmark = pytest.mark.live

TASK = "Add JWT-based authentication to our API."
CONTEXT = Context.from_mapping(
    {
        "README.md": "Technicians work at sites without mobile coverage for two to four days.",
        "security-policy.md": "Access must be revoked within one hour, including on devices.",
    }
)
CONFIG = ThinkConfig(max_thoughts=3, parallel=1)


async def check(thinker: Thinker) -> None:
    result = await thinker.athink(TASK, CONTEXT, budget=Budget(max_tokens=80_000), seed=1)
    assert result.answer.strip()
    assert result.stats["thoughts"] >= 1
    assert result.usage.failed_calls == 0, result.warnings
    assert result.usage.total.total_tokens <= 80_000


async def test_claude_end_to_end() -> None:
    model = os.environ.get("WIDETHINK_LIVE_ANTHROPIC_MODEL")
    llm = AnthropicLLM(model) if model else AnthropicLLM()
    await check(Thinker(llm, config=CONFIG))


async def test_openai_compatible_end_to_end() -> None:
    model = os.environ.get("WIDETHINK_LIVE_OPENAI_MODEL")
    if not model:
        pytest.skip("set WIDETHINK_LIVE_OPENAI_MODEL")
    llm = OpenAICompatibleLLM(model, base_url=os.environ.get("OPENAI_BASE_URL"))
    await check(Thinker(llm, config=CONFIG))
