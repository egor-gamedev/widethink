"""Smoke tests against real APIs. Skipped unless WIDETHINK_LIVE_TESTS=1 (they cost money).

    WIDETHINK_LIVE_TESTS=1 pytest tests/live -v

Each test also needs its provider's credentials and is skipped without them:
ANTHROPIC_API_KEY for Claude, DEEPSEEK_API_KEY for DeepSeek, and
WIDETHINK_LIVE_OPENAI_MODEL (plus OPENAI_BASE_URL for a local server) for the
OpenAI-compatible test.
"""

from __future__ import annotations

import os

import pytest

from widethink import Budget, Context, ThinkConfig, Thinker
from widethink.llm import (
    AnthropicLLM,
    DeepSeekLLM,
    LLMRequest,
    OpenAICompatibleLLM,
    generate_text,
)

pytestmark = pytest.mark.live

TASK = "Add JWT-based authentication to our API."
CONTEXT = Context.from_mapping(
    {
        "README.md": "Technicians work at sites without mobile coverage for two to four days.",
        "security-policy.md": "Access must be revoked within one hour, including on devices.",
    }
)
CONFIG = ThinkConfig(max_thoughts=3, parallel=1)


def needs(*variables: str) -> None:
    if not any(os.environ.get(name) for name in variables):
        pytest.skip(f"set {' or '.join(variables)}")


async def check(thinker: Thinker) -> None:
    result = await thinker.athink(TASK, CONTEXT, budget=Budget(max_tokens=80_000), seed=1)
    assert result.answer.strip()
    assert result.stats["thoughts"] >= 1
    assert result.usage.failed_calls == 0, result.warnings
    assert result.usage.total.total_tokens <= 80_000


async def test_claude_end_to_end() -> None:
    needs("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    model = os.environ.get("WIDETHINK_LIVE_ANTHROPIC_MODEL")
    llm = AnthropicLLM(model) if model else AnthropicLLM()
    await check(Thinker(llm, config=CONFIG))


async def test_deepseek_end_to_end() -> None:
    needs("DEEPSEEK_API_KEY")
    await check(
        Thinker(
            DeepSeekLLM(os.environ.get("WIDETHINK_LIVE_DEEPSEEK_MODEL") or "deepseek-flash"),
            config=CONFIG,
        )
    )


async def test_deepseek_thinking_mode_answers_in_text() -> None:
    needs("DEEPSEEK_API_KEY")
    llm = DeepSeekLLM(thinking=True)
    request = LLMRequest.single(
        "Answer briefly.", "Name one risk of long-lived JWTs.", max_tokens=4000, effort="low"
    )
    response = await generate_text(llm, request)
    assert response.text.strip()
    assert response.usage.output_tokens > 0


async def test_openai_compatible_end_to_end() -> None:
    needs("WIDETHINK_LIVE_OPENAI_MODEL")
    model = os.environ["WIDETHINK_LIVE_OPENAI_MODEL"]
    llm = OpenAICompatibleLLM(model, base_url=os.environ.get("OPENAI_BASE_URL"))
    await check(Thinker(llm, config=CONFIG))
