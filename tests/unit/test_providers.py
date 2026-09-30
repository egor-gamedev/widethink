"""Providers against fake SDK clients: what is sent, and how replies are normalized."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from widethink.errors import (
    RefusalError,
    ReplayMismatchError,
    StructuredOutputError,
    TruncatedOutputError,
)
from widethink.llm import (
    AnthropicLLM,
    LLMRequest,
    OpenAICompatibleLLM,
    RecordingLLM,
    ReplayLLM,
    ScriptedLLM,
    Usage,
    estimate_tokens,
    request_key,
)
from widethink.llm.anthropic import FALLBACK_BETA

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}


def request(**fields: Any) -> LLMRequest:
    defaults: dict[str, Any] = {"max_tokens": 500, "purpose": "expand"}
    return LLMRequest.single("rules", "focus", **{**defaults, **fields})


class Recorder:
    """Records the keyword arguments of ``create`` calls and returns a canned reply."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.reply


# ----------------------------------------------------------------------------- Anthropic


def anthropic_message(
    text: str = '{"a": 1}', stop: str = "end_turn", model: str = "claude-opus-5"
) -> Any:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=40, cache_read_input_tokens=900,
            cache_creation_input_tokens=50,
        ),
        model=model,
        stop_reason=stop,
        stop_details=SimpleNamespace(category="cyber") if stop == "refusal" else None,
    )  # fmt: skip


def anthropic_client(message: Any) -> tuple[Any, Recorder, Recorder]:
    stable, beta = Recorder(message), Recorder(message)
    client = SimpleNamespace(messages=stable, beta=SimpleNamespace(messages=beta))
    return client, stable, beta


class TestAnthropic:
    def test_params_are_minimal_and_cache_the_system_prompt(self) -> None:
        llm = AnthropicLLM(client=object())
        params = llm.build_params(request())
        assert params == {
            "model": "claude-opus-5",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": "focus"}],
            "system": [{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}],
        }
        assert "temperature" not in params  # current Claude models reject sampling params

    def test_params_carry_schema_effort_and_explicit_temperature(self) -> None:
        llm = AnthropicLLM("claude-sonnet-5", client=object())
        params = llm.build_params(
            request(json_schema=SCHEMA, effort="low", temperature=0.3, cache_system=False)
        )
        assert params["output_config"] == {
            "format": {"type": "json_schema", "schema": SCHEMA},
            "effort": "low",
        }
        assert params["temperature"] == 0.3
        assert "cache_control" not in params["system"][0]

    def test_empty_system_prompt_is_omitted(self) -> None:
        llm = AnthropicLLM(client=object())
        params = llm.build_params(LLMRequest.single("", "hi", max_tokens=10))
        assert "system" not in params

    async def test_fallbacks_use_the_beta_endpoint_and_usage_is_normalized(self) -> None:
        client, stable, beta = anthropic_client(anthropic_message())
        llm = AnthropicLLM(client=client)
        response = await llm.generate(request(json_schema=SCHEMA))
        assert not stable.calls
        assert beta.calls[0]["betas"] == [FALLBACK_BETA]
        assert beta.calls[0]["fallbacks"] == "default"
        assert response.data == {"a": 1}
        assert response.usage == Usage(
            input_tokens=1050, output_tokens=40, cache_read_tokens=900, cache_write_tokens=50
        )
        assert llm.name == "anthropic:claude-opus-5"

    async def test_without_fallbacks_the_stable_endpoint_is_used(self) -> None:
        client, stable, beta = anthropic_client(anthropic_message(text="plain"))
        response = await AnthropicLLM(client=client, fallbacks=False).generate(request())
        assert stable.calls
        assert not beta.calls
        assert response.data is None
        assert response.text == "plain"

    async def test_answering_model_is_reported_after_a_fallback(self) -> None:
        client, _, _ = anthropic_client(anthropic_message(model="claude-opus-4-8"))
        response = await AnthropicLLM(client=client).generate(request())
        assert response.model == "claude-opus-4-8"

    async def test_refusal_and_truncation_raise_with_usage(self) -> None:
        client, _, _ = anthropic_client(anthropic_message(stop="refusal"))
        with pytest.raises(RefusalError, match="cyber") as refused:
            await AnthropicLLM(client=client).generate(request())
        assert refused.value.usage is not None
        client, _, _ = anthropic_client(anthropic_message(stop="max_tokens"))
        with pytest.raises(TruncatedOutputError) as truncated:
            await AnthropicLLM(client=client).generate(request())
        assert truncated.value.usage.output_tokens == 40  # type: ignore[union-attr]

    async def test_invalid_json_leaves_data_empty(self) -> None:
        client, _, _ = anthropic_client(anthropic_message(text="{oops"))
        response = await AnthropicLLM(client=client).generate(request(json_schema=SCHEMA))
        assert response.data is None


# ----------------------------------------------------------------------------- OpenAI


def completion(content: str | None = '{"a": 2}', finish: str = "stop", refusal: Any = None) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=60,
            prompt_tokens_details=SimpleNamespace(cached_tokens=800),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=20),
        ),
        model="gpt-test",
    )


def openai_client(reply: Any) -> tuple[Any, Recorder]:
    recorder = Recorder(reply)
    return SimpleNamespace(chat=SimpleNamespace(completions=recorder)), recorder


class TestOpenAICompatible:
    def test_strict_json_schema_mode(self) -> None:
        llm = OpenAICompatibleLLM("gpt-test", client=object())
        params = llm.build_params(request(json_schema=SCHEMA, schema_name="Expansion Out!"))
        assert params["messages"][0] == {"role": "system", "content": "rules"}
        assert params["max_completion_tokens"] == 500
        assert params["response_format"]["json_schema"] == {
            "name": "Expansion_Out_",
            "schema": SCHEMA,
            "strict": True,
        }
        assert "reasoning_effort" not in params

    def test_json_object_mode_puts_the_schema_in_the_prompt(self) -> None:
        llm = OpenAICompatibleLLM(
            "llama", client=object(), base_url="http://localhost:8000/v1",
            structured="json_object", max_tokens_param="max_tokens", send_effort=True,
        )  # fmt: skip
        params = llm.build_params(request(json_schema=SCHEMA, effort="high", temperature=0.2))
        assert params["response_format"] == {"type": "json_object"}
        assert json.dumps(SCHEMA) in params["messages"][0]["content"]
        assert params["max_tokens"] == 500
        assert params["reasoning_effort"] == "high"
        assert params["temperature"] == 0.2
        assert llm.name == "openai-compatible:llama"

    def test_prompt_mode_sends_no_response_format(self) -> None:
        llm = OpenAICompatibleLLM("x", client=object(), structured="prompt")
        assert "response_format" not in llm.build_params(request(json_schema=SCHEMA))

    async def test_generate_parses_and_normalizes_usage(self) -> None:
        client, recorder = openai_client(completion('Result: {"a": 2}'))
        response = await OpenAICompatibleLLM("gpt-test", client=client).generate(
            request(json_schema=SCHEMA)
        )
        assert recorder.calls
        assert response.data == {"a": 2}
        assert response.usage == Usage(
            input_tokens=1000, output_tokens=60, cache_read_tokens=800, reasoning_tokens=20
        )

    async def test_generate_failure_modes(self) -> None:
        client, _ = openai_client(completion(finish="length"))
        with pytest.raises(TruncatedOutputError):
            await OpenAICompatibleLLM("m", client=client).generate(request())
        client, _ = openai_client(completion(content=None, refusal="I can't"))
        with pytest.raises(RefusalError, match="can't"):
            await OpenAICompatibleLLM("m", client=client).generate(request())
        client, _ = openai_client(completion(content="no json"))
        response = await OpenAICompatibleLLM("m", client=client).generate(
            request(json_schema=SCHEMA)
        )
        assert response.data is None


# ----------------------------------------------------------------------------- scripted


class TestScripted:
    async def test_replies_in_order_and_records_requests(self) -> None:
        llm = ScriptedLLM(replies=["first", {"a": 1}])
        assert (await llm.generate(request())).text == "first"
        second = await llm.generate(request(json_schema=SCHEMA))
        assert second.data == {"a": 1}
        assert len(llm.requests) == 2
        with pytest.raises(RuntimeError, match="ran out"):
            await llm.generate(request())

    async def test_responder_and_usage_estimate(self) -> None:
        llm = ScriptedLLM(lambda req: req.messages[-1].content.upper(), name="echo")
        response = await llm.generate(request())
        assert response.text == "FOCUS"
        assert response.model == "echo"
        assert response.usage.input_tokens == estimate_tokens("rulesfocus")

    def test_needs_something_to_say(self) -> None:
        with pytest.raises(ValueError, match="responder"):
            ScriptedLLM()


# ----------------------------------------------------------------------------- record & replay


class TestRecordReplay:
    async def test_replay_reproduces_responses_and_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "run.recording.jsonl"
        inner = ScriptedLLM(
            replies=['{"a": 1}', StructuredOutputError("bad", usage=Usage(output_tokens=7))]
        )
        recorder = RecordingLLM(inner, path)
        original = await recorder.generate(request(json_schema=SCHEMA))
        with pytest.raises(StructuredOutputError):
            await recorder.generate(request(purpose="critic"))
        assert recorder.name == "scripted"

        replay = ReplayLLM(path)
        assert replay.name == "scripted"
        assert replay.remaining() == 2
        assert await replay.generate(request(json_schema=SCHEMA)) == original
        with pytest.raises(StructuredOutputError) as caught:
            await replay.generate(request(purpose="critic"))
        assert caught.value.usage == Usage(output_tokens=7)
        assert replay.remaining() == 0

    async def test_unknown_request_is_a_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.recording.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ReplayMismatchError, match="differ"):
            await ReplayLLM(path, name="r").generate(request())

    def test_request_key_is_content_based(self) -> None:
        assert request_key(request()) == request_key(request())
        assert request_key(request()) != request_key(request(max_tokens=501))
