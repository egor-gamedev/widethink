"""Strict schemas, lenient parsing and the retry policy of structured calls."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from widethink.errors import (
    RefusalError,
    StructuredOutputError,
    TransientLLMError,
    TruncatedOutputError,
)
from widethink.llm import LLMRequest, ScriptedLLM, Usage, generate_structured, generate_text
from widethink.llm.jsonschema import extract_json, schema_example, strict_json_schema
from widethink.schemas import ExpansionOut, SynthesisOut


class Inner(BaseModel):
    title: str = Field(description="kept despite the keyword-like name")
    score: float = Field(ge=0, le=1)


class Outer(BaseModel):
    name: str = Field(min_length=1, max_length=5, default="x")
    items: list[Inner]
    note: str | None


def walk(node: Any) -> list[dict[str, Any]]:
    """Every schema node; the keys of a "properties" map are names, not keywords."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        found.append(node)
        for key, value in node.items():
            if key == "properties":
                for child in value.values():
                    found.extend(walk(child))
            else:
                found.extend(walk(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(walk(value))
    return found


class TestStrictSchema:
    def test_every_object_is_closed_and_fully_required(self) -> None:
        schema = strict_json_schema(Outer)
        objects = [node for node in walk(schema) if node.get("type") == "object"]
        assert len(objects) == 2
        for node in objects:
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])

    def test_unsupported_keywords_are_dropped_but_property_names_kept(self) -> None:
        schema = strict_json_schema(Outer)
        keys = {key for node in walk(schema) for key in node}
        assert not keys & {"minimum", "maximum", "minLength", "maxLength", "default", "title"}
        inner = schema["properties"]["items"]["items"]
        assert "title" in inner["properties"]
        assert inner["properties"]["title"]["description"] == "kept despite the keyword-like name"

    def test_nested_models_are_inlined(self) -> None:
        schema = strict_json_schema(Outer)
        assert "$defs" not in schema
        assert not any("$ref" in node for node in walk(schema))

    @pytest.mark.parametrize("model", [ExpansionOut, SynthesisOut])
    def test_real_schemas_are_strict(self, model: type[BaseModel]) -> None:
        schema = strict_json_schema(model)
        for node in walk(schema):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False


class TestExtractJson:
    def test_plain(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self) -> None:
        assert extract_json('Here:\n```json\n{"a": [1, 2]}\n```\nDone.') == {"a": [1, 2]}

    def test_embedded_in_prose(self) -> None:
        assert extract_json('Sure! {"a": {"b": "}"}} Hope it helps.') == {"a": {"b": "}"}}

    def test_skips_broken_candidates(self) -> None:
        assert extract_json('{broken {"ok": true}') == {"ok": True}

    def test_nothing_found(self) -> None:
        with pytest.raises(ValueError, match="no JSON"):
            extract_json("no json here")


class Answer(BaseModel):
    value: int


REQUEST = LLMRequest.single("system", "user", max_tokens=100, purpose="test")


def attempts() -> tuple[list[tuple[Usage, str, bool]], Any]:
    seen: list[tuple[Usage, str, bool]] = []
    return seen, lambda usage, model, ok: seen.append((usage, model, ok))


async def test_valid_reply_parses_and_sends_the_schema() -> None:
    llm = ScriptedLLM(replies=['{"value": 3}'])
    parsed, response = await generate_structured(llm, REQUEST, Answer)
    assert parsed.value == 3
    assert llm.requests[0].json_schema is not None
    assert llm.requests[0].schema_name == "Answer"
    assert response.usage.output_tokens > 0


async def test_invalid_reply_is_retried_with_the_error_shown() -> None:
    llm = ScriptedLLM(replies=["not json", '{"value": 5}'])
    seen, hook = attempts()
    parsed, _ = await generate_structured(llm, REQUEST, Answer, retries=1, on_attempt=hook)
    assert parsed.value == 5
    retry = llm.requests[1]
    assert [m.role for m in retry.messages] == ["user", "assistant", "user"]
    assert "did not match" in retry.messages[-1].content
    assert [ok for _, _, ok in seen] == [False, True]  # the failed attempt is billed too


async def test_exhausted_retries_raise_with_usage() -> None:
    llm = ScriptedLLM(replies=['{"value": "x"}', '{"wrong": 1}'])
    with pytest.raises(StructuredOutputError) as caught:
        await generate_structured(llm, REQUEST, Answer, retries=1)
    assert caught.value.usage is not None


async def test_truncation_doubles_max_tokens() -> None:
    truncated = TruncatedOutputError("cut", usage=Usage(output_tokens=100), model="m")
    llm = ScriptedLLM(replies=[truncated, '{"value": 1}'])
    seen, hook = attempts()
    await generate_structured(llm, REQUEST, Answer, on_attempt=hook)
    assert [r.max_tokens for r in llm.requests] == [100, 200]
    assert seen[0] == (Usage(output_tokens=100), "m", False)


async def test_last_truncation_is_raised() -> None:
    llm = ScriptedLLM(replies=[TruncatedOutputError("cut"), TruncatedOutputError("cut again")])
    with pytest.raises(TruncatedOutputError, match="again"):
        await generate_structured(llm, REQUEST, Answer, retries=1)


async def test_refusals_are_not_retried() -> None:
    llm = ScriptedLLM(replies=[RefusalError("no"), '{"value": 1}'])
    with pytest.raises(RefusalError):
        await generate_structured(llm, REQUEST, Answer, retries=3)
    assert len(llm.requests) == 1


async def test_transient_failures_are_retried_as_they_were() -> None:
    busy = TransientLLMError("busy", usage=Usage(input_tokens=5), model="m")
    llm = ScriptedLLM(replies=[busy, '{"value": 2}'])
    seen, hook = attempts()
    parsed, _ = await generate_structured(llm, REQUEST, Answer, on_attempt=hook)
    assert parsed.value == 2
    assert llm.requests[0] == llm.requests[1]  # same request, nothing to correct
    assert [ok for _, _, ok in seen] == [False, True]
    with pytest.raises(TransientLLMError):
        await generate_structured(ScriptedLLM(replies=[busy, busy]), REQUEST, Answer)


async def test_generate_text_retries_truncation_and_transient_failures() -> None:
    llm = ScriptedLLM(replies=[TruncatedOutputError("cut"), "full answer"])
    seen, hook = attempts()
    response = await generate_text(llm, REQUEST, on_attempt=hook)
    assert response.text == "full answer"
    assert llm.requests[0].json_schema is None
    assert [r.max_tokens for r in llm.requests] == [100, 200]
    assert [ok for _, _, ok in seen] == [False, True]
    llm = ScriptedLLM(replies=[TransientLLMError("busy"), "ok"])
    assert (await generate_text(llm, REQUEST)).text == "ok"
    with pytest.raises(TruncatedOutputError):
        await generate_text(ScriptedLLM(replies=[TruncatedOutputError("a")] * 2), REQUEST)


def test_schema_example_has_the_shape_of_the_schema() -> None:
    example = schema_example(strict_json_schema(ExpansionOut))
    assert set(example) == set(ExpansionOut.model_fields)
    assert example["resolution"] == "supported"  # first enum value
    assert example["surprise"] == {"level": 0.0, "about": "..."}
    assert set(example["next"][0]) == {"idea", "detail", "kind", "link", "value", "grounding"}
    assert example["next"][0]["grounding"] == ["..."]
    assert schema_example({"anyOf": [{"type": "null"}, {"type": "integer"}]}) == 0
    assert schema_example({"type": ["null", "boolean"]}) is False
    assert schema_example({"const": "x"}) == "x"
    assert schema_example({}) is None
