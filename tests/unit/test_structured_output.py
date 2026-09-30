"""Strict schemas, lenient parsing and the retry policy of structured calls."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from widethink.errors import RefusalError, StructuredOutputError, TruncatedOutputError
from widethink.llm import LLMRequest, ScriptedLLM, Usage, generate_structured
from widethink.llm.jsonschema import extract_json, strict_json_schema
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
