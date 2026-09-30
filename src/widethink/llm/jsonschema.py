"""JSON schemas for structured output, and lenient JSON extraction.

Providers with native structured output (Anthropic ``output_config.format``,
OpenAI ``response_format``) accept only a subset of JSON Schema: every object
must set ``additionalProperties: false`` and list all properties as required,
and numeric or length constraints are not allowed. :func:`strict_json_schema`
converts a Pydantic model into that subset; the model itself still validates
(and clamps) values client-side after parsing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

_DROPPED_KEYWORDS = frozenset(
    {
        "default",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "pattern",
        "title",
        "uniqueItems",
    }
)
_NAME_MAPS = frozenset({"properties", "$defs", "definitions"})
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the strict structured-output schema of a Pydantic model.

    Nested models are inlined instead of referenced through ``$defs``: support
    for ``$ref`` varies between providers and local inference servers, and the
    output schemas used here are small and never recursive.
    """
    raw = model.model_json_schema()
    definitions = raw.pop("$defs", {})
    schema = _strictify(_inline(raw, definitions))
    if not isinstance(schema, dict):  # pragma: no cover - pydantic always returns an object
        raise TypeError("expected an object schema")
    return schema


def _inline(node: Any, definitions: dict[str, Any], depth: int = 0) -> Any:
    if depth > 32:
        raise ValueError("schema nests too deeply; recursive models are not supported")
    if isinstance(node, list):
        return [_inline(item, definitions, depth + 1) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        target = definitions[str(node["$ref"]).rsplit("/", 1)[-1]]
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        return {**_inline(target, definitions, depth + 1), **_inline(siblings, definitions, depth)}
    return {key: _inline(value, definitions, depth + 1) for key, value in node.items()}


def _strictify(node: Any, *, name_map: bool = False) -> Any:
    if isinstance(node, list):
        return [_strictify(item) for item in node]
    if not isinstance(node, dict):
        return node
    if name_map:
        # Keys of "properties"/"$defs" are names chosen by the model author and
        # may collide with keywords ("title", "default"): keep them all.
        return {key: _strictify(value) for key, value in node.items()}
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED_KEYWORDS:
            continue
        out[key] = _strictify(value, name_map=key in _NAME_MAPS)
    if out.get("type") == "object" or "properties" in out:
        properties = out.setdefault("properties", {})
        out["additionalProperties"] = False
        out["required"] = list(properties)
    return out


def extract_json(text: str) -> Any:
    """Parse JSON from model output that may be wrapped in prose or code fences.

    Raises:
        ValueError: if no JSON value can be found.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for match in _FENCE.finditer(stripped):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char in "{[":
            try:
                value, _ = decoder.raw_decode(stripped, index)
            except json.JSONDecodeError:
                continue
            return value
    raise ValueError("no JSON value found in model output")
