# 3. Provider-neutral interface with native structured output

Date: 2026-09-30 · Status: Accepted

## Context

The harness needs machine-readable answers at every step (proposals with link
strengths and values, evidence, surprise). Providers offer different mechanisms:
Anthropic's `output_config.format` with a JSON schema, OpenAI's strict
`response_format`, and local servers with partial or no support. Prompt-only JSON
is fragile.

## Decision

- One interface, `LLM.generate(LLMRequest) -> LLMResponse`, with the schema as a
  plain JSON schema in the request.
- Pydantic models define the outputs; `strict_json_schema` converts them to the
  subset providers accept (closed objects, all properties required, nested models
  inlined, unsupported constraints removed) and the models validate and clamp
  client-side.
- Each provider uses its native mechanism; servers without it get the schema in
  the prompt and a lenient parser.
- One retry with the validation error shown to the model; truncation retries with
  a doubled limit; every attempt is billed.
- Sampling parameters are sent only when explicitly requested, because current
  Claude models reject them.

## Consequences

- The thinking logic never touches an SDK; tests run against deterministic fakes.
- Adding a provider is one class.
- Validators must stay forgiving (clamping, case normalization) so that open
  models degrade gracefully instead of failing steps.
