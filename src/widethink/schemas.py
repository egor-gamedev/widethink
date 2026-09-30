"""What the model returns at each kind of step.

These models are converted to strict JSON schemas for providers with native
structured output. Validators are deliberately forgiving (clamping numbers,
normalizing case) because open models served without schema enforcement
return slightly irregular JSON; the harness should degrade, not crash.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ProposalKind = Literal["solution", "context", "risk"]
ResolutionOut = Literal["supported", "contradicted", "unknown", "not_applicable"]
Confidence = Literal["low", "medium", "high"]


class _Out(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _number(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return min(high, max(low, number))


def _choice(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return text if text in allowed else default


class ProposalOut(_Out):
    idea: str = Field(description="The next idea as a short label, at most 15 words.")
    detail: str = Field(description="One sentence: what exactly the idea means here.")
    kind: ProposalKind = Field(
        description=(
            "solution: a way to do the task; context: a question or hypothesis about "
            "what is special about this user or project; risk: a way the usual approach "
            "could fail for them."
        )
    )
    link: float = Field(description="How directly the idea follows from the focus, 0 to 1.")
    value: float = Field(
        description="How much the idea matters for THIS user's needs, -1 (does not fit) to 1."
    )
    grounding: list[str] = Field(
        description="Ids of the context items the idea is based on; empty if none."
    )

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, value: Any) -> str:
        return _choice(value, ("solution", "context", "risk"), "solution")

    @field_validator("link", mode="before")
    @classmethod
    def _link(cls, value: Any) -> float:
        return _number(value, 0.0, 1.0, 0.5)

    @field_validator("value", mode="before")
    @classmethod
    def _value(cls, value: Any) -> float:
        return _number(value, -1.0, 1.0, 0.0)


class EvidenceOut(_Out):
    source: str = Field(description="Id of a context item.")
    quote: str = Field(description="A short exact quote from that item.")


class SurpriseOut(_Out):
    level: float = Field(
        description="0: nothing unexpected. 1: a finding that breaks the usual approach for them."
    )
    about: str = Field(description="What was unexpected, in one sentence; empty if level is 0.")

    @field_validator("level", mode="before")
    @classmethod
    def _level(cls, value: Any) -> float:
        return _number(value, 0.0, 1.0, 0.0)


class ExpansionOut(_Out):
    """One thought about the focus idea, and where to go next."""

    elaboration: str = Field(
        description="The thought itself: the focus developed in light of the task and context."
    )
    evidence: list[EvidenceOut] = Field(description="Quotes from the user's context.")
    resolution: ResolutionOut = Field(
        description=(
            "Does the user's context support the focus (supported), contradict it "
            "(contradicted), leave it open (unknown), or is it not a claim about the user "
            "(not_applicable)?"
        )
    )
    question: str = Field(
        description=(
            "Only if resolution is unknown and the answer would change the solution: one "
            "concrete question for the user. Otherwise an empty string."
        )
    )
    surprise: SurpriseOut
    next: list[ProposalOut] = Field(description="Where thinking could go next.")

    @field_validator("resolution", mode="before")
    @classmethod
    def _resolution(cls, value: Any) -> str:
        allowed = ("supported", "contradicted", "unknown", "not_applicable")
        return _choice(value, allowed, "unknown")


class CriticScoreOut(_Out):
    index: int = Field(description="Number of the idea as listed.")
    value: float = Field(description="-1 (wrong or irrelevant for this user) to 1 (crucial).")
    reason: str = Field(description="One short sentence.")

    @field_validator("value", mode="before")
    @classmethod
    def _value(cls, value: Any) -> float:
        return _number(value, -1.0, 1.0, 0.0)


class CriticOut(_Out):
    scores: list[CriticScoreOut]


class BaselineOut(_Out):
    """The answer a model gives without any widening - the anchor of the tree."""

    approach: str = Field(description="The approach in one or two sentences.")
    key_decisions: list[str] = Field(
        description="The 3-7 key design decisions of the approach, each a short phrase."
    )
    answer: str = Field(description="The complete answer to the task.")


class DeviationOut(_Out):
    standard: str = Field(description="What the standard solution does.")
    instead: str = Field(description="What this solution does instead.")
    because: str = Field(description="Why, grounded in the findings.")
    evidence_nodes: list[str] = Field(description="Ids of the findings (like n12) behind it.")


class QuestionOut(_Out):
    question: str
    why_it_matters: str = Field(description="How the answer would change the solution.")


class SynthesisOut(_Out):
    answer: str = Field(description="The complete final answer for this user.")
    deviations: list[DeviationOut] = Field(description="Every departure from the standard.")
    questions: list[QuestionOut] = Field(description="Open questions for the user.")
    confidence: Confidence

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, value: Any) -> str:
        return _choice(value, ("low", "medium", "high"), "medium")
