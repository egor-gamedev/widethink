"""widethink - brain-inspired wide thinking for LLMs.

A harness between the user and a language model that decides what the model
thinks about at each step: it keeps a tree of ideas, goes deeper by weighted
lottery, returns up when a branch gets boring or weak, lets details of the
user's context cue new branches, and is captured by surprises. The goal is to
notice where the standard solution does not fit this particular user.

Quick start::

    from widethink import Thinker
    from widethink.llm import AnthropicLLM

    result = Thinker(AnthropicLLM()).think(
        "Add JWT authentication to our API", "./my-project", budget=60_000
    )
    print(result.answer)
    print(result.render())
"""

from widethink.__about__ import __version__
from widethink.budget import Budget, Pricing, UsageSummary
from widethink.config import ABLATIONS, ThinkConfig
from widethink.context import Context, ContextItem
from widethink.engine import Thinker
from widethink.events import Event
from widethink.memory import Finding, FindingStore
from widethink.result import Baseline, Deviation, Question, ThinkResult
from widethink.tree import ThoughtNode, ThoughtTree

__all__ = [
    "ABLATIONS",
    "Baseline",
    "Budget",
    "Context",
    "ContextItem",
    "Deviation",
    "Event",
    "Finding",
    "FindingStore",
    "Pricing",
    "Question",
    "ThinkConfig",
    "ThinkResult",
    "Thinker",
    "ThoughtNode",
    "ThoughtTree",
    "UsageSummary",
    "__version__",
]
