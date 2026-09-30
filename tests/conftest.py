"""Shared fixtures: a deterministic "fake mind" that stands in for a model.

The fake mind reads the harness's prompts the way a model would (focus, the
requested number of proposals, the relevant context, the already-considered
list) and answers with valid JSON. Switches make it surprise, ask questions,
repeat itself or fail on chosen foci, so every mechanism can be exercised
offline and deterministically.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest

from widethink import Context, ContextItem, ThinkConfig, Thinker
from widethink.llm import LLMRequest, ScriptedLLM


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests that call real APIs unless explicitly enabled (they cost money)."""
    if os.environ.get("WIDETHINK_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live test: set WIDETHINK_LIVE_TESTS=1 to call real APIs")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


_FOCUS = re.compile(r"^FOCUS \[(\w+)\]: (.+)$", re.MULTILINE)
_COUNT = re.compile(r"propose (\d+)")
_ITEM = re.compile(r'<item id="([^"]+)"[^>]*>\n([^\n]*)')
_CONSIDERED = re.compile(r"ALREADY CONSIDERED - do not propose again:\n((?:- .+\n?)*)")
_SYLLABLES = [
    "ka",
    "zo",
    "mi",
    "ru",
    "te",
    "vo",
    "lin",
    "qua",
    "pe",
    "dax",
    "sho",
    "ul",
    "ne",
    "fy",
]


def pseudo_word(seed: str, syllables: int = 3) -> str:
    """A deterministic made-up word: distinct seeds give (almost) disjoint character n-grams."""
    digest = hashlib.sha256(seed.encode()).digest()
    return "".join(_SYLLABLES[b % len(_SYLLABLES)] for b in digest[:syllables])


BASELINE = {
    "approach": "Short-lived JWT access tokens with refresh token rotation",
    "key_decisions": ["Fifteen minute access tokens", "Refresh token rotation", "RS256 signing"],
    "answer": "Issue a 15-minute access JWT and a rotating refresh token.",
}

SYNTHESIS = {
    "answer": "Device-bound long-lived tokens with offline verification.",
    "deviations": [
        {
            "standard": "15-minute access tokens",
            "instead": "long-lived device-bound tokens",
            "because": "technicians are offline for days",
            "evidence_nodes": ["n5"],
        }
    ],
    "questions": [{"question": "How long can a device stay offline?", "why_it_matters": "TTL"}],
    "confidence": "medium",
}


class FakeMind:
    """Callable responder for :class:`ScriptedLLM` with switchable behaviour."""

    def __init__(
        self,
        *,
        surprise_on: str | None = None,
        question_on: str | None = None,
        fail_on: str | None = None,
        repeat: bool = False,
        proposal_kind: str = "solution",
    ) -> None:
        self.surprise_on = surprise_on
        self.question_on = question_on
        self.fail_on = fail_on
        self.repeat = repeat
        self.proposal_kind = proposal_kind
        self.calls: Counter[str] = Counter()
        self.foci: list[str] = []

    def __call__(self, request: LLMRequest) -> Any:
        self.calls[request.purpose] += 1
        if request.purpose == "baseline":
            return BASELINE
        if request.purpose == "synthesis":
            return SYNTHESIS
        if request.purpose == "critic":
            count = request.messages[-1].content.count("\n") + 1
            return {"scores": [{"index": i, "value": 0.5, "reason": "ok"} for i in range(1, count)]}
        return self._expand(request)

    def _expand(self, request: LLMRequest) -> Any:
        text = request.messages[0].content  # retries append a correction after it
        found = _FOCUS.search(text)
        assert found is not None, text
        kind, focus = found.groups()
        self.foci.append(focus)
        wanted = int(_COUNT.search(text).group(1))  # type: ignore[union-attr]
        if self.fail_on and self.fail_on in focus:
            return "this is not json at all"
        items = _ITEM.findall(text)
        evidence = [{"source": items[0][0], "quote": items[0][1][:40]}] if items else []
        number = self.calls["expand"]
        if self.repeat:
            considered = _CONSIDERED.search(text)
            labels = [line[2:] for line in (considered.group(1).splitlines() if considered else [])]
            proposals = [self._proposal(label, 0.9) for label in labels[:wanted]]
        else:
            proposals = [
                self._proposal(
                    f"{pseudo_word(f'a{number}-{i}')} {pseudo_word(f'b{number}-{i}')}",
                    0.9 - 0.1 * i,
                )
                for i in range(wanted)
            ]
        surprise = {"level": 0.0, "about": ""}
        if self.surprise_on and self.surprise_on in focus and kind != "surprise":
            surprise = {"level": 0.95, "about": f"Surprise {pseudo_word(focus, 4)} breaks it"}
        asked = self.question_on is not None and self.question_on in focus
        return {
            "elaboration": f"Thinking about {focus}.",
            "evidence": evidence,
            "resolution": "unknown" if asked else ("supported" if evidence else "not_applicable"),
            "question": f"What about {focus}?" if asked else "",
            "surprise": surprise,
            "next": proposals,
        }

    def _proposal(self, label: str, link: float) -> dict[str, Any]:
        return {
            "idea": label,
            "detail": f"Detail of {label}.",
            "kind": self.proposal_kind,
            "link": link,
            "value": 0.4,
            "grounding": [],
        }


@pytest.fixture
def jwt_context() -> Context:
    return Context.of(
        ContextItem(
            id="README.md",
            title="README.md",
            kind="doc",
            salience=0.7,
            content="FieldOps: technicians work in basements and stay offline for days.",
        ),
        ContextItem(
            id="app/models.py",
            title="app/models.py",
            kind="file",
            content="class Device(Base):\n    serial = Column(String)\n    owner_id = Column(Int)",
        ),
        ContextItem(
            id="docs/compliance.md",
            title="docs/compliance.md",
            kind="doc",
            salience=0.7,
            content="Access of a dismissed employee must be revoked within 1 hour.",
        ),
    )


MakeThinker = Callable[..., tuple[Thinker, FakeMind, ScriptedLLM]]


@pytest.fixture
def make_thinker() -> MakeThinker:
    """Build a Thinker over a FakeMind; keyword arguments go to FakeMind or ThinkConfig."""

    def build(
        config: ThinkConfig | None = None, **mind_options: Any
    ) -> tuple[Thinker, FakeMind, ScriptedLLM]:
        mind = FakeMind(**mind_options)
        llm = ScriptedLLM(mind)
        return Thinker(llm, config=config or ThinkConfig(max_thoughts=10)), mind, llm

    return build
