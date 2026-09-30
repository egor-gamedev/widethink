"""Offline demo: the JWT scenario from the project plan, without any API key.

A scripted "mind" stands in for the model: it answers each thinking step from
canned knowledge about one project (FieldOps, whose technicians work offline
for days on shared tablets). Everything else is the real harness - the baseline,
the lottery, satiation, inhibition of return, context cueing, capture by
surprise, questions for the user and the synthesis.

    python examples/offline_demo.py
    python examples/offline_demo.py --mermaid   # the tree as a Mermaid chart

The replies are canned, so the *content* is only as clever as this script; the
point is to watch how the harness steers thinking away from the template.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any

from widethink import Context, ContextItem, ThinkConfig, Thinker
from widethink.cli import describe
from widethink.llm import LLMRequest, ScriptedLLM

TASK = "Add JWT-based authentication to our API. Right now every endpoint is open."

CONTEXT = Context.of(
    ContextItem(
        id="README.md",
        title="README.md",
        kind="doc",
        salience=0.7,
        content=(
            "FieldOps is used by maintenance technicians of water utilities. Most sites have "
            "no mobile coverage, and a typical inspection trip lasts two to four days. The app "
            "stores reports locally and syncs them when the tablet is back in coverage."
        ),
    ),
    ContextItem(
        id="app/models.py",
        title="app/models.py",
        kind="file",
        content=(
            "class Tablet(Base):\n"
            "    serial = Column(String, unique=True)\n"
            "    assigned_to = Column(ForeignKey('technicians.id'), nullable=True)\n"
            "    # tablets are pooled: a technician takes any free tablet at the depot\n"
        ),
    ),
    ContextItem(
        id="docs/security-policy.md",
        title="docs/security-policy.md",
        kind="doc",
        salience=0.7,
        content=(
            "When a contractor's engagement ends, their access to operational data must be "
            "revoked within one hour, including data already downloaded to devices."
        ),
    ),
)

_FOCUS = re.compile(r"^FOCUS \[(\w+)\]: (.+)$", re.MULTILINE)
_COUNT = re.compile(r"propose (\d+)")


def idea(
    text: str, detail: str, kind: str, link: float, value: float, *grounding: str
) -> dict[str, Any]:
    return {"idea": text, "detail": detail, "kind": kind, "link": link, "value": value,
            "grounding": list(grounding)}  # fmt: skip


def step(
    elaboration: str,
    proposals: list[dict[str, Any]],
    *,
    evidence: tuple[str, str] | None = None,
    resolution: str = "not_applicable",
    question: str = "",
    surprise: tuple[float, str] = (0.0, ""),
) -> dict[str, Any]:
    return {
        "elaboration": elaboration,
        "evidence": [{"source": evidence[0], "quote": evidence[1]}] if evidence else [],
        "resolution": resolution,
        "question": question,
        "surprise": {"level": surprise[0], "about": surprise[1]},
        "next": proposals,
    }


# What the scripted mind "knows", keyed by phrases that appear in the focus. Branches
# cued by a context item ('What does "README.md" imply...') reach the same insight as
# the matching question - and the harness prunes the repeated proposals.
KNOWLEDGE: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    (
        ("Add JWT",),
        step(
            "The request is standard: protect the API with JWTs. Before choosing token lifetimes "
            "it is worth checking how and where this API is actually used.",
            [
                idea(
                    "Short-lived access tokens with refresh token rotation",
                    "The usual 15-minute access token plus rotating refresh token.",
                    "solution",
                    0.9,
                    0.3,
                ),
                idea(
                    "How do technicians reach the API in the field?",
                    "Connectivity decides whether tokens can be refreshed.",
                    "context",
                    0.8,
                    0.8,
                    "README.md",
                ),
                idea(
                    "Are tablets personal or shared?",
                    "Shared devices change what a token should be bound to.",
                    "context",
                    0.7,
                    0.7,
                    "app/models.py",
                ),
                idea(
                    "What are the rules for ending access?",
                    "Revocation requirements constrain stateless tokens.",
                    "context",
                    0.7,
                    0.7,
                    "docs/security-policy.md",
                ),
                idea(
                    "Asymmetric signing so the app can verify tokens itself",
                    "RS256/EdDSA lets a device check a token without calling the server.",
                    "solution",
                    0.6,
                    0.5,
                ),
            ],
        ),
    ),
    (
        ("reach the API", '"README.md"'),
        step(
            "Technicians spend two to four days without coverage; a token that needs the network "
            "to refresh every 15 minutes would lock them out for the whole trip.",
            [
                idea(
                    "Offline session bound to technician and tablet",
                    "A long-lived signed session usable offline, verified locally.",
                    "solution",
                    0.9,
                    0.9,
                    "README.md",
                ),
                idea(
                    "How long must offline access last?",
                    "The trip length sets the offline grace period.",
                    "context",
                    0.8,
                    0.8,
                    "README.md",
                ),
            ],
            evidence=("README.md", "a typical inspection trip lasts two to four days"),
            resolution="supported",
            surprise=(0.9, "15-minute tokens would lock technicians out for their whole trip"),
        ),
    ),
    (
        ("lock technicians out",),
        step(
            "The standard refresh flow cannot work offline, so the token lifetime must cover "
            "a whole trip - which collides with fast revocation.",
            [
                idea(
                    "Offline grace period with re-validation on sync",
                    "Work offline for a bounded time; re-check everything when back online.",
                    "solution",
                    0.9,
                    0.9,
                ),
                idea(
                    "Encrypt local data with a key that expires",
                    "Downloaded data becomes unreadable when the key lapses.",
                    "solution",
                    0.7,
                    0.8,
                ),
            ],
            resolution="unknown",
            question="What is the longest trip without coverage that the app must support?",
        ),
    ),
    (
        ("personal or shared", '"app/models.py"'),
        step(
            "Tablets are pooled at the depot, so a token bound to the device alone would let "
            "the next technician act as the previous one.",
            [
                idea(
                    "Bind sessions to the technician, sign in on each check-out",
                    "Each check-out starts a new session; local data is per technician.",
                    "solution",
                    0.9,
                    0.8,
                    "app/models.py",
                ),
                idea(
                    "Wipe local data on check-in",
                    "Returning a tablet clears the previous technician's data.",
                    "solution",
                    0.7,
                    0.6,
                ),
            ],
            evidence=("app/models.py", "tablets are pooled"),
            resolution="supported",
        ),
    ),
    (
        ("ending access", '"docs/security-policy.md"'),
        step(
            "Access must be revoked within one hour, including data on devices. Stateless JWTs "
            "cannot be revoked early, and an offline tablet cannot be reached at all.",
            [
                idea(
                    "Revocation list pushed to tablets on reconnect",
                    "Revoked sessions are rejected and wiped when a tablet syncs.",
                    "solution",
                    0.9,
                    0.8,
                ),
                idea(
                    "Choose between uninterrupted offline work and one-hour revocation",
                    "The two requirements conflict for tablets that are offline.",
                    "context",
                    0.8,
                    0.9,
                    "docs/security-policy.md",
                ),
            ],
            evidence=("docs/security-policy.md", "revoked within one hour"),
            resolution="supported",
            question="For a tablet that is offline, may revocation wait until it reconnects?",
            surprise=(0.6, "Revocation within an hour is impossible for tablets that are offline"),
        ),
    ),
]


def respond(request: LLMRequest) -> Any:
    if request.purpose == "baseline":
        return {
            "approach": "Short-lived JWT access tokens with rotating refresh tokens",
            "key_decisions": ["15-minute access tokens", "Refresh token rotation",
                              "RS256 signatures", "Tokens stored in secure storage"],
            "answer": "Issue a 15-minute RS256 access token and a rotating refresh token...",
        }  # fmt: skip
    if request.purpose == "synthesis":
        return SYNTHESIS
    text = request.messages[0].content
    focus = _FOCUS.search(text)
    wanted = int(_COUNT.search(text).group(1))  # type: ignore[union-attr]
    label = focus.group(2) if focus else ""
    for keys, reply in KNOWLEDGE:
        if any(key in label for key in keys):
            return reply | {"next": reply["next"][:wanted]}
    return step(f"{label}: a refinement with nothing new for this project.", [])


SYNTHESIS = {
    "answer": (
        "Use JWTs, but not the standard short-lived scheme. Technicians sign in on every tablet "
        "check-out and receive a signed offline session (EdDSA) valid for the trip plus a grace "
        "period; the app verifies it locally. Local data is encrypted with a key tied to that "
        "session and wiped on check-in. When a tablet reconnects it syncs a revocation list and "
        "re-validates the session; online requests use normal short-lived access tokens."
    ),
    "deviations": [
        {"standard": "15-minute access tokens refreshed over the network",
         "instead": "a trip-long offline session verified on the tablet",
         "because": "technicians are without coverage for two to four days",
         "evidence_nodes": []},
        {"standard": "tokens bound to the device",
         "instead": "sessions bound to the technician, renewed on each check-out",
         "because": "tablets are pooled between technicians", "evidence_nodes": []},
        {"standard": "revocation by waiting for expiry",
         "instead": "revocation list and remote wipe on reconnect",
         "because": "access must end within one hour, including data on devices",
         "evidence_nodes": []},
    ],
    "questions": [
        {"question": "What is the longest trip without coverage?",
         "why_it_matters": "It sets the offline session lifetime."},
        {"question": "For an offline tablet, may revocation wait until it reconnects?",
         "why_it_matters": "Otherwise offline work and one-hour revocation cannot both hold."},
    ],
    "confidence": "medium",
}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mermaid", action="store_true", help="print the tree as Mermaid")
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    thinker = Thinker(
        ScriptedLLM(respond, name="scripted-demo"), config=ThinkConfig(max_thoughts=12)
    )
    result = thinker.think(TASK, CONTEXT, seed=args.seed)
    print(describe(result, tree="mermaid" if args.mermaid else "text"))


if __name__ == "__main__":
    main()
