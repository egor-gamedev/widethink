from __future__ import annotations

from widethink import ThoughtTree
from widethink.render import render_mermaid, render_text


def sample() -> ThoughtTree:
    tree = ThoughtTree("Add JWT auth")
    tree.root.status = "expanded"
    tree.root.step, tree.root.why = 1, "the task itself"
    standard = tree.add(
        "n0", kind="solution", origin="baseline", label="Standard", status="expanded"
    )
    tree.add(
        standard.id, kind="solution", origin="baseline", label="15 min tokens", status="expanded"
    )
    cued = tree.add("n0", kind="context", origin="cued", label='Offline "field" work', value=0.8)
    cued.status, cued.step, cued.why, cued.strength = "expanded", 2, "cued by [README]", 1.0
    cued.resolution, cued.question = "unknown", "How long offline?"
    cued.closed_reason = "bored"
    surprise = tree.add(cued.id, kind="surprise", origin="captured", label="Tokens expire offline")
    surprise.status, surprise.step, surprise.why = "expanded", 3, "captured by a surprise"
    tree.add(
        cued.id, kind="solution", origin="proposal", label="dup", status="pruned"
    ).duplicate_of = "n2"
    tree.add(cued.id, kind="risk", origin="proposal", label="later")
    return tree


def test_text_render_explains_every_node() -> None:
    text = render_text(sample())
    lines = text.splitlines()
    assert lines[0].startswith("[task] Add JWT auth")
    assert "standard" in lines[1]
    assert "#2 cued" in text
    assert "v+0.8" in text
    assert "✗ bored" in text
    assert "❓ How long offline?" in text
    assert "⚡" in text
    assert "pruned: duplicate of n2" in text
    assert "unexplored" in text
    assert "├── " in text
    assert "└── " in text


def test_ascii_render_is_plain() -> None:
    text = render_text(sample(), ascii_only=True)
    assert text.isascii()
    assert "x bored" in text
    assert "? How long offline?" in text


def test_mermaid_render() -> None:
    chart = render_mermaid(sample())
    assert chart.startswith("flowchart TD")
    assert 'n3["#2 Offline #quot;field#quot; work<br/>❓ How long offline?"]' in chart
    assert "n0 --> n1" in chart
    assert "n3 -.-> n5" in chart  # pruned and unexplored ideas hang by dotted lines
    assert "n3 -.-> n6" in chart
    assert "class n1 standard;" in chart
    assert "class n5 pruned;" in chart
    assert "class n6 unexplored;" in chart
    assert "class n4 surprise;" in chart


def test_synthesis_separates_evidence_from_hypotheses() -> None:
    from widethink.config import ReinstatementConfig
    from widethink.prompts import is_evidenced, synthesis_user
    from widethink.tree import Evidence

    tree = ThoughtTree("task")
    proven = tree.add("n0", kind="context", origin="proposal", label="Offline for days")
    proven.resolution, proven.evidence = "supported", [Evidence(source="README", quote="days")]
    guess = tree.add("n0", kind="context", origin="proposal", label="Email may be shared")
    guess.resolution = "unknown"
    assert is_evidenced(proven)
    assert not is_evidenced(guess)
    prompt = synthesis_user(
        task="t", approach=None, decisions=[], findings=[proven, guess], questions=[],
        relevant=[], cfg=ReinstatementConfig(),
    )  # fmt: skip
    evidenced, hypotheses = prompt.split("UNVERIFIED HYPOTHESES")
    assert "Offline for days" in evidenced
    assert "Email may be shared" in hypotheses
    empty = synthesis_user(
        task="t", approach=None, decisions=[], findings=[guess], questions=[], relevant=[],
        cfg=ReinstatementConfig(),
    )  # fmt: skip
    assert "(none - keep the standard solution)" in empty
