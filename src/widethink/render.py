"""Rendering the thought tree for people: console text and Mermaid flowcharts."""

from __future__ import annotations

from widethink.tree import ThoughtNode, ThoughtTree

_UNICODE = {"branch": "├── ", "last": "└── ", "pipe": "│   ", "gap": "    "}
_ASCII = {"branch": "|-- ", "last": "`-- ", "pipe": "|   ", "gap": "    "}
_MERMAID_LABEL = 70


def render_text(tree: ThoughtTree, *, ascii_only: bool = False) -> str:
    """Indented tree with what happened to every idea.

    ``#7`` is the thought's step, ``s0.74`` its strength, ``v+0.6`` the critic's
    value; a branch that ended says why (bored, exhausted, too deep, nowhere).
    """
    glyphs = _ASCII if ascii_only else _UNICODE
    lines = [_describe(tree.root, ascii_only)]

    def visit(node: ThoughtNode, prefix: str) -> None:
        children = tree.children(node.id)
        extra = (
            [f"? {node.question}" if ascii_only else f"❓ {node.question}"] if node.question else []
        )
        entries: list[ThoughtNode | str] = [*extra, *children]
        for position, entry in enumerate(entries):
            last = position == len(entries) - 1
            connector = glyphs["last"] if last else glyphs["branch"]
            if isinstance(entry, str):
                lines.append(prefix + connector + entry)
                continue
            lines.append(prefix + connector + _describe(entry, ascii_only))
            visit(entry, prefix + (glyphs["gap"] if last else glyphs["pipe"]))

    visit(tree.root, "")
    return "\n".join(lines)


def _describe(node: ThoughtNode, ascii_only: bool) -> str:
    head = f"[{node.kind}] {node.label}"
    notes: list[str] = []
    if node.origin == "baseline":
        notes.append("standard")
    elif node.status == "open":
        notes.append("unexplored")
    elif node.status == "pruned":
        notes.append(f"pruned: duplicate of {node.duplicate_of}")
    elif node.status == "failed":
        notes.append("failed")
    if node.step is not None:
        notes.append(f"#{node.step} {_why(node.why)}")
    if node.strength is not None and node.kind != "task":
        notes.append(f"s{node.strength:.2f}")
    if node.value is not None:
        notes.append(f"v{node.value:+.1f}")
    if node.resolution not in (None, "not_applicable"):
        notes.append(str(node.resolution))
    if node.kind == "surprise":
        notes.append("!" if ascii_only else "⚡")
    if node.closed_reason is not None:
        notes.append(("x " if ascii_only else "✗ ") + node.closed_reason.replace("_", " "))
    separator = " | " if ascii_only else " · "
    return head + ("   " + separator.strip() + " " + separator.join(notes) if notes else "")


def _why(why: str | None) -> str:
    if not why:
        return ""
    for short in ("association", "return", "cued", "captured", "recalled", "resurfaced"):
        if short in why:
            return short
    return "start" if why == "the task itself" else why.split(":")[0]


def render_mermaid(tree: ThoughtTree) -> str:
    """A Mermaid flowchart (renders natively on GitHub)."""
    lines = [
        "flowchart TD",
        "  classDef task fill:#1f2937,color:#fff,stroke:#111827;",
        "  classDef standard fill:#e5e7eb,color:#374151,stroke:#9ca3af;",
        "  classDef context fill:#dbeafe,color:#1e3a8a,stroke:#1d4ed8;",
        "  classDef solution fill:#dcfce7,color:#14532d,stroke:#15803d;",
        "  classDef risk fill:#fee2e2,color:#7f1d1d,stroke:#b91c1c;",
        "  classDef surprise fill:#fef3c7,color:#78350f,stroke:#b45309;",
        "  classDef recalled fill:#ede9fe,color:#4c1d95,stroke:#6d28d9;",
        "  classDef unexplored fill:#ffffff,color:#6b7280,stroke:#d1d5db,stroke-dasharray:4 3;",
        "  classDef pruned fill:#f9fafb,color:#9ca3af,stroke:#e5e7eb,stroke-dasharray:2 2;",
    ]
    for node in tree.walk():
        label = (
            node.label if len(node.label) <= _MERMAID_LABEL else node.label[:_MERMAID_LABEL] + "…"
        )
        prefix = f"#{node.step} " if node.step is not None else ""
        text = _mermaid_escape(f"{prefix}{label}")
        if node.question:
            text += "<br/>❓ " + _mermaid_escape(node.question[:_MERMAID_LABEL])
        lines.append(f'  {node.id}["{text}"]')
        if node.parent_id is not None:
            arrow = "-.->" if node.status in ("pruned", "open") else "-->"
            lines.append(f"  {node.parent_id} {arrow} {node.id}")
        lines.append(f"  class {node.id} {_mermaid_class(node)};")
    return "\n".join(lines)


def _mermaid_class(node: ThoughtNode) -> str:
    if node.kind == "task":
        return "task"
    if node.origin == "baseline":
        return "standard"
    if node.status == "pruned":
        return "pruned"
    if node.status == "open":
        return "unexplored"
    return node.kind


def _mermaid_escape(text: str) -> str:
    replacements = {'"': "#quot;", "<": "#lt;", ">": "#gt;", "\n": " ", "[": "(", "]": ")"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
