"""Prompt templates.

Prompts are versioned: :data:`PROMPTS_VERSION` is stored with every result, and
any change to the text below must bump it, because recorded runs and benchmark
numbers are only comparable under identical prompts.

Layout matters for cost. The system prompt of a thinking step holds everything
that is constant during a run (rules, task, context index), so providers can
cache it; the user message holds what changes from step to step (the focus,
its path, the reinstated context, the already-considered list).
"""

from __future__ import annotations

from collections.abc import Sequence

from widethink.config import ReinstatementConfig
from widethink.context import Context, ContextItem, render_item
from widethink.tree import ThoughtNode

PROMPTS_VERSION = "2026-09-30.1"

EXPANSION_RULES = """\
You are one step of a "wide thinking" process driven by a harness. The harness \
decides what to think about; you think about exactly one idea - the FOCUS - and \
propose where thinking could go next.

The purpose of the whole process is to notice where the standard, most typical \
solution does not fit this particular user. Such hidden requirements live in the \
user's context: their code and data model, documents, deployment, users and \
constraints, and the exact wording of the request.

How to think in this step:
- Stay on the FOCUS. Develop it in light of the task and the user's context; do not \
solve the whole task.
- Use the user's context as evidence: cite items by id with short exact quotes. Never \
invent facts about the user; say what is unknown instead.
- resolution: does the context support the focus (supported), contradict it \
(contradicted), leave it open (unknown), or is the focus not a claim about the user \
(not_applicable)?
- question: only when resolution is unknown and the answer would change the solution - \
one concrete question the user can answer. Otherwise an empty string.
- surprise: keep the level near 0 normally. Raise it only for a genuinely unexpected \
finding: a contradiction between the context and the usual approach, a fact in the \
code that breaks a common assumption, a requirement the context implies but nobody \
stated.
- next: propose continuations (each step says how many). Mix deeper or alternative \
solutions with ideas of kind "context" that ask what is non-standard about THIS user \
or project. Each is a short label plus one sentence.
- link: how directly a continuation follows from the focus (0..1); tangents get low links.
- value: how much a continuation matters for this user's actual needs (-1..1), negative \
when it probably does not fit them. Judge fit to this user, not general merit.
- Never propose anything from ALREADY CONSIDERED, not even reworded. If only \
repetitions come to mind, propose fewer.
- Write all free text in the language of the task."""

BASELINE_SYSTEM = "You are an expert assistant. Answer the user's task as you normally would."

CRITIC_SYSTEM = """\
You are the critic of a thinking process. For every idea, estimate how much it \
matters for THIS user's actual needs, given the task and their context: -1 means \
likely wrong or irrelevant for them, 0 neutral or unknown, 1 crucial for them. Judge \
fit to this user, not general merit, and be sparing with high scores."""

SYNTHESIS_SYSTEM = """\
You write the final answer of a wide-thinking process. You receive the task, the \
STANDARD solution a typical model gives, the findings of the thinking process (ideas \
with evidence from the user's context, contradictions, surprises and open questions, \
each with an id like n12) and the relevant parts of the user's context.

Write the best solution for THIS user:
- Keep the standard solution wherever the findings do not justify a change; deviate \
only where evidence from the user's context supports it.
- For every deviation state what the standard solution does, what you do instead and \
why, citing the ids of the findings.
- Put open questions whose answers would change the solution into questions, saying \
why they matter; where the solution depends on an answer, state the default you chose \
meanwhile.
- Do not invent requirements the findings do not support.
- Write in the language of the task."""


# --------------------------------------------------------------------------- thinking step


def expansion_system(task: str, context: Context, cfg: ReinstatementConfig) -> str:
    """The run-constant part of every thinking step."""
    parts = [
        EXPANSION_RULES,
        f"TASK:\n{task}",
        f"USER CONTEXT - index:\n{context.digest(cfg.digest_items)}",
    ]
    if cfg.mode == "full" and context.items:
        parts.append("USER CONTEXT - full text:\n" + render_context(context.items, cfg.full_chars))
    return "\n\n".join(parts)


def expansion_user(
    *,
    path: Sequence[ThoughtNode],
    focus: ThoughtNode,
    relevant: Sequence[ContextItem],
    cfg: ReinstatementConfig,
    considered: Sequence[str],
    proposals: int,
    context_proposals: int,
    detailed: bool,
) -> str:
    """The step-specific part: focus, path, reinstated context, what not to repeat."""
    lines = [f"{index}. [{node.kind}] {node.label}" for index, node in enumerate(path[:-1])]
    blocks = [
        "PATH - how thinking arrived here:\n"
        + ("\n".join(lines) if lines else "(this is the start)"),
        f"FOCUS [{focus.kind}]: {focus.label}" + (f"\n{focus.detail}" if focus.detail else ""),
    ]
    if focus.why:
        blocks[-1] += f"\nReached by: {focus.why}"
    if cfg.mode == "full":
        if relevant:
            ids = ", ".join(item.id for item in relevant)
            blocks.append(f"MOST RELEVANT CONTEXT ITEMS (full text above): {ids}")
    elif relevant:
        blocks.append("RELEVANT CONTEXT:\n" + render_context(relevant, cfg.item_chars))
    blocks.append(
        "ALREADY CONSIDERED - do not propose again:\n"
        + ("\n".join(f"- {label}" for label in considered) if considered else "(nothing yet)")
    )
    context_ask = (
        f', at least {context_proposals} of them of kind "context"' if context_proposals else ""
    )
    if focus.kind == "task":
        blocks.append(
            "This is the start: the focus is the task itself. Say briefly what it really asks "
            f"for, then propose {proposals} first directions{context_ask}. Include the "
            "standard approach unless it is already considered."
        )
    else:
        length = "3-5 sentences" if detailed else "1-2 sentences"
        blocks.append(
            f"Think about the focus in {length}. Then propose {proposals} "
            f"continuations{context_ask}."
        )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- other steps


def baseline_user(task: str, context: Context, cfg: ReinstatementConfig) -> str:
    rendered = render_context(context.items, cfg.full_chars) if context.items else "(none)"
    return (
        f"TASK:\n{task}\n\nCONTEXT:\n{rendered}\n\n"
        "Give your answer. Also state your approach in one or two sentences and list its "
        "key design decisions."
    )


def critic_user(
    task: str,
    focus: ThoughtNode,
    relevant: Sequence[ContextItem],
    ideas: Sequence[ThoughtNode],
    cfg: ReinstatementConfig,
) -> str:
    listed = "\n".join(
        f"{index}. [{idea.kind}] {idea.label} - {idea.detail}"
        for index, idea in enumerate(ideas, 1)
    )
    context = render_context(relevant, cfg.item_chars) if relevant else "(none)"
    return (
        f"TASK:\n{task}\n\nRELEVANT CONTEXT:\n{context}\n\n"
        f"IDEAS (proposed while thinking about: {focus.label}):\n{listed}\n\n"
        "Score every idea by its number."
    )


def synthesis_user(
    *,
    task: str,
    approach: str | None,
    decisions: Sequence[str],
    findings: Sequence[ThoughtNode],
    questions: Sequence[ThoughtNode],
    relevant: Sequence[ContextItem],
    cfg: ReinstatementConfig,
) -> str:
    if approach is None:
        standard = "(no standard solution was produced for this run)"
    else:
        standard = f"Approach: {approach}\nKey decisions:\n" + "\n".join(
            f"- {decision}" for decision in decisions
        )
    finding_lines = [_finding(node) for node in findings] or ["(no findings)"]
    question_lines = [f"- [{node.id}] {node.question}" for node in questions] or ["(none)"]
    context = render_context(relevant, cfg.item_chars) if relevant else "(none)"
    return "\n\n".join(
        [
            f"TASK:\n{task}",
            f"STANDARD SOLUTION (what a typical model answers):\n{standard}",
            "FINDINGS OF THE THINKING PROCESS:\n" + "\n".join(finding_lines),
            "OPEN QUESTIONS RAISED:\n" + "\n".join(question_lines),
            f"RELEVANT CONTEXT:\n{context}",
            "Write the final answer.",
        ]
    )


def render_context(items: Sequence[ContextItem], max_chars: int) -> str:
    return "\n\n".join(render_item(item, max_chars) for item in items)


def _finding(node: ThoughtNode) -> str:
    facts: list[str] = [node.kind, node.resolution or "unresolved"]
    if node.value is not None:
        facts.append(f"value {node.value:+.1f}")
    if node.surprise:
        facts.append(f"surprise {node.surprise:.1f}")
    lines = [f"[{node.id}] {' · '.join(facts)} · {node.label}"]
    if node.elaboration:
        lines.append(f"  {node.elaboration}")
    lines.extend(f'  Evidence [{e.source}]: "{e.quote}"' for e in node.evidence[:3])
    return "\n".join(lines)
