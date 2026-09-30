"""Command-line interface.

    widethink think "Add JWT auth to our API" -c ./my-project --budget 60000
    widethink render result.json --format mermaid
    widethink bench validate --tasks benchmark/tasks
    widethink bench run --tasks benchmark/tasks --solver direct --solver widethink \\
        --budget 60000 --judge-provider openai --judge-model <model> --out results.jsonl
    widethink bench report results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from widethink.__about__ import __version__
from widethink.budget import Budget
from widethink.config import ABLATIONS, ThinkConfig
from widethink.context import Context, ContextItem
from widethink.embeddings.base import Embedder
from widethink.errors import WideThinkError
from widethink.events import Event
from widethink.llm.base import LLM
from widethink.result import ThinkResult

SOLVERS = ("direct", "direct-high", "broad", "best-of-n", "widethink")
_MARKS = {
    "baseline": "≡",
    "selected": "→",
    "captured": "⚡",
    "question": "?",
    "closed": "✗",
    "failed": "!",
    "stopped": "■",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``widethink`` command."""
    _utf8_console()
    args = build_parser().parse_args(argv)
    try:
        code: int = args.handler(args)
    except (WideThinkError, ValueError, OSError) as error:
        print(f"widethink: error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="widethink",
        description="Brain-inspired wide thinking for LLMs.",
    )
    parser.add_argument("--version", action="version", version=f"widethink {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    think = commands.add_parser("think", help="think about a task in the light of a project")
    think.add_argument("task", help="what the user asks for")
    think.add_argument(
        "-c",
        "--context",
        action="append",
        default=[],
        metavar="PATH",
        help="project directory or file to use as context (repeatable)",
    )
    think.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="TEXT",
        help="something the user said about their situation (repeatable)",
    )
    _provider_arguments(think)
    think.add_argument("--embedder", choices=("hashing", "openai", "local"), default="hashing")
    think.add_argument("--embedding-model", metavar="NAME")
    think.add_argument("--budget", type=int, metavar="TOKENS", help="total token limit")
    think.add_argument("--max-thoughts", type=int, default=24, metavar="N")
    think.add_argument("--parallel", type=int, default=2, metavar="N")
    think.add_argument("--seed", type=int)
    think.add_argument(
        "--without",
        action="append",
        default=[],
        choices=ABLATIONS,
        help="switch a mechanism off (repeatable)",
    )
    think.add_argument("--out", type=Path, metavar="FILE", help="save the full result as JSON")
    think.add_argument(
        "--tree",
        choices=("text", "mermaid", "none"),
        default="text",
        help="how to print the thought tree",
    )
    recording = think.add_mutually_exclusive_group()
    recording.add_argument(
        "--record", type=Path, metavar="FILE", help="record every model call to a JSONL file"
    )
    recording.add_argument(
        "--replay",
        type=Path,
        metavar="FILE",
        help="answer model calls from a recording instead of the API",
    )
    think.add_argument("-q", "--quiet", action="store_true", help="no live progress")
    think.set_defaults(handler=cmd_think)

    render = commands.add_parser("render", help="print the thought tree of a saved result")
    render.add_argument("result", type=Path)
    render.add_argument("--format", choices=("text", "mermaid"), default="text")
    render.add_argument("--ascii", action="store_true", help="plain ASCII tree")
    render.set_defaults(handler=cmd_render)

    bench = commands.add_parser("bench", help="the Hidden Requirements benchmark")
    bench_commands = bench.add_subparsers(dest="bench_command", required=True)

    validate = bench_commands.add_parser("validate", help="check task files")
    validate.add_argument("--tasks", type=Path, default=Path("benchmark/tasks"))
    validate.set_defaults(handler=cmd_bench_validate)

    run = bench_commands.add_parser("run", help="run solvers at an equal budget and grade them")
    run.add_argument("--tasks", type=Path, default=Path("benchmark/tasks"))
    run.add_argument("--solver", action="append", choices=SOLVERS, required=True)
    _provider_arguments(run)
    run.add_argument("--judge-provider", choices=("anthropic", "openai"), required=True)
    run.add_argument("--judge-model")
    run.add_argument("--judge-base-url")
    run.add_argument("--budget", type=int, required=True, metavar="TOKENS")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="FILE",
        help="JSONL file; an interrupted run resumes from it",
    )
    run.set_defaults(handler=cmd_bench_run)

    report = bench_commands.add_parser("report", help="summarize graded runs")
    report.add_argument("results", type=Path)
    report.add_argument("--format", choices=("markdown", "json"), default="markdown")
    report.set_defaults(handler=cmd_bench_report)
    return parser


# --------------------------------------------------------------------------- commands


def cmd_think(args: argparse.Namespace) -> int:
    from widethink.engine import Thinker
    from widethink.llm.recording import RecordingLLM, ReplayLLM

    context = Context()
    for path in args.context:
        context = context + Context.from_path(path)
    for number, note in enumerate(args.note, start=1):
        context = context + Context.of(
            ContextItem(
                id=f"note-{number}", title=f"Note {number}", content=note, kind="user", salience=0.9
            )
        )
    llm: LLM
    if args.replay is not None:
        llm = ReplayLLM(args.replay)
    else:
        llm = make_llm(args.provider, args.model, args.base_url, args.structured)
        if args.record is not None:
            llm = RecordingLLM(llm, args.record)
    config = ThinkConfig(max_thoughts=args.max_thoughts, parallel=args.parallel)
    thinker = Thinker(
        llm,
        embedder=make_embedder(args.embedder, args.embedding_model, args.base_url),
        config=config.without(*args.without),
        hooks=[] if args.quiet else [_progress],
    )
    result = thinker.think(args.task, context, budget=args.budget, seed=args.seed)
    if args.out is not None:
        result.save(args.out)
    print(describe(result, tree=args.tree))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    result = ThinkResult.load(args.result)
    print(result.render(args.format, ascii_only=args.ascii))
    return 0


def cmd_bench_validate(args: argparse.Namespace) -> int:
    from widethink.bench.task import CANARY, load_tasks

    tasks = load_tasks(args.tasks)
    for task in tasks:
        kind = "control" if task.control else f"{len(task.hidden_requirements)} hidden"
        canary = "" if task.canary == CANARY else "  (missing canary!)"
        print(f"{task.id:40} {task.domain:22} {kind}{canary}")
    missing = sum(1 for task in tasks if task.canary != CANARY)
    controls = sum(1 for task in tasks if task.control)
    print(f"\n{len(tasks)} tasks valid ({controls} controls).")
    return 1 if missing else 0


def cmd_bench_run(args: argparse.Namespace) -> int:
    from widethink.bench import (
        BestOfNSolver,
        BroadPromptSolver,
        DirectSolver,
        LLMJudge,
        RunRecord,
        Solver,
        WideThinkSolver,
        load_tasks,
        run_benchmark,
        summarize,
        to_markdown,
    )

    llm = make_llm(args.provider, args.model, args.base_url, args.structured)
    judge = LLMJudge(make_llm(args.judge_provider, args.judge_model, args.judge_base_url, None))
    builders: dict[str, Solver] = {
        "direct": DirectSolver(llm),
        "direct-high": DirectSolver(llm, effort="high"),
        "broad": BroadPromptSolver(llm),
        "best-of-n": BestOfNSolver(llm),
        "widethink": WideThinkSolver(llm),
    }
    solvers = [builders[name] for name in dict.fromkeys(args.solver)]

    def progress(record: RunRecord) -> None:
        status = "error: " + record.error if record.error else "ok"
        print(f"  {record.task_id} / {record.solver} #{record.repeat}: {status}", file=sys.stderr)

    records = asyncio.run(
        run_benchmark(
            load_tasks(args.tasks),
            solvers,
            judge=judge,
            budget=Budget(max_tokens=args.budget),
            repeats=args.repeats,
            seed=args.seed,
            concurrency=args.concurrency,
            out=args.out,
            on_record=progress,
        )
    )
    print(to_markdown(summarize(records)))
    return 0


def cmd_bench_report(args: argparse.Namespace) -> int:
    from widethink.bench import load_records, summarize, to_markdown

    summaries = summarize(load_records(args.results))
    if args.format == "json":
        print(json.dumps([s.model_dump() for s in summaries], indent=2))
    else:
        print(to_markdown(summaries))
    return 0


# --------------------------------------------------------------------------- helpers


def make_llm(provider: str, model: str | None, base_url: str | None, structured: str | None) -> LLM:
    """Build a provider from command-line options."""
    if provider == "anthropic":
        from widethink.llm.anthropic import DEFAULT_MODEL, AnthropicLLM

        return AnthropicLLM(model or DEFAULT_MODEL)
    if provider == "openai":
        from widethink.llm.openai import OpenAICompatibleLLM

        if not model:
            raise ValueError("--model is required for the openai provider")
        return OpenAICompatibleLLM(
            model,
            base_url=base_url,
            structured=structured or "json_schema",  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown provider {provider!r}")


def make_embedder(kind: str, model: str | None, base_url: str | None) -> Embedder:
    """Build an embedder from command-line options."""
    if kind == "openai":
        from widethink.embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder(model or "text-embedding-3-small", base_url=base_url)
    if kind == "local":
        from widethink.embeddings.sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model) if model else SentenceTransformerEmbedder()
    from widethink.embeddings.hashing import HashingEmbedder

    return HashingEmbedder()


def describe(result: ThinkResult, *, tree: str = "text") -> str:
    """Human-readable report of a run."""
    sections = [f"=== Answer ===\n{result.answer.strip() or '(no answer)'}"]
    if result.deviations:
        sections.append(
            "=== Departures from the standard solution ===\n"
            + "\n".join(
                f"- instead of {d.standard}: {d.instead}\n  because {d.because}"
                + (f" [{', '.join(d.evidence_nodes)}]" if d.evidence_nodes else "")
                for d in result.deviations
            )
        )
    if result.questions:
        sections.append(
            "=== Questions for you ===\n"
            + "\n".join(
                f"- {q.question}" + (f"\n  ({q.why_it_matters})" if q.why_it_matters else "")
                for q in result.questions
            )
        )
    if tree != "none":
        sections.append(
            "=== Thought tree ===\n" + result.render("mermaid" if tree == "mermaid" else "text")
        )
    usage = result.usage
    total = usage.total
    sections.append(
        "=== Usage ===\n"
        f"{usage.calls} calls ({usage.failed_calls} failed), {total.total_tokens:,} tokens "
        f"(input {total.input_tokens:,}, of them cached {total.cache_read_tokens:,}; "
        f"output {total.output_tokens:,}); {result.stats.get('thoughts', 0)} thoughts; "
        f"stopped: {result.meta.stop_reason}; seed {result.meta.seed}"
    )
    sections.extend(f"warning: {warning}" for warning in result.warnings)
    return "\n\n".join(sections)


def _provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("anthropic", "openai"), default="anthropic")
    parser.add_argument("--model", help="model id (default for anthropic: claude-opus-5)")
    parser.add_argument(
        "--base-url", help="OpenAI-compatible server, e.g. http://localhost:8000/v1"
    )
    parser.add_argument(
        "--structured",
        choices=("json_schema", "json_object", "prompt"),
        help="structured-output mode for OpenAI-compatible servers",
    )


def _progress(event: Event) -> None:
    mark = _MARKS.get(event.type)
    if mark is None:
        return
    step = f"{event.step:>3}" if event.step is not None else "  -"
    stream = f"s{event.stream}" if event.stream is not None else "  "
    print(f"  [{step} {stream}] {mark} {event.message}", file=sys.stderr, flush=True)


def _utf8_console() -> None:
    """Windows consoles may default to a legacy code page; trees use Unicode glyphs."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
