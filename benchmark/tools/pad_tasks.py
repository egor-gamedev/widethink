"""Make long-context variants of benchmark tasks by burying their context in filler.

The question behind it: do models still notice hidden requirements when the
evidence is a few lines inside a large project, not the whole prompt? Filler is
real code from any directory you choose (your own projects - do not commit third
party code into the benchmark). The original items are scattered through the
middle of the filler, and filler items get kind "filler", which the judge skips.

    python benchmark/tools/pad_tasks.py --filler ../other-project --only hard- \
        --chars 220000 --out results/long-tasks

Output is JSON task files (git-ignored under results/) that `widethink bench run
--tasks` accepts.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from widethink.bench.judge import FILLER_KIND
from widethink.bench.task import BenchTask, load_tasks
from widethink.context import Context, ContextItem

EXCLUDE = {"_forgotten", ".claude", "node_modules", ".git", ".venv", "__pycache__"}


def filler_items(root: Path, chars: int, seed: int) -> list[ContextItem]:
    """Shuffled files of ``root`` totalling about ``chars`` characters, marked as filler."""
    items = list(Context.from_path(root, exclude_dirs=EXCLUDE, max_items=5000).items)
    random.Random(seed).shuffle(items)
    chosen: list[ContextItem] = []
    total = 0
    for item in items:
        if total >= chars:
            break
        chosen.append(
            item.model_copy(
                update={
                    "id": f"legacy/{item.id}",
                    "title": f"legacy/{item.title}",
                    "kind": FILLER_KIND,
                }
            )
        )
        total += len(item.content)
    return chosen


def pad(task: BenchTask, filler: list[ContextItem], seed: int) -> BenchTask:
    """Place the task's own items at random positions in the middle 60% of the filler."""
    rng = random.Random(f"{seed}:{task.id}")
    items = list(filler)
    low, high = len(items) // 5, max(len(items) // 5 + 1, len(items) * 4 // 5)
    for original in task.context:
        items.insert(rng.randint(low, high), original)
    return task.model_copy(
        update={"id": f"long-{task.id}", "title": f"{task.title} (long context)", "context": items}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", type=Path, default=Path("benchmark/tasks"))
    parser.add_argument("--only", action="append", default=[], help="task id substring")
    parser.add_argument("--filler", type=Path, required=True, help="directory with filler code")
    parser.add_argument("--chars", type=int, default=220_000, help="filler size per task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/long-tasks"))
    args = parser.parse_args()

    tasks = [
        t for t in load_tasks(args.tasks) if not args.only or any(o in t.id for o in args.only)
    ]
    filler = filler_items(args.filler, args.chars, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        long_task = pad(task, filler, args.seed)
        path = args.out / f"{long_task.id}.json"
        path.write_text(long_task.model_dump_json(indent=1), encoding="utf-8")
        size = sum(len(i.content) for i in long_task.context)
        print(f"{path}  items={len(long_task.context)}  chars={size:,}  (~{size // 3:,} tokens)")


if __name__ == "__main__":
    main()
