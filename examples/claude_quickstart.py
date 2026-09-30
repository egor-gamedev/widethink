"""Wide thinking with Claude on a project directory.

    pip install "widethink[anthropic]"
    export ANTHROPIC_API_KEY=...          # or `ant auth login`
    python examples/claude_quickstart.py "Add JWT auth to our API" ./my-project

Every model call is recorded to runs/, so the run can be replayed for free with
``widethink think ... --replay runs/<name>.recording.jsonl``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from widethink import Budget, Thinker
from widethink.llm import AnthropicLLM, RecordingLLM


def main() -> None:
    task, project = sys.argv[1], Path(sys.argv[2])
    runs = Path("runs")
    llm = RecordingLLM(AnthropicLLM(), runs / "quickstart.recording.jsonl")
    thinker = Thinker(llm, hooks=[lambda event: print(" ", event.type, event.message)])
    result = thinker.think(task, project, budget=Budget(max_tokens=80_000))

    print("\n=== Answer ===\n" + result.answer)
    for deviation in result.deviations:
        print(f"- instead of {deviation.standard}: {deviation.instead} ({deviation.because})")
    for question in result.questions:
        print("?", question.question)
    print("\n" + result.render())
    print(f"\n{result.usage.total.total_tokens:,} tokens in {result.usage.calls} calls")
    result.save(runs / "quickstart.json")


if __name__ == "__main__":
    main()
