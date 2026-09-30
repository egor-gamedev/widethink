"""The command line, end to end, with model providers replaced by fakes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeMind
from widethink import ThinkResult, cli
from widethink.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from widethink.llm import LLMRequest, ScriptedLLM

TASKS_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "tasks"
TASK = "Add JWT authentication to our API"


@pytest.fixture(autouse=True)
def isolated_directory(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run every CLI test from an empty directory, so a developer's real .env is never read."""
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


def test_dotenv_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nWT_A=plain\nexport WT_B='quoted value'\nWT_C=\"double\"\n"
        "WT_EMPTY=\nnot a line\nWT_KEPT=from-file\n",
        encoding="utf-8",
    )
    for name in ("WT_A", "WT_B", "WT_C", "WT_EMPTY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WT_KEPT", "from-environment")
    assert cli.load_dotenv(env) == ["WT_A", "WT_B", "WT_C"]
    assert cli.os.environ["WT_B"] == "quoted value"
    assert cli.os.environ["WT_C"] == "double"
    assert cli.os.environ["WT_KEPT"] == "from-environment"  # the environment wins
    assert "WT_EMPTY" not in cli.os.environ
    assert cli.load_dotenv(tmp_path / "missing.env") == []
    for name in ("WT_A", "WT_B", "WT_C"):
        monkeypatch.delenv(name)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "README.md").write_text("Technicians work offline for days.\n", encoding="utf-8")
    (root / "models.py").write_text("class Device: pass\n", encoding="utf-8")
    return root


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> list[ScriptedLLM]:
    made: list[ScriptedLLM] = []
    mind = FakeMind()

    def respond(request: LLMRequest) -> Any:
        if request.purpose in ("answer", "sample"):
            return "per-customer cache keys"
        if request.purpose == "judge":
            return {"verdicts": [], "unsupported_claims": [], "kept_standard": True}
        return mind(request)

    def make(
        provider: str,
        model: str | None,
        base_url: str | None,
        structured: str | None,
        *,
        reasoning: bool = False,
    ) -> Any:
        suffix = "+thinking" if reasoning else ""
        llm = ScriptedLLM(respond, name=f"fake-{provider}:{model}{suffix}")
        made.append(llm)
        return llm

    monkeypatch.setattr(cli, "make_llm", make)
    return made


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.main(["--version"])
    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith("widethink ")


def test_think_prints_a_report_and_saves_the_result(
    fake_models: list[ScriptedLLM],
    project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "result.json"
    code = cli.main(
        ["think", TASK, "-c", str(project), "--note", "We have 40 technicians",
         "--max-thoughts", "3", "--seed", "1", "--out", str(out), "--without", "capture"]
    )  # fmt: skip
    captured = capsys.readouterr()
    assert code == 0
    for heading in ("=== Answer ===", "=== Departures", "=== Questions", "=== Thought tree ===",
                    "=== Usage ==="):  # fmt: skip
        assert heading in captured.out
    assert "→" in captured.err  # live progress
    result = ThinkResult.load(out)
    assert result.events[0].data["context_items"] == 3  # two files and the note
    grounded = {g for n in result.nodes for g in n.grounding}
    assert grounded
    assert grounded <= {"README.md", "models.py", "note-1"}
    assert not result.config.capture.enabled


def test_record_then_replay_gives_the_same_tree(
    fake_models: list[ScriptedLLM],
    project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recording = tmp_path / "calls.recording.jsonl"
    common = ["think", TASK, "-c", str(project), "--max-thoughts", "4", "--seed", "5", "-q"]
    assert cli.main([*common, "--record", str(recording), "--tree", "mermaid"]) == 0
    recorded = capsys.readouterr().out
    assert cli.main([*common, "--replay", str(recording), "--tree", "mermaid"]) == 0
    replayed = capsys.readouterr().out
    tree = recorded.split("=== Thought tree ===")[1].split("=== Usage ===")[0]
    assert tree.strip().startswith("flowchart TD")
    assert tree == replayed.split("=== Thought tree ===")[1].split("=== Usage ===")[0]


def test_render_saved_result(
    fake_models: list[ScriptedLLM],
    project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "result.json"
    cli.main(["think", TASK, "-c", str(project), "--max-thoughts", "2", "-q", "--out", str(out)])
    capsys.readouterr()
    assert cli.main(["render", str(out), "--format", "mermaid"]) == 0
    assert capsys.readouterr().out.startswith("flowchart TD")
    assert cli.main(["render", str(out), "--ascii"]) == 0
    plain = capsys.readouterr().out
    assert not set(plain) & set("├└│·✗⚡❓")  # glyphs are ASCII; labels stay as written
    assert "|-- " in plain


def test_bench_validate(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["bench", "validate", "--tasks", str(TASKS_DIR)]) == 0
    output = capsys.readouterr().out
    assert "tasks valid" in output
    assert "control" in output


def test_bench_run_report_and_show(
    fake_models: list[ScriptedLLM], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results.jsonl"
    trees = tmp_path / "trees"
    code = cli.main(
        ["bench", "run", "--tasks", str(TASKS_DIR), "--solver", "direct", "--solver", "broad",
         "--solver", "direct-high", "--solver", "widethink", "--provider", "deepseek",
         "--judge-provider", "deepseek", "--budget", "200000", "--out", str(results),
         "--save-trees", str(trees), "--concurrency", "2",
         "--price-input", "0.3", "--price-output", "1.2"]
    )  # fmt: skip
    assert code == 0
    table = capsys.readouterr().out
    for solver in ("broad", "direct", "direct-high", "widethink"):
        assert f"| {solver} |" in table
    assert "$" in table  # cost column filled from the given prices
    assert sorted(llm.name for llm in fake_models) == [
        "fake-deepseek:None",
        "fake-deepseek:None+thinking",  # the reasoning baseline gets a thinking client
        "fake-deepseek:deepseek-v4-pro",  # the judge defaults to the stronger model
    ]
    tasks = len(list(TASKS_DIR.glob("*.yaml")))
    assert len(results.read_text(encoding="utf-8").splitlines()) == 4 * tasks
    assert len(list(trees.glob("*__widethink__seed0.json"))) == tasks

    assert cli.main(["bench", "report", str(results), "--format", "json"]) == 0
    summaries = json.loads(capsys.readouterr().out)
    assert {s["solver"] for s in summaries} == {"direct", "broad", "direct-high", "widethink"}

    task = "auth-offline-field-app"
    assert cli.main(["bench", "show", str(results), "--task", task, "--answers"]) == 0
    shown = capsys.readouterr().out
    assert shown.count(f"=== {task} /") == 4
    assert "tree: widethink render" in shown
    assert "| per-customer cache keys" in shown
    assert cli.main(["bench", "show", str(results), "--task", "no-such-task"]) == 1


def test_errors_are_reported_not_raised(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["think", TASK, "--provider", "openai", "-q"]) == 1
    assert "--model is required" in capsys.readouterr().err


def test_embedder_factory() -> None:
    assert isinstance(cli.make_embedder("hashing", None, None), HashingEmbedder)
    local = cli.make_embedder("local", "some/model", None)
    assert isinstance(local, SentenceTransformerEmbedder)
    assert local.name == "st:some/model"
