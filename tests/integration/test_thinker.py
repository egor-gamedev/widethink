"""The whole harness over a fake mind: behaviour, determinism, budgets and failure paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import BASELINE, SYNTHESIS, FakeMind, MakeThinker
from widethink import (
    ABLATIONS,
    Budget,
    Context,
    Event,
    Finding,
    FindingStore,
    ThinkConfig,
    Thinker,
    ThinkResult,
)
from widethink.config import EmergenceConfig, ValueConfig
from widethink.llm import LLMRequest, RecordingLLM, ReplayLLM, ScriptedLLM

TASK = "Add JWT authentication to our API"


def fingerprint(result: ThinkResult) -> list[tuple[Any, ...]]:
    return [(n.id, n.label, n.status, n.step, n.stream, n.why) for n in result.nodes]


def expansion_prompts(llm: ScriptedLLM) -> list[str]:
    return [r.messages[-1].content for r in llm.requests if r.purpose == "expand"]


async def test_a_run_produces_answer_tree_and_accounting(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    thinker, mind, _ = make_thinker()
    result = await thinker.athink(TASK, jwt_context, seed=1)

    assert result.answer == SYNTHESIS["answer"]
    assert result.deviations[0].instead == "long-lived device-bound tokens"
    assert result.confidence == "medium"
    assert result.baseline is not None
    assert result.baseline.approach == BASELINE["approach"]

    tree = result.tree()
    assert tree.root.status == "expanded"
    assert tree.root.step == 1
    standard = [n for n in result.nodes if n.origin == "baseline"]
    assert len(standard) == 1 + len(BASELINE["key_decisions"])

    assert result.stats["thoughts"] == 10
    assert result.meta.stop_reason == "max_thoughts"
    assert result.usage.calls == 1 + 10 + 1
    assert set(result.usage.by_purpose) == {"baseline", "expand", "synthesis"}
    assert result.usage.models == ["scripted"]
    assert mind.calls["expand"] == 10

    kinds = {event.type for event in result.events}
    assert {
        "run_started",
        "baseline",
        "selected",
        "expanded",
        "synthesized",
        "run_finished",
    } <= kinds
    assert result.meta.llm == "scripted"
    assert result.meta.seed == 1
    assert result.warnings == []


async def test_same_seed_same_tree_different_seed_different_tree(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    first = await make_thinker()[0].athink(TASK, jwt_context, seed=3)
    again = await make_thinker()[0].athink(TASK, jwt_context, seed=3)
    other = await make_thinker()[0].athink(TASK, jwt_context, seed=4)
    assert fingerprint(first) == fingerprint(again)
    assert fingerprint(first) != fingerprint(other)


async def test_parallel_streams_never_think_the_same_idea(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    thinker, _, _ = make_thinker(ThinkConfig(parallel=3, max_thoughts=18))
    result = await thinker.athink(TASK, jwt_context, seed=2)
    selected = [e.node_id for e in result.events if e.type == "selected"]
    assert len(selected) == len(set(selected)) == 18
    assert {n.stream for n in result.nodes if n.step is not None} == {0, 1, 2}


async def test_streams_see_the_standard_solution_as_already_considered(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    thinker, _, llm = make_thinker()
    await thinker.athink(TASK, jwt_context, seed=1)
    for prompt in expansion_prompts(llm):
        assert "- Fifteen minute access tokens" in prompt


async def test_context_details_cue_grounded_branches(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    thinker, _, _ = make_thinker()
    result = await thinker.athink(TASK, jwt_context, seed=1)
    cued = [n for n in result.nodes if n.origin == "cued"]
    assert cued, "the second stream starts from a detail of the context"
    for node in cued:
        assert node.kind == "context"
        assert node.grounding[0] in jwt_context.ids()
        assert node.evidence[0].source == node.grounding[0]  # reinstated first, then cited
        assert node.resolution == "supported"


async def test_unresolvable_context_becomes_a_question(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    thinker, _, llm = make_thinker(question_on="imply")
    result = await thinker.athink(TASK, jwt_context, seed=1)
    asked = [n for n in result.nodes if n.question]
    assert asked
    assert all(n.resolution == "unknown" for n in asked)
    assert result.stats["questions"] == len(asked)
    synthesis = next(r for r in llm.requests if r.purpose == "synthesis")
    assert asked[0].question in synthesis.messages[-1].content


async def test_a_surprise_captures_the_next_thought(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    thinker, _, _ = make_thinker(surprise_on="imply")
    result = await thinker.athink(TASK, jwt_context, seed=1)
    surprises = [n for n in result.nodes if n.kind == "surprise"]
    assert surprises
    first = surprises[0]
    parent = next(n for n in result.nodes if n.id == first.parent_id)
    assert first.origin == "captured"
    assert first.why == "captured by a surprise"
    assert first.stream == parent.stream
    later = [
        n.step
        for n in result.nodes
        if n.stream == parent.stream and n.step and n.step > parent.step
    ]  # type: ignore[operator]
    assert first.step == min(later)  # the very next thought of that stream
    assert result.stats["captured"] >= 1


async def test_repetition_prunes_duplicates_and_bores_the_branch(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    thinker, _, _ = make_thinker(ThinkConfig(max_thoughts=40), repeat=True)
    result = await thinker.athink(TASK, jwt_context, seed=1)
    pruned = [n for n in result.nodes if n.status == "pruned"]
    assert pruned
    assert all(n.duplicate_of for n in pruned)
    assert result.stats.get("closed_bored", 0) >= 1
    assert result.meta.stop_reason == "exhausted"  # nothing new left to think about


async def test_without_inhibition_duplicates_stay_in_the_tree(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    config = ThinkConfig(max_thoughts=6).without("inhibition")
    thinker, _, llm = make_thinker(config, repeat=True)
    result = await thinker.athink(TASK, jwt_context, seed=1)
    assert not [n for n in result.nodes if n.status == "pruned"]
    assert all("(nothing yet)" in prompt for prompt in expansion_prompts(llm))


async def test_budget_stops_thinking_but_keeps_the_synthesis(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    thinker, _, _ = make_thinker(ThinkConfig(max_thoughts=None))
    result = await thinker.athink(TASK, jwt_context, seed=1, budget=20_000)
    assert result.meta.stop_reason == "budget"
    assert result.answer == SYNTHESIS["answer"]
    assert result.usage.total.total_tokens <= 20_000
    assert result.stats["thoughts"] >= 2


async def test_call_budget_counts_the_reserved_synthesis(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    thinker, _, _ = make_thinker(ThinkConfig(max_thoughts=None, parallel=1))
    result = await thinker.athink(TASK, jwt_context, seed=1, budget=Budget(max_calls=5))
    assert result.usage.calls == 5  # baseline + 3 thoughts + synthesis
    assert result.stats["thoughts"] == 3


async def test_a_failed_thought_is_recorded_and_thinking_goes_on(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    thinker, _, _ = make_thinker(fail_on="imply")
    result = await thinker.athink(TASK, jwt_context, seed=1)
    failed = [n for n in result.nodes if n.status == "failed"]
    assert failed
    assert "did not match" in (failed[0].error or "") or "no JSON" in (failed[0].error or "")
    assert result.usage.failed_calls >= 2  # the attempt and its retry are both billed
    assert result.stats["thoughts"] == 10


async def test_synthesis_failure_falls_back_to_the_baseline(jwt_context: Context) -> None:
    mind = FakeMind()

    def responder(request: LLMRequest) -> Any:
        return "not json" if request.purpose == "synthesis" else mind(request)

    thinker = Thinker(ScriptedLLM(responder), config=ThinkConfig(max_thoughts=4))
    result = await thinker.athink(TASK, jwt_context, seed=1)
    assert result.answer == BASELINE["answer"]
    assert result.confidence is None
    assert any("synthesis failed" in w for w in result.warnings)


async def test_baseline_can_be_skipped(make_thinker: MakeThinker, jwt_context: Context) -> None:
    thinker, mind, _ = make_thinker(ThinkConfig(max_thoughts=4, baseline=False))
    result = await thinker.athink(TASK, jwt_context, seed=1)
    assert result.baseline is None
    assert mind.calls["baseline"] == 0
    assert not [n for n in result.nodes if n.origin == "baseline"]


@pytest.mark.parametrize("mechanism", ABLATIONS)
async def test_every_ablation_runs(
    mechanism: str, make_thinker: MakeThinker, jwt_context: Context
) -> None:
    thinker, _, _ = make_thinker(ThinkConfig(max_thoughts=6).without(mechanism))
    result = await thinker.athink(TASK, jwt_context, seed=1)
    assert result.answer
    assert result.stats["thoughts"] == 6


async def test_separate_critic_scores_the_proposals(
    make_thinker: MakeThinker, jwt_context: Context
) -> None:
    config = ThinkConfig(max_thoughts=4, value=ValueConfig(critic="separate"))
    thinker, mind, _ = make_thinker(config)
    result = await thinker.athink(TASK, jwt_context, seed=1)
    assert mind.calls["critic"] >= 1
    proposals = [n for n in result.nodes if n.origin == "proposal"]
    assert {n.value for n in proposals} <= {0.5, None}
    assert any(n.value == 0.5 for n in proposals)
    assert "critic" in result.usage.by_purpose


async def test_memory_stores_findings_and_brings_them_back(jwt_context: Context) -> None:
    store = FindingStore()
    thinker = Thinker(ScriptedLLM(FakeMind()), config=ThinkConfig(max_thoughts=6), memory=store)
    first = await thinker.athink(TASK, jwt_context, seed=1)
    remembered = {f.label for f in store.findings.values()}
    thought_labels = {n.label for n in first.nodes if n.status == "expanded"}
    assert remembered
    assert remembered <= thought_labels

    recall_store = FindingStore()
    await recall_store.remember([Finding(label="JWT authentication for offline technicians")])
    config = ThinkConfig(parallel=1, baseline=False, emergence=EmergenceConfig(cued=0.0))
    recaller = Thinker(ScriptedLLM(FakeMind(repeat=True)), config=config, memory=recall_store)
    result = await recaller.athink(TASK, seed=1)
    recalled = [n for n in result.nodes if n.origin == "recalled"]
    assert [n.label for n in recalled] == ["JWT authentication for offline technicians"]
    assert result.stats["recalled"] == 1


async def test_hooks_see_events_live_and_a_broken_hook_is_harmless(
    jwt_context: Context, caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[Event] = []

    def broken(_: Event) -> None:
        raise RuntimeError("observer bug")

    thinker = Thinker(
        ScriptedLLM(FakeMind()), config=ThinkConfig(max_thoughts=3), hooks=[seen.append, broken]
    )
    result = await thinker.athink(TASK, jwt_context, seed=1)
    assert seen == result.events
    assert "observer bug" in caplog.text


async def test_record_and_replay_reproduce_a_run(tmp_path: Path, jwt_context: Context) -> None:
    path = tmp_path / "run.recording.jsonl"
    config = ThinkConfig(max_thoughts=8, parallel=2)
    recorded = await Thinker(
        RecordingLLM(ScriptedLLM(FakeMind(surprise_on="imply")), path), config=config
    ).athink(TASK, jwt_context, seed=11)
    replayer = ReplayLLM(path)
    replayed = await Thinker(replayer, config=config).athink(TASK, jwt_context, seed=11)
    assert fingerprint(replayed) == fingerprint(recorded)
    assert replayed.answer == recorded.answer
    assert replayer.remaining() == 0


async def test_result_round_trips_through_json(
    tmp_path: Path, make_thinker: MakeThinker, jwt_context: Context
) -> None:
    result = await make_thinker()[0].athink(TASK, jwt_context, seed=1)
    loaded = ThinkResult.load(result.save(tmp_path / "result.json"))
    assert loaded == result
    assert loaded.render().startswith("[task]")
    assert loaded.render("mermaid").startswith("flowchart TD")


def test_sync_think_works_outside_an_event_loop(jwt_context: Context) -> None:
    thinker = Thinker(ScriptedLLM(FakeMind()), config=ThinkConfig(max_thoughts=2))
    assert thinker.think(TASK, jwt_context, seed=1).answer


async def test_sync_think_refuses_inside_an_event_loop() -> None:
    thinker = Thinker(ScriptedLLM(FakeMind()))
    with pytest.raises(RuntimeError, match="athink"):
        thinker.think(TASK)


async def test_empty_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        await Thinker(ScriptedLLM(FakeMind())).athink("   ")
