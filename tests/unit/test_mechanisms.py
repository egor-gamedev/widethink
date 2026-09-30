"""The brain mechanisms, one by one, against the behaviour the brain documents."""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from widethink.config import (
    CaptureConfig,
    ExpansionConfig,
    FadeConfig,
    FatigueConfig,
    ReinstatementConfig,
    ValueConfig,
)
from widethink.mechanisms import (
    Candidate,
    InhibitionOfReturn,
    Option,
    Source,
    SurpriseMonitor,
    child_strength,
    choose_emergence,
    draw,
    exhausted,
    mean_repetition,
    proposal_count,
    recall_weight,
    reinstated_items,
    repetition,
    satiated,
    score,
    too_deep,
    update_fatigue,
    value_pull,
)


def no_pull(_: float | None) -> float:
    return 1.0


# ----------------------------------------------------------------------------- selection


class TestSelection:
    def test_score_multiplies_link_freshness_value_and_grounding(self) -> None:
        candidate = Candidate("a", link=0.8, redundancy=0.25, value=None, grounded=True)
        assert score(candidate, pull=no_pull, grounded_bonus=1.2) == pytest.approx(0.8 * 0.75 * 1.2)

    def test_score_clamps_out_of_range_inputs(self) -> None:
        assert score(Candidate("a", link=3.0, redundancy=-1.0), pull=no_pull) == 1.0

    def test_draw_is_a_lottery_not_argmax(self) -> None:
        candidates = [Candidate("strong", link=0.9), Candidate("weak", link=0.45)]
        rng = random.Random(0)
        wins = Counter(draw(candidates, rng, pull=no_pull).key for _ in range(4000))  # type: ignore[union-attr]
        # squared weights: 0.81 vs 0.2025 -> the weak one still wins about 20% of draws
        assert 0.15 < wins["weak"] / 4000 < 0.25

    def test_greedy_ablation_always_takes_the_best(self) -> None:
        candidates = [Candidate("strong", link=0.9), Candidate("weak", link=0.45)]
        rng = random.Random(0)
        picks = {draw(candidates, rng, pull=no_pull, greedy=True).key for _ in range(50)}  # type: ignore[union-attr]
        assert picks == {"strong"}

    def test_draw_ignores_worthless_candidates(self) -> None:
        candidates = [Candidate("dead", link=0.0), Candidate("repeat", link=1.0, redundancy=1.0)]
        assert draw(candidates, random.Random(0), pull=no_pull) is None

    def test_draw_is_reproducible_for_a_seed(self) -> None:
        candidates = [Candidate(str(i), link=0.1 * i) for i in range(1, 10)]
        first = [draw(candidates, random.Random(5), pull=no_pull) for _ in range(3)]
        second = [draw(candidates, random.Random(5), pull=no_pull) for _ in range(3)]
        assert first == second


# ----------------------------------------------------------------------------- value


class TestValuePull:
    def test_good_beats_bad_about_twenty_to_one_after_squaring(self) -> None:
        cfg = ValueConfig()
        good, bad = value_pull(1.0, cfg) ** 2, value_pull(-1.0, cfg) ** 2
        # The brain documents about 20:1 in favour of good over bad at equal links.
        assert good / bad == pytest.approx(math.exp(3.0))
        assert 19 < good / bad < 21

    def test_unknown_or_disabled_value_is_neutral(self) -> None:
        assert value_pull(None, ValueConfig()) == 1.0
        assert value_pull(1.0, ValueConfig(enabled=False)) == 1.0

    def test_invalid_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="floor"):
            ValueConfig(floor=1.0, ceiling=0.5)


# ----------------------------------------------------------------------------- fatigue


class TestFatigue:
    def test_repetition_maps_similarity_between_thresholds(self) -> None:
        assert repetition(0.2, related=0.3, duplicate=0.8) == 0.0
        assert repetition(0.55, related=0.3, duplicate=0.8) == pytest.approx(0.5)
        assert repetition(0.95, related=0.3, duplicate=0.8) == 1.0

    def test_repetition_rejects_inverted_thresholds(self) -> None:
        with pytest.raises(ValueError, match="above"):
            repetition(0.5, related=0.8, duplicate=0.3)

    def test_one_fully_repetitive_thought_is_boring(self) -> None:
        cfg = FatigueConfig()
        # 0.12 per tick x 5.5 ticks per thought = 0.66 >= SATIATED 0.45
        assert satiated(update_fatigue(0.0, 1.0, cfg), cfg)

    def test_half_repetitive_thoughts_bore_on_the_third(self) -> None:
        cfg = FatigueConfig()
        fatigue = 0.0
        history = []
        for _ in range(3):
            fatigue = update_fatigue(fatigue, 0.5, cfg)
            history.append(satiated(fatigue, cfg))
        assert history == [False, False, True]

    def test_novelty_recovers_and_fatigue_is_bounded(self) -> None:
        cfg = FatigueConfig()
        assert update_fatigue(0.3, 0.0, cfg) == pytest.approx(0.0)
        assert update_fatigue(0.79, 1.0, cfg) == cfg.max_fatigue

    def test_disabled_fatigue_never_bores(self) -> None:
        cfg = FatigueConfig(enabled=False)
        assert update_fatigue(0.7, 1.0, cfg) == 0.0
        assert not satiated(0.99, cfg)

    def test_mean_repetition_of_nothing_is_zero(self) -> None:
        assert mean_repetition([]) == 0.0
        assert mean_repetition([0.0, 1.0]) == 0.5


# ----------------------------------------------------------------------------- fade


class TestFade:
    def test_strong_link_fades_by_the_brain_factor(self) -> None:
        assert child_strength(1.0, 0.9, FadeConfig()) == pytest.approx(0.86)

    def test_weak_link_fades_faster(self) -> None:
        assert child_strength(1.0, 0.2, FadeConfig()) == pytest.approx(0.86 * 0.6)

    def test_link_modulation_can_be_disabled(self) -> None:
        assert child_strength(1.0, 0.0, FadeConfig(link_modulation=None)) == pytest.approx(0.86)

    def test_exhaustion_and_depth_limits(self) -> None:
        cfg = FadeConfig()
        assert not exhausted(0.3, cfg)
        assert exhausted(0.25, cfg)  # 0.25 * 0.86 < 0.22
        assert too_deep(7, cfg)
        assert not too_deep(6, cfg)

    def test_budget_shrinks_with_strength(self) -> None:
        cfg = ExpansionConfig()
        assert proposal_count(1.0, cfg) == 6
        assert proposal_count(0.0, cfg) == 3
        assert reinstated_items(0.9, ReinstatementConfig()) == 3
        assert reinstated_items(0.3, ReinstatementConfig()) == 1


# ----------------------------------------------------------------------------- inhibition


class TestInhibitionOfReturn:
    def test_window_limits_how_long_a_thought_blocks(self) -> None:
        inhibition = InhibitionOfReturn(window=3)
        inhibition.visit("a", 1)
        assert inhibition.is_recent("a", 3)
        assert not inhibition.is_recent("a", 4)
        assert inhibition.recent(2) == ["a"]

    def test_session_window_blocks_forever(self) -> None:
        inhibition = InhibitionOfReturn(window=None)
        inhibition.visit("a", 1)
        assert inhibition.is_recent("a", 10_000)

    def test_revisiting_moves_a_key_to_the_end(self) -> None:
        inhibition = InhibitionOfReturn()
        for key, step in (("a", 1), ("b", 2), ("a", 3)):
            inhibition.visit(key, step)
        assert inhibition.visited() == ["b", "a"]

    def test_disabled_inhibition_blocks_nothing_but_remembers(self) -> None:
        inhibition = InhibitionOfReturn(enabled=False)
        inhibition.visit("a", 1)
        assert not inhibition.is_recent("a", 1)
        assert inhibition.recent(1) == []
        assert inhibition.was_visited("a")

    def test_window_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            InhibitionOfReturn(window=0)


# ----------------------------------------------------------------------------- capture


class TestCapture:
    def test_spike_above_baseline_captures(self) -> None:
        monitor = SurpriseMonitor(CaptureConfig())
        assert monitor.observe(0.9, step=1)
        assert monitor.captures == 1

    def test_habitual_surprise_stops_capturing(self) -> None:
        monitor = SurpriseMonitor(CaptureConfig(refractory=0))
        captured = [monitor.observe(0.5, step) for step in range(1, 30)]
        # the baseline catches up with a constant level: only the first ones capture
        assert captured[0]
        assert not any(captured[5:])

    def test_refractory_period_limits_capture_rate(self) -> None:
        monitor = SurpriseMonitor(CaptureConfig(baseline_rate=0.01))
        results = [monitor.observe(1.0, step) for step in range(1, 8)]
        assert results == [True, False, False, True, False, False, True]

    def test_disabled_capture_still_tracks_baseline(self) -> None:
        monitor = SurpriseMonitor(CaptureConfig(enabled=False))
        assert not monitor.observe(1.0, 1)
        assert monitor.baseline > 0

    def test_defaults_are_converted_from_ticks(self) -> None:
        cfg = CaptureConfig()
        assert cfg.refractory == 3  # 15 ticks / 5.5 ticks per thought
        assert cfg.baseline_rate == pytest.approx(1 - 0.95**5.5)


# ----------------------------------------------------------------------------- emergence


class TestEmergence:
    def test_only_cues_means_cued(self) -> None:
        choice = choose_emergence(
            random.Random(0), cued_share=0.5, cues=[Option("readme", 1.0)], recalls=[]
        )
        assert choice == (Source.CONTEXT, "readme")

    def test_only_recalls_means_recall(self) -> None:
        choice = choose_emergence(
            random.Random(0), cued_share=1.0, cues=[], recalls=[Option("n3", 1.0)]
        )
        assert choice == (Source.RECALL, "n3")

    def test_nothing_to_emerge(self) -> None:
        assert choose_emergence(random.Random(0), cued_share=0.5, cues=[], recalls=[]) is None

    def test_cued_share_splits_sources(self) -> None:
        rng = random.Random(1)
        sources = Counter(
            choose_emergence(rng, cued_share=0.5, cues=[Option("c", 1)], recalls=[Option("r", 1)])[
                0
            ]  # type: ignore[index]
            for _ in range(2000)
        )
        assert 0.45 < sources[Source.CONTEXT] / 2000 < 0.55

    def test_zero_weights_fall_back_to_uniform(self) -> None:
        choice = choose_emergence(
            random.Random(0), cued_share=1.0, cues=[Option("a", 0.0), Option("b", 0.0)], recalls=[]
        )
        assert choice is not None
        assert choice[1] in {"a", "b"}

    def test_recall_weight_follows_the_brain_formula(self) -> None:
        weight = recall_weight(0.5, age=30 / 5.5, pull=2.0, recency_scale=30 / 5.5)
        assert weight == pytest.approx((0.05 + 0.5 + 0.5) * 2.0)
