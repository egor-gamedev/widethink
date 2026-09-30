from __future__ import annotations

import pytest
from pydantic import ValidationError

from widethink import ABLATIONS, ThinkConfig
from widethink.config import ExpansionConfig


def test_defaults_come_from_the_brain() -> None:
    cfg = ThinkConfig()
    assert cfg.fade.factor == 0.86
    assert cfg.fade.min_strength == 0.22
    assert cfg.fade.max_depth == 7
    assert cfg.fatigue.satiated == 0.45
    assert cfg.fatigue.gain == pytest.approx(0.66)
    assert cfg.emergence.cued == 0.5
    assert cfg.reinstatement.items == 3
    assert cfg.expansion.max_candidates == 6
    assert cfg.parallel == 2


@pytest.mark.parametrize("mechanism", ABLATIONS)
def test_each_ablation_changes_the_config(mechanism: str) -> None:
    base = ThinkConfig()
    assert base.without(mechanism) != base


def test_ablations_compose() -> None:
    cfg = ThinkConfig().without("fatigue", "lottery", "parallel")
    assert not cfg.fatigue.enabled
    assert cfg.selection.greedy
    assert cfg.parallel == 1


def test_unknown_ablation_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown mechanisms"):
        ThinkConfig().without("telepathy")


def test_config_is_immutable_and_strict() -> None:
    cfg = ThinkConfig()
    with pytest.raises(ValidationError):
        cfg.parallel = 3  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ThinkConfig(unknown_option=1)  # type: ignore[call-arg]


def test_candidate_counts_are_validated() -> None:
    with pytest.raises(ValidationError, match="min_candidates"):
        ExpansionConfig(min_candidates=5, max_candidates=4)
    with pytest.raises(ValidationError, match="context_candidates"):
        ExpansionConfig(context_candidates=7)


def test_config_round_trips_through_json() -> None:
    cfg = ThinkConfig().without("capture")
    assert ThinkConfig.model_validate_json(cfg.model_dump_json()) == cfg
