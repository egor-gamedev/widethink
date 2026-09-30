"""Configuration of the thinking process.

Every default is taken from the thought stream of the digital brain
(``digital_brain/neurons/thought.py`` and the branching mock-up
``runtime/thinking.py``); ``docs/mechanisms.md`` gives the derivation of each one.

The brain measures time in *ticks* of 0.1 s, and one thought dwells for
``DWELL = (3, 8)`` ticks. The harness measures time in *thoughts* (one model
call about one idea), so per-tick constants are converted with the mean dwell
:data:`TICKS_PER_THOUGHT`.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Mean length of one thought in brain ticks: DWELL = (3, 8) -> 5.5.
TICKS_PER_THOUGHT = 5.5

Effort = Literal["low", "medium", "high"]

#: Names accepted by :meth:`ThinkConfig.without` for ablation studies.
ABLATIONS = (
    "baseline",
    "capture",
    "cueing",
    "fade",
    "fatigue",
    "grounding",
    "inhibition",
    "lottery",
    "parallel",
    "value",
)


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SelectionConfig(_Section):
    """How the next thought is drawn among candidates (brain: ``_associate``)."""

    greedy: bool = False
    """Ablation: always take the best candidate instead of drawing by lottery."""

    square_weights: bool = True
    """Brain: lottery weights are ``score**2`` - strong links win more often,
    weak ones still can."""

    grounded_bonus: float = Field(default=1.2, ge=1.0)
    """Brain: memory traces get x1.2; here ideas grounded in the user's context do."""


class FatigueConfig(_Section):
    """Satiation: a branch that keeps producing the same ideas closes.

    Brain: ``Neuron.fatigue``.
    """

    enabled: bool = True

    gain: float = Field(default=0.12 * TICKS_PER_THOUGHT, ge=0.0)
    """Added per fully repetitive thought (brain: +0.12 per repeated tick)."""

    recovery: float = Field(default=0.06 * TICKS_PER_THOUGHT, ge=0.0)
    """Removed per fully novel thought (brain: -0.06 per tick without repetition)."""

    satiated: float = Field(default=0.45, gt=0.0)
    """Brain ``SATIATED``: at this fatigue the branch is closed as "bored"."""

    max_fatigue: float = Field(default=0.8, gt=0.0)
    """Brain ``Neuron.max_fatigue``."""


class InhibitionConfig(_Section):
    """Inhibition of return: what was just considered is not considered again."""

    enabled: bool = True

    window: int | None = Field(default=None, ge=1)
    """Thoughts during which a visited idea blocks its duplicates. ``None`` = the
    whole session. The brain blocks for 60 ticks (about 11 thoughts) because its
    stream never ends; a bounded session gains nothing from re-thinking an idea."""

    shown: int = Field(default=30, ge=0)
    """How many already-considered ideas the model sees in each step."""


class FadeConfig(_Section):
    """Thoughts weaken with depth (brain: ``FADE``, ``MIN_STRENGTH``, ``MAX_DEPTH``)."""

    factor: float = Field(default=0.86, gt=0.0, le=1.0)
    min_strength: float = Field(default=0.22, ge=0.0)
    max_depth: int = Field(default=7, ge=1)

    link_modulation: float | None = Field(default=0.4, ge=0.0, le=1.0)
    """Mock-up ``thinking.py``: child strength also scales by ``min(1, 0.4 + link)``,
    so weakly linked tangents exhaust sooner. ``None`` disables it."""


class ValueConfig(_Section):
    """Pull towards valuable ideas (brain milestone M3)."""

    enabled: bool = True

    pull: float = Field(default=1.0, ge=0.0)
    """Brain ``VALUE_PULL``: candidate weight is multiplied by ``exp(pull * value)``."""

    floor: float = -0.5
    ceiling: float = 1.0
    """Brain ``VALUE_RANGE``: with squared weights, good vs bad is about 20:1 -
    worse ideas are considered less often, but never never."""

    critic: Literal["inline", "separate"] = "inline"
    """``inline``: the expanding call also scores its proposals (cheaper).
    ``separate``: an extra critic call scores them against the user's needs."""

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.floor >= self.ceiling:
            raise ValueError("value floor must be below ceiling")
        return self


class CaptureConfig(_Section):
    """Capture by the unexpected (brain: ``CAPTURE``, ``CAPTURE_GAP``)."""

    enabled: bool = True

    threshold: float = Field(default=0.2, ge=0.0)
    """How far a surprise must rise above the running baseline to hijack the stream."""

    baseline_rate: float = Field(default=1.0 - 0.95**TICKS_PER_THOUGHT, gt=0.0, le=1.0)
    """Baseline update per thought (brain: EMA rate 0.05 per tick)."""

    refractory: int = Field(default=round(15 / TICKS_PER_THOUGHT), ge=0)
    """Thoughts between two captures (brain: 15 ticks)."""


class EmergenceConfig(_Section):
    """Where a new thought comes from when a stream has nothing to continue."""

    cued: float = Field(default=0.5, ge=0.0, le=1.0)
    """Brain ``CUED``: share of new thoughts prompted by what is in front of the eyes -
    here, a detail of the user's context."""

    recency_scale: float = Field(default=30 / TICKS_PER_THOUGHT, gt=0.0)
    """Brain: recall weight has a recency term ``1 / (1 + age / 30 ticks)``."""

    memory_recall: int = Field(default=5, ge=0)
    """Findings recalled from long-term memory at the start of a run."""


class ReinstatementConfig(_Section):
    """What context accompanies the focus (brain: reinstatement, ``CONTEXT = 3``)."""

    mode: Literal["focused", "full"] = "focused"
    """``focused``: each step sees the few context items relevant to its focus.
    ``full``: the whole context sits in the (cached) system prompt."""

    items: int = Field(default=3, ge=0)
    """Brain ``CONTEXT``: a strong thought lights up the 3 parts it is made of."""

    weak_items: int = Field(default=1, ge=0)
    weak_below: float = Field(default=0.5, ge=0.0, le=1.0)
    """Brain: below strength 0.5 only 1 part is reinstated."""

    item_chars: int = Field(default=6000, ge=200)
    full_chars: int = Field(default=200_000, ge=1000)
    digest_items: int = Field(default=80, ge=0)


class ExpansionConfig(_Section):
    """One step: a model call that thinks about one idea and proposes the next ones."""

    max_candidates: int = Field(default=6, ge=1)
    """Brain: the stream looks at the 6 strongest links."""

    min_candidates: int = Field(default=3, ge=1)
    """Weak (deep) thoughts propose fewer continuations: budget shrinks with depth."""

    context_candidates: int = Field(default=2, ge=0)
    """At least this many proposals must ask what is non-standard about this user."""

    max_tokens: int = Field(default=4096, ge=256)
    effort: Effort | None = None
    retries: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if self.min_candidates > self.max_candidates:
            raise ValueError("min_candidates must not exceed max_candidates")
        if self.context_candidates > self.max_candidates:
            raise ValueError("context_candidates must not exceed max_candidates")
        return self


class SynthesisConfig(_Section):
    """The final call that turns the thought tree into an answer."""

    max_tokens: int = Field(default=8000, ge=512)
    top_nodes: int = Field(default=24, ge=1)
    effort: Effort | None = None
    reserve_input_tokens: int = Field(default=6000, ge=0)
    reserve_output_tokens: int = Field(default=4000, ge=0)
    """Tokens held back for the synthesis call while thinking, so it can always run."""


class ThinkConfig(_Section):
    """Complete configuration of a :class:`~widethink.Thinker`."""

    parallel: int = Field(default=2, ge=1)
    """Streams thinking concurrently over one shared tree (mock-up ``branching = 2``)."""

    max_thoughts: int | None = Field(default=24, ge=1)
    """Upper bound on thoughts (expansion calls) per run, in addition to the token budget."""

    baseline: bool = True
    """Start from the standard answer and mark its decisions as already considered."""

    baseline_max_tokens: int = Field(default=6000, ge=512)

    selection: SelectionConfig = SelectionConfig()
    fatigue: FatigueConfig = FatigueConfig()
    inhibition: InhibitionConfig = InhibitionConfig()
    fade: FadeConfig = FadeConfig()
    value: ValueConfig = ValueConfig()
    capture: CaptureConfig = CaptureConfig()
    emergence: EmergenceConfig = EmergenceConfig()
    reinstatement: ReinstatementConfig = ReinstatementConfig()
    expansion: ExpansionConfig = ExpansionConfig()
    synthesis: SynthesisConfig = SynthesisConfig()

    def without(self, *mechanisms: str) -> ThinkConfig:
        """Return a copy with the named mechanisms switched off (see :data:`ABLATIONS`)."""
        unknown = sorted(set(mechanisms) - set(ABLATIONS))
        if unknown:
            raise ValueError(f"unknown mechanisms {unknown}; expected any of {list(ABLATIONS)}")
        cfg = self
        for name in mechanisms:
            cfg = cfg.model_copy(update=_ablation(cfg, name))
        return cfg


def _ablation(cfg: ThinkConfig, name: str) -> dict[str, object]:
    match name:
        case "baseline":
            return {"baseline": False}
        case "capture":
            return {"capture": cfg.capture.model_copy(update={"enabled": False})}
        case "cueing":
            return {"emergence": cfg.emergence.model_copy(update={"cued": 0.0})}
        case "fade":
            return {
                "fade": cfg.fade.model_copy(
                    update={"factor": 1.0, "min_strength": 0.0, "link_modulation": None}
                )
            }
        case "fatigue":
            return {"fatigue": cfg.fatigue.model_copy(update={"enabled": False})}
        case "grounding":
            return {"selection": cfg.selection.model_copy(update={"grounded_bonus": 1.0})}
        case "inhibition":
            return {"inhibition": cfg.inhibition.model_copy(update={"enabled": False})}
        case "lottery":
            return {"selection": cfg.selection.model_copy(update={"greedy": True})}
        case "parallel":
            return {"parallel": 1}
        case "value":
            return {"value": cfg.value.model_copy(update={"enabled": False})}
    raise AssertionError(name)  # pragma: no cover - guarded by ABLATIONS
