"""Mechanisms of the thought stream, each a small pure unit.

They are transplanted from the digital brain's thought stream
(``digital_brain/neurons/thought.py``); ``docs/mechanisms.md`` maps every
brain constant to its counterpart here. Keeping each mechanism separate makes
it testable in isolation and lets ablation studies switch it off alone.
"""

from widethink.mechanisms.capture import SurpriseMonitor
from widethink.mechanisms.emergence import Option, Source, choose_emergence, recall_weight
from widethink.mechanisms.fade import (
    child_strength,
    exhausted,
    proposal_count,
    reinstated_items,
    too_deep,
)
from widethink.mechanisms.fatigue import mean_repetition, repetition, satiated, update_fatigue
from widethink.mechanisms.inhibition import InhibitionOfReturn
from widethink.mechanisms.selection import Candidate, draw, score
from widethink.mechanisms.value import value_pull

__all__ = [
    "Candidate",
    "InhibitionOfReturn",
    "Option",
    "Source",
    "SurpriseMonitor",
    "child_strength",
    "choose_emergence",
    "draw",
    "exhausted",
    "mean_repetition",
    "proposal_count",
    "recall_weight",
    "reinstated_items",
    "repetition",
    "satiated",
    "score",
    "too_deep",
    "update_fatigue",
    "value_pull",
]
