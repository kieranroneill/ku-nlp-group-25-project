from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Thresholds:
    """
    Fitted cutoffs for one abstention rule, in the rule's own score space.

    Attributes:
        cutoff (float): Cutoff applied to every example, the language cutoffs aside.
        language_cutoffs (Mapping[str, float]): Per-language overrides in the same score space.
        temperature (float): Positive temperature used to calibrate the no-answer score.
    """

    cutoff: float
    language_cutoffs: Mapping[str, float] = field(default_factory=dict)
    temperature: float = 1.0

    ###
    # public functions
    ###

    def to_dict(self) -> dict:
        return {
            "cutoff": self.cutoff,
            "language_cutoffs": dict(self.language_cutoffs),
            "temperature": self.temperature,
        }
