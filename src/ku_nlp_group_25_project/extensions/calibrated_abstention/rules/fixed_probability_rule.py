from __future__ import annotations

import math
from typing import Sequence

from ..example_score import ExampleScore
from ..thresholds import Thresholds
from .abstention_rule import AbstentionRule


class FixedProbabilityRule(AbstentionRule):
    """
     Untuned cut at 0.5 on the calibrated answerability probability, a deployment
     ablation that uses no validation labels.

     Attributes:
         _temperature (float): Temperature fitted by the calibrated rule.
    """

    ###
    # constructors
    ###

    def __init__(self, temperature: float = 1.0) -> None:
        """
        Initializes the fixed-probability rule.

        Args:
            temperature (float): Temperature fitted on validation. Defaults to 1.0.

        Raises:
            ValueError: If the temperature is not finite and positive.
        """
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")

        self._temperature = temperature

    ###
    # properties
    ###

    @property
    def name(self) -> str:
        return "F_fixed_probability"

    ###
    # public functions
    ###

    def fit(self, scores: Sequence[ExampleScore]) -> Thresholds:
        """Returns the untuned cut without looking at any label."""
        if not scores:
            raise ValueError("scores must not be empty.")

        return Thresholds(cutoff=-0.5, temperature=self._temperature)

    def decide(self, scores: Sequence[ExampleScore], thresholds: Thresholds) -> list[bool]:
        """Answers when the calibrated probability reaches one half."""
        calibrated = -self._answer_probability(self._margins(scores), thresholds.temperature)

        return [value <= thresholds.cutoff for value in calibrated]
