from __future__ import annotations

from typing import Sequence

from ..types import ExampleScore, Thresholds
from .abstention_rule import AbstentionRule


class TemperatureScaledRule(AbstentionRule):
    """
     Temperature-scaled answerability probability with one pooled cutoff.
    """

    ###
    # properties
    ###

    @property
    def name(self) -> str:
        return "C_temperature_scaled"

    ###
    # public functions
    ###

    def fit(self, scores: Sequence[ExampleScore]) -> Thresholds:
        """Fits the temperature on validation, then the cutoff on the calibrated probability."""
        if not scores:
            raise ValueError("scores must not be empty.")

        margins, labels = self._margins(scores), self._labels(scores)
        temperature = self._fit_temperature(margins, labels)
        calibrated = -self._answer_probability(margins, temperature)

        return Thresholds(cutoff=self._best_cutoff(calibrated, labels), temperature=temperature)

    def decide(self, scores: Sequence[ExampleScore], thresholds: Thresholds) -> list[bool]:
        """Answers when the calibrated answerability probability clears the fitted cutoff."""
        calibrated = -self._answer_probability(self._margins(scores), thresholds.temperature)

        return [value <= thresholds.cutoff for value in calibrated]
