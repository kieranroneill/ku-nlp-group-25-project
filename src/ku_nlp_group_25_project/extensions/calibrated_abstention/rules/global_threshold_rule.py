from __future__ import annotations

from typing import Sequence

from ..example_score import ExampleScore
from ..thresholds import Thresholds
from .abstention_rule import AbstentionRule


class GlobalThresholdRule(AbstentionRule):
    """
     One pooled no-answer cutoff, the core baseline.
    """

    ###
    # properties
    ###

    @property
    def name(self) -> str:
        return "A_global_threshold"

    ###
    # public functions
    ###

    def fit(self, scores: Sequence[ExampleScore]) -> Thresholds:
        """Fits the single cutoff maximising answerability macro-F1."""
        if not scores:
            raise ValueError("scores must not be empty.")

        return Thresholds(cutoff=self._best_cutoff(self._margins(scores), self._labels(scores)))

    def decide(self, scores: Sequence[ExampleScore], thresholds: Thresholds) -> list[bool]:
        """Answers every example whose margin does not exceed the cutoff."""
        return [score.margin <= thresholds.cutoff for score in scores]
