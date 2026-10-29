from __future__ import annotations

from typing import Sequence

from ..example_score import ExampleScore
from ..thresholds import Thresholds
from .abstention_rule import AbstentionRule


class ConstrainedRecallRule(AbstentionRule):
    """
     Per-language cutoffs chosen to reach a stated abstention-recall guarantee.

     Attributes:
         _target_recall (float): Required unanswerable-class recall.
    """

    ###
    # constructors
    ###

    def __init__(self, target_recall: float = 0.90) -> None:
        """
        Initializes the constrained rule.

        Args:
            target_recall (float): Required unanswerable-class recall. Defaults to 0.90.

        Raises:
            ValueError: If the target is not strictly between zero and one.
        """
        if not 0.0 < target_recall < 1.0:
            raise ValueError("target_recall must lie strictly between 0 and 1.")

        self._target_recall = target_recall

    ###
    # properties
    ###

    @property
    def name(self) -> str:
        return "D_constrained_recall"

    ###
    # public functions
    ###

    def fit(self, scores: Sequence[ExampleScore]) -> Thresholds:
        """Fits the largest cutoffs that still satisfy the recall target."""
        if not scores:
            raise ValueError("scores must not be empty.")

        pooled = self._recall_cutoff(self._margins(scores), self._labels(scores), self._target_recall)
        language_cutoffs = {
            language: self._recall_cutoff(
                self._margins([score for score in scores if score.language == language]),
                self._labels([score for score in scores if score.language == language]),
                self._target_recall,
            )
            for language in sorted({score.language for score in scores})
        }

        return Thresholds(cutoff=pooled, language_cutoffs=language_cutoffs)

    def decide(self, scores: Sequence[ExampleScore], thresholds: Thresholds) -> list[bool]:
        """Answers using the language cutoff when available, otherwise the pooled cutoff."""
        return [
            score.margin <= thresholds.language_cutoffs.get(score.language, thresholds.cutoff)
            for score in scores
        ]
