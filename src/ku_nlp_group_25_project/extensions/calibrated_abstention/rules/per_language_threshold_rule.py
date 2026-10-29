from __future__ import annotations

from typing import Sequence

import numpy as np

from ..example_score import ExampleScore
from ..thresholds import Thresholds
from .abstention_rule import AbstentionRule


class PerLanguageThresholdRule(AbstentionRule):
    """
     One no-answer cutoff per question language, falling back to the pooled cutoff.
    """

    ###
    # properties
    ###

    @property
    def name(self) -> str:
        return "B_per_language_thresholds"

    ###
    # public functions
    ###

    def fit(self, scores: Sequence[ExampleScore]) -> Thresholds:
        """Fits a cutoff per language on that language's validation subset."""
        if not scores:
            raise ValueError("scores must not be empty.")

        pooled = self._best_cutoff(self._margins(scores), self._labels(scores))
        language_cutoffs: dict[str, float] = {}

        for language in sorted({score.language for score in scores}):
            subset = [score for score in scores if score.language == language]
            labels = self._labels(subset)

            language_cutoffs[language] = (self._best_cutoff(self._margins(subset), labels)
                                          if len(np.unique(labels)) > 1 else pooled)

        return Thresholds(cutoff=pooled, language_cutoffs=language_cutoffs)

    def decide(self, scores: Sequence[ExampleScore], thresholds: Thresholds) -> list[bool]:
        """Answers using the language cutoff when available, otherwise the pooled cutoff."""
        return [
            score.margin <= thresholds.language_cutoffs.get(score.language, thresholds.cutoff)
            for score in scores
        ]
