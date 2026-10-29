from __future__ import annotations

from typing import Sequence

import numpy as np

from ..example_score import ExampleScore
from ..thresholds import Thresholds
from .abstention_rule import AbstentionRule


class FeatureCombinerRule(AbstentionRule):
    """
     Logistic regression over post-hoc features, scored out of fold so no example is
     decided by a model that saw its own label.

     Attributes:
         _folds (int): Cross-fitting folds.
         _seed (int): Fold shuffle seed.
         _probabilities (dict[str, float]): Out-of-fold answerability probability per identifier.
    """

    ###
    # constructors
    ###

    def __init__(self, folds: int = 5, seed: int = 13) -> None:
        """
        Initializes an unfitted combiner.

        Args:
            folds (int): Cross-fitting folds. Defaults to 5.
            seed (int): Fold shuffle seed. Defaults to 13.

        Raises:
            ValueError: If fewer than two folds are requested.
        """
        if folds < 2:
            raise ValueError("folds must be at least 2.")

        self._folds = folds
        self._seed = seed
        self._probabilities: dict[str, float] = {}

    ###
    # properties
    ###

    @property
    def name(self) -> str:
        return "E_feature_combiner"

    ###
    # public functions
    ###

    def fit(self, scores: Sequence[ExampleScore]) -> Thresholds:
        """
        Fits the combiner and stores out-of-fold probabilities.

        Args:
            scores (Sequence[ExampleScore]): Validation examples.

        Returns:
            Thresholds: Fitted cutoff on the negated out-of-fold probability.

        Raises:
            ValueError: If no examples are supplied.
            ImportError: If scikit-learn is unavailable.
        """
        if not scores:
            raise ValueError("scores must not be empty.")

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as error:
            raise ImportError("the feature combiner requires scikit-learn.") from error

        features, labels = self._features(scores), self._labels(scores)
        self._probabilities = {}
        folds = StratifiedKFold(n_splits=self._folds, shuffle=True, random_state=self._seed)

        for trained, held_out in folds.split(features, labels):
            pipeline = make_pipeline(StandardScaler(),
                                     LogisticRegression(max_iter=1000, class_weight="balanced"))
            pipeline.fit(features[trained], labels[trained])

            for index, probability in zip(held_out, pipeline.predict_proba(features[held_out])[:, 1]):
                self._probabilities[scores[index].id] = float(probability)

        confidence = np.array([self._probabilities[score.id] for score in scores])

        return Thresholds(cutoff=self._best_cutoff(-confidence, labels))

    def decide(self, scores: Sequence[ExampleScore], thresholds: Thresholds) -> list[bool]:
        """
        Answers when the out-of-fold probability clears the fitted cutoff.

        Args:
            scores (Sequence[ExampleScore]): Examples passed to fit.
            thresholds (Thresholds): Cutoffs returned by fit.

        Returns:
            list[bool]: Whether the system answers each example.

        Raises:
            RuntimeError: If the rule has not been fitted on these examples.
        """
        confidence = self.probabilities(scores)

        return [-value <= thresholds.cutoff for value in confidence]

    def probabilities(self, scores: Sequence[ExampleScore]) -> np.ndarray:
        """
        Gets the stored out-of-fold answerability probabilities.

        Args:
            scores (Sequence[ExampleScore]): Examples passed to fit.

        Returns:
            np.ndarray: Probability per example, aligned with scores.

        Raises:
            RuntimeError: If the rule has not been fitted on these examples.
        """
        if not self._probabilities:
            raise RuntimeError("rule has not been fitted.")

        try:
            return np.array([self._probabilities[score.id] for score in scores])
        except KeyError as error:
            raise RuntimeError("scores must be the ones passed to fit.") from error
