from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ..types import (
    ExampleScore,
    Thresholds,
)


class AbstentionRule(ABC):
    """
     Base class for an abstention rule: how the no-answer cutoff is chosen.

     Every rule shares one score and one decision form, abstain when the score exceeds
     the cutoff, and differs only in how the cutoff is fitted. Calling decide with the
     same scores passed to fit is required.
    """

    ###
    # properties
    ###

    @property
    @abstractmethod
    def name(self) -> str:
        """Gets the rule label used in the report tables."""

    ###
    # private static functions
    ###

    @staticmethod
    def _labels(scores: Sequence[ExampleScore]) -> np.ndarray:
        """Returns answerability as an integer array."""
        return np.array([score.answerable for score in scores], dtype=int)

    @staticmethod
    def _margins(scores: Sequence[ExampleScore]) -> np.ndarray:
        """Returns the no-answer margins as a float array."""
        return np.array([score.margin for score in scores], dtype=float)

    @staticmethod
    def _sweep(cutoffs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates every achievable cutoff in O(n log n).

        Args:
            cutoffs (np.ndarray): Score per example; answering means score <= cutoff.
            labels (np.ndarray): Answerability per example.

        Returns:
            tuple[np.ndarray, np.ndarray]: Candidate cutoffs and their macro-F1.
        """
        order = np.argsort(cutoffs, kind="mergesort")
        ordered, labels = cutoffs[order], labels[order]
        count = len(ordered)
        answered = np.arange(count + 1)
        true_positive = np.concatenate([[0], np.cumsum(labels)]).astype(float)
        positive = labels.sum()
        false_negative = positive - true_positive
        false_positive = answered - true_positive
        true_negative = count - positive - false_positive
        answerable_f1 = AbstentionRule._safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
        unanswerable_f1 = AbstentionRule._safe_ratio(2 * true_negative, 2 * true_negative + false_positive + false_negative)
        candidates = np.concatenate([[ordered[0] - 1.0],
                                     (ordered[:-1] + ordered[1:]) / 2.0,
                                     [ordered[-1] + 1.0]])

        return candidates, (answerable_f1 + unanswerable_f1) / 2.0

    @staticmethod
    def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Divides elementwise, returning zero wherever the denominator vanishes."""
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                         where=np.asarray(denominator) != 0)

    @classmethod
    def _best_cutoff(cls, cutoffs: np.ndarray, labels: np.ndarray) -> float:
        """Returns the cutoff maximising macro-F1, the middle of any tied plateau."""
        candidates, macro_f1 = cls._sweep(cutoffs, labels)
        tied = np.flatnonzero(np.isclose(macro_f1, macro_f1.max(), atol=1e-12))

        return float(candidates[int(np.median(tied))])

    @classmethod
    def _recall_cutoff(cls, cutoffs: np.ndarray, labels: np.ndarray, target: float) -> float:
        """Returns the largest cutoff whose unanswerable-class recall still reaches the target."""
        candidates, _ = cls._sweep(cutoffs, labels)
        negatives = cutoffs[labels == 0]

        if len(negatives) == 0:
            return float(candidates[-1])

        eligible = candidates[candidates <= np.quantile(negatives, 1.0 - target)]

        return float(eligible.max() if len(eligible) else candidates[0])

    @staticmethod
    def _answer_probability(margins: np.ndarray, temperature: float) -> np.ndarray:
        """Maps the margin to an answerability probability through a two-way softmax."""
        return 1.0 / (1.0 + np.exp(margins / temperature))

    @classmethod
    def _fit_temperature(cls, margins: np.ndarray, labels: np.ndarray,
                         lower: float = 0.05, upper: float = 50.0) -> float:
        """
        Fits one temperature by golden-section search on the validation negative log-likelihood.

        Args:
            margins (np.ndarray): No-answer margins.
            labels (np.ndarray): Answerability, one for answerable.
            lower (float): Smallest allowed temperature. Defaults to 0.05.
            upper (float): Largest allowed temperature. Defaults to 50.0.

        Returns:
            float: Temperature minimising the negative log-likelihood.
        """
        def loss(temperature: float) -> float:
            probability = np.clip(cls._answer_probability(margins, temperature), 1e-12, 1 - 1e-12)

            return float(-(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)).mean())

        low, high = math.log(lower), math.log(upper)
        ratio = (math.sqrt(5) - 1) / 2
        inner_low, inner_high = high - ratio * (high - low), low + ratio * (high - low)

        for _ in range(80):
            if loss(math.exp(inner_low)) < loss(math.exp(inner_high)):
                high = inner_high
            else:
                low = inner_low

            inner_low, inner_high = high - ratio * (high - low), low + ratio * (high - low)

        return math.exp((low + high) / 2)

    @staticmethod
    def _expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray,
                                    bins: int = 15) -> float:
        """Returns the bin-weighted gap between predicted probability and observed frequency."""
        edges = np.linspace(0.0, 1.0, bins + 1)
        assignment = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
        error = 0.0

        for index in range(bins):
            members = assignment == index

            if members.any():
                error += members.mean() * abs(labels[members].mean() - probabilities[members].mean())

        return float(error)

    @staticmethod
    def _average_risk(scores: Sequence[ExampleScore], confidence: np.ndarray) -> float:
        """Returns the area under the risk-coverage curve of a confidence ranking."""
        order = np.argsort(-confidence)
        exact_matches = np.array([score.exact_match for score in scores], dtype=float)[order]
        coverage = np.arange(1, len(scores) + 1) / len(scores)
        risk = 1.0 - np.cumsum(exact_matches) / np.arange(1, len(scores) + 1)
        integrate = getattr(np, "trapezoid", None) or np.trapz

        return float(integrate(risk, coverage))

    @staticmethod
    def _area_under_roc(scores: Sequence[ExampleScore], confidence: np.ndarray) -> float:
        """Returns the rank-based AUC of a confidence score against answerability."""
        labels = np.array([score.answerable for score in scores], dtype=int)
        order = np.argsort(confidence)
        ranks = np.arange(1, len(labels) + 1)
        ranked = labels[order]
        positives, negatives = ranked.sum(), len(ranked) - ranked.sum()

        if positives == 0 or negatives == 0:
            return float("nan")

        return float((ranks[ranked == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))

    @staticmethod
    def _features(scores: Sequence[ExampleScore]) -> np.ndarray:
        """Builds the post-hoc feature matrix from scores only, never from gold answers."""
        return np.array([[score.margin,
                          math.log1p(score.context_tokens),
                          math.log1p(score.question_tokens),
                          score.overlap,
                          score.span_tokens] for score in scores], dtype=float)

    ###
    # public functions
    ###

    @abstractmethod
    def fit(self, scores: Sequence[ExampleScore]) -> Thresholds:
        """
        Fits the rule's cutoffs on the supplied examples.

        Args:
            scores (Sequence[ExampleScore]): Examples used for tuning, the validation split.

        Returns:
            Thresholds: Fitted cutoffs in the rule's score space.
        """

    @abstractmethod
    def decide(self, scores: Sequence[ExampleScore], thresholds: Thresholds) -> list[bool]:
        """
        Applies the fitted rule.

        Args:
            scores (Sequence[ExampleScore]): Examples to decide, the same ones passed to fit.
            thresholds (Thresholds): Cutoffs returned by fit.

        Returns:
            list[bool]: Whether the system answers each example.
        """
