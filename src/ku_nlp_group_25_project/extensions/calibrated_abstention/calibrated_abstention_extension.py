from __future__ import annotations

import csv
import json
import os
from typing import Sequence

import numpy as np

from .rules import (
    AbstentionRule,
    ConstrainedRecallRule,
    FeatureCombinerRule,
    FixedProbabilityRule,
    GlobalThresholdRule,
    PerLanguageThresholdRule,
    TemperatureScaledRule,
)
from .types import (
    Calibration,
    ExampleScore,
    Metrics,
    PairedComparison,
    Result,
    Transitions,
)


class CalibratedAbstentionExtension:
    """
     Runs the calibrated-abstention extension: six ways of choosing the no-answer cutoff
     on one frozen reader and one validation split.

     Attributes:
         _seed (int): Seed for the paired bootstrap and the cross-fitting folds.
         _bootstrap_samples (int): Resamples used for the paired interval.
         _target_recall (float): Abstention-recall guarantee used by rule D.
         _folds (int): Cross-fitting folds used by rule E.
    """

    ###
    # constructors
    ###

    def __init__(self,
        seed: int,
        bootstrap_samples: int = 2000,
        target_recall: float = 0.90,
        folds: int = 5,
    ) -> None:
        """
        Initializes the extension.

        Args:
            seed (int): Seed for the bootstrap and cross-fitting.
            bootstrap_samples (int): Resamples for the paired interval. Defaults to 2000.
            target_recall (float): Abstention-recall guarantee for rule D. Defaults to 0.90.
            folds (int): Cross-fitting folds for rule E. Defaults to 5.

        Raises:
            ValueError: If the sample count or fold count is not positive.
        """
        if bootstrap_samples < 1 or folds < 2:
            raise ValueError("bootstrap_samples and folds must be positive.")

        self._seed = seed
        self._bootstrap_samples = bootstrap_samples
        self._target_recall = target_recall
        self._folds = folds

    ###
    # private static functions
    ###

    @staticmethod
    def _calibrate(scores: Sequence[ExampleScore], temperature: float,
                   rules: dict[str, AbstentionRule]) -> Calibration:
        """Builds the confidence diagnostics for the no-answer score."""
        margins = np.array([score.margin for score in scores], dtype=float)
        labels = np.array([int(score.answerable) for score in scores], dtype=float)
        languages = sorted({score.language for score in scores})

        return Calibration(
            temperature=temperature,
            expected_calibration_error=AbstentionRule._expected_calibration_error(
                AbstentionRule._answer_probability(margins, 1.0), labels),
            calibrated_expected_calibration_error=AbstentionRule._expected_calibration_error(
                AbstentionRule._answer_probability(margins, temperature), labels),
            average_risk=AbstentionRule._average_risk(scores, -margins),
            average_risk_by_language={
                language: AbstentionRule._average_risk(
                    [score for score in scores if score.language == language],
                    -np.array([score.margin for score in scores if score.language == language]),
                )
                for language in languages
            },
            feature_area_under_roc=AbstentionRule._area_under_roc(
                scores, rules["E_feature_combiner"].probabilities(scores)),
        )

    @staticmethod
    def _compute_metrics(scores: Sequence[ExampleScore], decisions: Sequence[bool]) -> "Metrics":
        """
        Computes the metrics for one rule's decisions.

        Args:
            scores (Sequence[ExampleScore]): Evaluated examples.
            decisions (Sequence[bool]): Whether the rule answered each example, aligned with scores.

        Returns:
            Metrics: Metrics over the supplied examples.

        Raises:
            ValueError: If no examples are supplied or the lengths differ.
        """
        if not scores:
            raise ValueError("scores must not be empty.")

        if len(scores) != len(decisions):
            raise ValueError("decisions must align with scores.")

        answered = [bool(decision) for decision in decisions]
        positives = [score for score in scores if score.answerable]
        negatives = [score for score in scores if not score.answerable]
        true_positive = sum(1 for score, answer in zip(scores, answered) if score.answerable and answer)
        false_positive = sum(1 for score, answer in zip(scores, answered) if not score.answerable and answer)
        false_negative = len(positives) - true_positive
        true_negative = len(negatives) - false_positive
        answerable_f1 = CalibratedAbstentionExtension._f1(true_positive, false_positive, false_negative)
        unanswerable_f1 = CalibratedAbstentionExtension._f1(true_negative, false_negative, false_positive)
        answered_positives = [answer for score, answer in zip(scores, answered) if score.answerable]
        answered_negatives = [answer for score, answer in zip(scores, answered) if not score.answerable]
        exact_matches = [score.exact_match if answer else 0.0 for score, answer in zip(scores, answered)]
        token_f1s = [score.token_f1 if answer else 0.0 for score, answer in zip(scores, answered)]

        return Metrics(
            count=len(scores),
            macro_f1=(answerable_f1 + unanswerable_f1) / 2.0,
            answerable_f1=answerable_f1,
            unanswerable_f1=unanswerable_f1,
            exact_match=sum(exact_matches) / len(scores),
            token_f1=sum(token_f1s) / len(scores),
            answerable_coverage=CalibratedAbstentionExtension._mean(answered_positives),
            abstention_recall=1.0 - CalibratedAbstentionExtension._mean(answered_negatives),
            answered_precision=CalibratedAbstentionExtension._mean([score.answerable for score, answer in zip(scores, answered) if answer]),
        )

    @staticmethod
    def _compute_transitions(
        scores: Sequence[ExampleScore],
        baseline: Sequence[bool],
        candidate: Sequence[bool]
    ) -> "Transitions":
        """
        Counts paired movement between a baseline and a candidate rule.

        Args:
            scores (Sequence[ExampleScore]): Evaluated examples.
            baseline (Sequence[bool]): Baseline decisions, aligned with scores.
            candidate (Sequence[bool]): Candidate decisions, aligned with scores.

        Returns:
            Transitions: Counts over the supplied examples.

        Raises:
            ValueError: If no examples are supplied or the lengths differ.
        """
        if not scores:
            raise ValueError("scores must not be empty.")

        if not len(scores) == len(baseline) == len(candidate):
            raise ValueError("decisions must align with scores.")

        rescued_abstentions = broken_abstentions = rescued_answers = broken_answers = unchanged = 0

        for score, was_answered, is_answered in zip(scores, baseline, candidate):
            if was_answered == is_answered:
                unchanged += 1
            elif is_answered and score.answerable:
                rescued_answers += 1
            elif is_answered and not score.answerable:
                broken_abstentions += 1
            elif not is_answered and score.answerable:
                broken_answers += 1
            else:
                rescued_abstentions += 1

        return Transitions(
            rescued_abstentions=rescued_abstentions,
            broken_abstentions=broken_abstentions,
            rescued_answers=rescued_answers,
            broken_answers=broken_answers,
            unchanged=unchanged,
        )

    @staticmethod
    def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
        """Computes a binary F1 score, zero when the denominator vanishes."""
        denominator = 2 * true_positive + false_positive + false_negative

        return 0.0 if denominator == 0 else 2.0 * true_positive / denominator

    @staticmethod
    def _mean(flags: Sequence[bool]) -> float:
        """Returns the share of true flags, zero for an empty sequence."""
        return sum(1 for flag in flags if flag) / len(flags) if flags else 0.0

    ###
    # private functions
    ###

    def _bootstrap(
        self,
        scores: Sequence[ExampleScore],
        baseline: Sequence[bool],
        candidate: Sequence[bool],
    ) -> PairedComparison:
        """Bootstraps the macro-F1 difference over examples, candidate minus baseline."""
        generator = np.random.default_rng(self._seed)
        indices = np.arange(len(scores))
        observed = (CalibratedAbstentionExtension._compute_metrics(scores, candidate).macro_f1
                    - CalibratedAbstentionExtension._compute_metrics(scores, baseline).macro_f1)
        differences = np.empty(self._bootstrap_samples)

        for sample in range(self._bootstrap_samples):
            chosen = generator.choice(indices, size=len(indices), replace=True)
            resampled = [scores[index] for index in chosen]
            differences[sample] = (
                CalibratedAbstentionExtension._compute_metrics(resampled, [candidate[index] for index in chosen]).macro_f1
                - CalibratedAbstentionExtension._compute_metrics(resampled, [baseline[index] for index in chosen]).macro_f1
            )

        lower, upper = np.percentile(differences, [2.5, 97.5])

        return PairedComparison(
            difference=observed,
            lower=float(lower),
            upper=float(upper),
            not_better_fraction=float((differences <= 0).mean()),
        )

    def _rules(self, trained_temperature: float | None = None) -> list[AbstentionRule]:
        """Builds the rule set in report order, threading the fitted temperature into rule F."""
        rules: list[AbstentionRule] = [
            GlobalThresholdRule(),
            PerLanguageThresholdRule(),
            TemperatureScaledRule(),
            ConstrainedRecallRule(target_recall=self._target_recall),
            FeatureCombinerRule(folds=self._folds, seed=self._seed),
        ]

        if trained_temperature is not None:
            rules.append(FixedProbabilityRule(temperature=trained_temperature))

        return rules

    ###
    # public functions
    ###

    def evaluate(self, scores: Sequence[ExampleScore]) -> Result:
        """
        Fits every rule on the supplied examples and reports them under one protocol.

        Args:
            scores (Sequence[ExampleScore]): Validation examples scored by the frozen reader.

        Returns:
            Result: Metrics, thresholds, transitions and confidence diagnostics.

        Raises:
            ValueError: If no examples are supplied or a language is missing.
        """
        if not scores:
            raise ValueError("scores must not be empty.")

        temperature_rule = TemperatureScaledRule()
        temperature_thresholds = temperature_rule.fit(scores)
        rules = {rule.name: rule for rule in self._rules(temperature_thresholds.temperature)}
        thresholds = {
            name: (temperature_thresholds if name == temperature_rule.name else rule.fit(scores))
            for name, rule in rules.items()
        }
        decisions = {name: rule.decide(scores, thresholds[name]) for name, rule in rules.items()}
        baseline = decisions["A_global_threshold"]
        metrics = {name: CalibratedAbstentionExtension._compute_metrics(scores, decision) for name, decision in decisions.items()}
        metrics_by_language = {
            name: {
                language: CalibratedAbstentionExtension._compute_metrics([score for score in scores if score.language == language],
                                          [decision for score, decision in zip(scores, decisions[name])
                                           if score.language == language])
                for language in sorted({score.language for score in scores})
            }
            for name in decisions
        }

        return Result(
            calibration=self._calibrate(scores, temperature_thresholds.temperature, rules),
            metrics=metrics,
            metrics_by_language=metrics_by_language,
            thresholds=thresholds,
            transitions={
                name: CalibratedAbstentionExtension._compute_transitions(scores, baseline, decisions[name])
                for name in ("B_per_language_thresholds", "D_constrained_recall")
            },
            comparison=self._bootstrap(scores, baseline, decisions["B_per_language_thresholds"]),
            decisions=decisions,
        )

    def save(self, result: ExtensionResult, directory: str) -> None:
        """
        Writes the reportable results as CSV and JSON files.

        Args:
            result (ExtensionResult): Result returned by evaluate.
            directory (str): Destination directory, created when missing.

        Raises:
            ValueError: If the result has no metrics.
            OSError: If the directory cannot be created or written.
        """
        if not result.metrics:
            raise ValueError("result must contain metrics.")

        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, "metrics.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["rule"] + list(next(iter(result.metrics.values())).to_dict()))
            writer.writeheader()

            for rule, metrics in result.metrics.items():
                writer.writerow({"rule": rule, **metrics.to_dict()})

        with open(os.path.join(directory, "metrics_by_language.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["rule", "language"]
                                    + list(next(iter(next(iter(result.metrics_by_language.values())).values())).to_dict()))
            writer.writeheader()

            for rule, per_language in result.metrics_by_language.items():
                for language, metrics in per_language.items():
                    writer.writerow({"rule": rule, "language": language, **metrics.to_dict()})

        with open(os.path.join(directory, "thresholds.json"), "w", encoding="utf-8") as handle:
            json.dump({rule: thresholds.to_dict() for rule, thresholds in result.thresholds.items()},
                      handle, indent=2)

        with open(os.path.join(directory, "calibration.json"), "w", encoding="utf-8") as handle:
            json.dump(result.calibration.to_dict(), handle, indent=2)

        with open(os.path.join(directory, "transitions.json"), "w", encoding="utf-8") as handle:
            json.dump({rule: transitions.to_dict() for rule, transitions in result.transitions.items()},
                      handle, indent=2)

        with open(os.path.join(directory, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)
