from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .calibration import Calibration
from .metrics import Metrics
from .paired_comparison import PairedComparison
from .thresholds import Thresholds
from .transitions import Transitions


@dataclass(frozen=True)
class Result:
    """
    Attributes:
        calibration (Calibration): Confidence diagnostics for the no-answer score.
        comparison (PairedComparison): Paired bootstrap of the per-language rule.
        decisions (Mapping[str, Sequence[bool]]): Per-example decisions per rule, kept for
            error inspection only and never serialised.
        metrics (Mapping[str, Metrics]): Overall metrics per rule.
        metrics_by_language (Mapping[str, Mapping[str, Metrics]]): Metrics per rule and language.
        thresholds (Mapping[str, Thresholds]): Fitted cutoffs per rule.
        transitions (Mapping[str, Transitions]): Paired movement against the baseline rule.
    """

    calibration: Calibration
    comparison: PairedComparison
    metrics: Mapping[str, Metrics]
    metrics_by_language: Mapping[str, Mapping[str, Metrics]]
    thresholds: Mapping[str, Thresholds]
    transitions: Mapping[str, Transitions]
    decisions: Mapping[str, Sequence[bool]] = field(default_factory=dict, repr=False)

    ###
    # public functions
    ###

    def to_dict(self) -> dict:
        return {
            "calibration": self.calibration.to_dict(),
            "comparison": self.comparison.to_dict(),
            "metrics": {rule: metrics.to_dict() for rule, metrics in self.metrics.items()},
            "metrics_by_language": {
                rule: {language: metrics.to_dict() for language, metrics in per_language.items()}
                for rule, per_language in self.metrics_by_language.items()
            },
            "thresholds": {rule: thresholds.to_dict() for rule, thresholds in self.thresholds.items()},
            "transitions": {rule: transitions.to_dict() for rule, transitions in self.transitions.items()},
        }
