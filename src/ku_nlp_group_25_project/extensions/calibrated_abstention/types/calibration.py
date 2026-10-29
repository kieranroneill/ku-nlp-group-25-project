from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Calibration:
    """
    Confidence diagnostics for the no-answer score.

    Attributes:
        average_risk (float): Area under the risk-coverage curve of the margin ranking.
        average_risk_by_language (Mapping[str, float]): The same per language.
        calibrated_expected_calibration_error (float): ECE after temperature scaling.
        expected_calibration_error (float): ECE of the raw answerability probability.
        feature_area_under_roc (float): AUC of the post-hoc feature combiner.
        temperature (float): Fitted positive temperature.
    """

    average_risk: float
    average_risk_by_language: Mapping[str, float]
    calibrated_expected_calibration_error: float
    expected_calibration_error: float
    feature_area_under_roc: float
    temperature: float

    ###
    # public functions
    ###

    def to_dict(self) -> dict:
        return {
            "average_risk": self.average_risk,
            "average_risk_by_language": dict(self.average_risk_by_language),
            "calibrated_expected_calibration_error": self.calibrated_expected_calibration_error,
            "expected_calibration_error": self.expected_calibration_error,
            "feature_area_under_roc": self.feature_area_under_roc,
            "temperature": self.temperature,
        }
