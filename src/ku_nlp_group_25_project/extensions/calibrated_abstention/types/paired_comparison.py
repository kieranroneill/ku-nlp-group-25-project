from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairedComparison:
    """
    Paired bootstrap of the macro-F1 difference between two decision rules.

    Attributes:
        difference (float): Observed macro-F1 difference, candidate minus baseline.
        lower (float): Lower bound of the 95 percent interval over examples.
        not_better_fraction (float): Share of replicates with no improvement.
        upper (float): Upper bound of the 95 percent interval over examples.
    """

    difference: float
    lower: float
    not_better_fraction: float
    upper: float

    ###
    # public functions
    ###

    def to_dict(self) -> dict:
        return {
            "difference": self.difference,
            "lower": self.lower,
            "not_better_fraction": self.not_better_fraction,
            "upper": self.upper,
        }
