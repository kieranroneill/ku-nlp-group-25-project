from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    """
    Answerability and span metrics for one decision rule on one example set.

    Attributes:
        abstention_recall (float): Share of unanswerable examples the rule refused.
        answerable_coverage (float): Share of answerable examples the rule answered.
        answerable_f1 (float): F1 of the answered class.
        answered_precision (float): Share of answered examples that were answerable.
        count (int): Number of evaluated examples.
        exact_match (float): Mean SQuAD-v2 exact match, zero for abstained examples.
        macro_f1 (float): Unweighted mean of answerable- and unanswerable-class F1.
        token_f1 (float): Mean SQuAD-v2 token F1, zero for abstained examples.
        unanswerable_f1 (float): F1 of the abstained class.
    """

    abstention_recall: float
    answerable_coverage: float
    answerable_f1: float
    answered_precision: float
    count: int
    exact_match: float
    macro_f1: float
    token_f1: float
    unanswerable_f1: float

    ###
    # public functions
    ###

    def to_dict(self) -> dict:
        return {
            "abstention_recall": self.abstention_recall,
            "answerable_coverage": self.answerable_coverage,
            "answerable_f1": self.answerable_f1,
            "answered_precision": self.answered_precision,
            "count": self.count,
            "exact_match": self.exact_match,
            "macro_f1": self.macro_f1,
            "token_f1": self.token_f1,
            "unanswerable_f1": self.unanswerable_f1,
        }
