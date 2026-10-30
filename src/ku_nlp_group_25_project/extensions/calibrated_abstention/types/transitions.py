from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Transitions:
    """
    Paired per-example movement between two decision rules on the same examples.

    Attributes:
        broken_abstentions (int): Unanswerable examples the new rule now answers.
        broken_answers (int): Answerable examples the new rule now refuses.
        rescued_abstentions (int): Unanswerable examples the new rule now refuses.
        rescued_answers (int): Answerable examples the new rule now answers.
        unchanged (int): Examples whose decision did not change.
    """

    broken_abstentions: int
    broken_answers: int
    rescued_abstentions: int
    rescued_answers: int
    unchanged: int

    ###
    # public functions
    ###

    def to_dict(self) -> dict:
        return {
            "broken_abstentions": self.broken_abstentions,
            "broken_answers": self.broken_answers,
            "rescued_abstentions": self.rescued_abstentions,
            "rescued_answers": self.rescued_answers,
            "unchanged": self.unchanged,
        }
