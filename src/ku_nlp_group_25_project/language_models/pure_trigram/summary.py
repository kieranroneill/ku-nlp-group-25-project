from __future__ import annotations

from dataclasses import dataclass

from .question_score import QuestionScore

@dataclass(frozen=True)
class Summary:
    minimum_count: int
    perplexity: float
    question_scores: list[QuestionScore]
    smoothing: float
    token_count: int
    unknown_rate: float
    vocabulary_size: int

    def to_dict(self) -> dict:
        return {
            "minimum_count": self.minimum_count,
            "perplexity": self.perplexity,
            "question_scores": self.question_scores,
            "smoothing": self.smoothing,
            "token_count": self.token_count,
            "unknown_rate": self.unknown_rate,
            "vocabulary_size": self.vocabulary_size,
        }
