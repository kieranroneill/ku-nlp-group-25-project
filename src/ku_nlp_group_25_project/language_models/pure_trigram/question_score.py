from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class QuestionScore:
    index: int
    negative_log_likelihood: float
    perplexity: float
    question: str
    tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "negative_log_likelihood": self.negative_log_likelihood,
            "perplexity": self.perplexity,
            "question": self.question,
            "tokens": self.tokens,
        }
