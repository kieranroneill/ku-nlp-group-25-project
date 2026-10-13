from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    answerable: bool
    context_token_count: int
    context_tokens: list[str]
    question_tokens: list[str]
    question_token_count: int
    shared_token_count: int
    shared_tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "answerable": self.answerable,
            "context_token_count": self.context_token_count,
            "context_tokens": self.context_tokens,
            "question_tokens": self.question_tokens,
            "question_token_count": self.question_token_count,
            "shared_token_count": self.shared_token_count,
            "shared_tokens": self.shared_tokens,
        }
