from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ExampleScore:
    """
    Validation example score that inclides the two span scores, the prediction and its gold match.

    Attributes:
        answerable (bool): Whether the context supports an answer.
        best_span_score (float): Best non-empty in-context span score, maximum over windows.
        context_tokens (int): Tokenised context length.
        exact_match (float): SQuAD-v2 exact match of the prediction against the gold answer.
        gold_answer (str): Gold answer span; empty for unanswerable items.
        id (str): Item identifier, unique within the validation split.
        language (str): Question language code, one of ar, ko, te.
        null_score (float): CLS-span score, the minimum over windows.
        overlap (float): Share of question tokens occurring in the context.
        predicted_span (str): Best extracted span, empty when no span was found.
        question (str): Original question text.
        question_tokens (int): Tokenised question length.
        span_tokens (int): Whitespace tokens in the predicted span.
        token_f1 (float): SQuAD-v2 token F1 of the prediction against the gold answer.
    """

    answerable: bool
    best_span_score: float
    context_tokens: int
    exact_match: float
    gold_answer: str
    id: str
    language: str
    null_score: float
    overlap: float
    predicted_span: str
    question: str
    question_tokens: int
    span_tokens: int
    token_f1: float

    ###
    # properties
    ###

    @property
    def margin(self) -> float:
        """Gets the no-answer margin; a larger value means a stronger abstention signal."""
        return self.null_score - self.best_span_score

    ###
    # public functions
    ###

    def to_dict(self) -> dict:
        return {
            "answerable": self.answerable,
            "best_span_score": self.best_span_score,
            "exact_match": self.exact_match,
            "context_tokens": self.context_tokens,
            "gold_answer": self.gold_answer,
            "id": self.id,
            "language": self.language,
            "null_score": self.null_score,
            "margin": self.margin,
            "overlap": self.overlap,
            "predicted_span": self.predicted_span,
            "question": self.question,
            "question_tokens": self.question_tokens,
            "span_tokens": self.span_tokens,
            "token_f1": self.token_f1,
        }
