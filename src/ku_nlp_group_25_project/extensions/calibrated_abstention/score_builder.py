from __future__ import annotations

import math
import re
import string
from collections import Counter

from .types import ExampleScore


class ScoreBuilder:
    """
     Attributes:
         _tolerance (float): Allowed gap between the stored no-answer score and null minus best.
    """

    ###
    # constructors
    ###

    def __init__(self, tolerance: float = 1e-3) -> None:
        """
        Initialises the builder.

        Args:
            tolerance (float): Allowed gap between no_answer_score and null minus best. Defaults to 1e-3.

        Raises:
            ValueError: If the tolerance is not finite and positive.
        """
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive.")

        self._tolerance = tolerance

    ###
    # private static functions
    ###

    @staticmethod
    def _normalise(text: str) -> str:
        text = str(text)
        text = "".join(character for character in text.lower() if character not in set(string.punctuation))

        return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())

    @classmethod
    def _token_f1(cls, prediction: str, answer: str) -> float:
        """Returns the token-level F1 of the normalised prediction against the normalised answer."""
        predicted = cls._normalise(prediction).split()
        gold = cls._normalise(answer).split()

        if not predicted or not gold:
            return float(predicted == gold)

        overlap = sum((Counter(predicted) & Counter(gold)).values())

        if overlap == 0:
            return 0.0

        precision = overlap / len(predicted)
        recall = overlap / len(gold)

        return 2 * precision * recall / (precision + recall)

    ###
    # private functions
    ###

    def _score(self, prediction: dict) -> ExampleScore:
        """
        Parses the score from a raw prediction.

        Args:
            prediction (dict): Raw predictions row.

        Returns:
            ExampleScore: Scored example with recomputed span metrics.
        """
        label = str(prediction["gold_answerable"]).strip().lower()

        if label not in {"0", "1", "true", "false"}:
            raise ValueError(f"row {prediction['id']}: invalid gold_answerable {label!r}.")

        answerable = label in {"1", "true"}
        context = str(prediction.get("context", ""))
        gold_answer = "" if not answerable else prediction["gold_answer"]
        predicted_span = prediction["best_span_text"]
        null_score = float(prediction["null_score"])
        best_span_score = float(prediction["best_span_score"])
        margin = null_score - best_span_score
        question = str(prediction["question"])
        stored_margin = float(prediction["no_answer_score"])

        if not all(math.isfinite(value) for value in (null_score, best_span_score, stored_margin)):
            raise ValueError(f"row {prediction['id']}: scores must be finite.")

        if abs(margin - stored_margin) > self._tolerance:
            raise ValueError(
                f"row {prediction['id']}: no_answer_score {prediction['no_answer_score']} does not equal "
                f"null minus best span {margin}."
            )

        asked = set(ScoreBuilder._normalise(question).split())
        given = set(ScoreBuilder._normalise(context).split())

        return ExampleScore(
            answerable=answerable,
            best_span_score=best_span_score,
            context_tokens=len(context.split()),
            exact_match=float(ScoreBuilder._normalise(predicted_span) == ScoreBuilder._normalise(gold_answer)) if answerable else 0.0,
            gold_answer=gold_answer,
            id=prediction["id"],
            language=prediction["lang"],
            null_score=null_score,
            overlap=len(asked & given) / len(asked) if asked else 0.0,
            predicted_span=predicted_span,
            question=question,
            question_tokens=len(question.split()),
            span_tokens= len(str(predicted_span).split()),
            token_f1=self._token_f1(predicted_span, gold_answer) if answerable else 0.0,
        )

    ###
    # public functions
    ###

    def parse(self, predictions: list[dict]) -> list[ExampleScore]:
        """
        Parses the predictions into scores.

        Args:
            predictions (list[dict]): The raw list of predictions.

        Returns:
            list[ExampleScore]: The parsed example scores.
        """
        return [self._score(prediction) for prediction in predictions]
