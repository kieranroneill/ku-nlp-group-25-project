from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from .question_score import QuestionScore
from .summary import Summary
from ku_nlp_group_25_project.constants import BOS, EOS, UNK


class PureTrigramLanguageModel:
    """
     Pure add-k trigram model for Korean questions.

     Attributes:
         _minimum_count (int): Minimum total training frequency for a retained text token.
         _smoothing (float): Positive add-k constant.
         _vocabulary (set[str]): Prediction targets, including UNK and EOS.
         _counts (Counter): Counts of (older history, newer history, target) triples.
         _histories (Counter): Counts of two-token histories with an outgoing target.
         _trained (bool): Whether training has completed.
    """

    ###
    # constructors
    ###

    def __init__(self, minimum_count: int = 2, smoothing: float = 0.1) -> None:
        """
        Initializes an unfitted pure trigram.

        Args:
            minimum_count (int): Total-frequency cutoff. Defaults to 2.
            smoothing (float): Add-k constant. Defaults to 0.1.

        Raises:
            ValueError: If the cutoff is not a positive integer or smoothing is not finite and positive.
        """
        if type(minimum_count) is not int or minimum_count < 1:
            raise ValueError("minimum_count must be a positive integer.")
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be finite and positive.")

        self._minimum_count = minimum_count
        self._smoothing = smoothing
        self._vocabulary = set()
        self._counts = Counter()
        self._histories = Counter()
        self._trained = False

    ###
    # private functions
    ###

    @staticmethod
    def _tokenise(question: str) -> list[str]:
        """
        Splits on whitespace without changing case, punctuation or Unicode normalisation.

        Args:
            question (str): Original question text.

        Returns:
            list[str]: Tokens in occurrence order; empty text yields an empty list.

        Raises:
            TypeError: If the question is not a string.
            ValueError: If a reserved boundary or unknown marker occurs as a literal token.
        """
        if not isinstance(question, str):
            raise TypeError("question must be a string.")

        tokens = question.split()

        if {BOS, EOS, UNK}.intersection(tokens):
            raise ValueError("question contains a reserved token.")

        return tokens

    def _encode(self, tokens: Sequence[str]) -> list[str]:
        """Maps tokens outside the training vocabulary to UNK, preserving order."""
        return [token if token in self._vocabulary else UNK for token in tokens]

    def _probability(self, older: str, newer: str, target: str) -> float:
        """
        Computes P(target | older, newer) for already encoded tokens.

        Args:
            older (str): Older encoded history token, or BOS.
            newer (str): Immediately preceding encoded token, or BOS.
            target (str): Encoded prediction target; BOS is not allowed.

        Returns:
            float: Add-k smoothed conditional probability.

        Raises:
            ValueError: If the supplied tokens are not valid encoded tokens.
        """

        if target not in self._vocabulary:
            raise ValueError("target must belong to the prediction vocabulary.")

        if any(token not in self._vocabulary and token != BOS for token in (older, newer)):
            raise ValueError("history must contain encoded tokens or BOS.")

        return (self._counts[(older, newer, target)] + self._smoothing) / (
            self._histories[(older, newer)] + self._smoothing * len(self._vocabulary)
        )

    def _score(self, question: str, index: int = 0) -> QuestionScore:
        """
        Scores one question's tokens plus one EOS, excluding both BOS markers.

        Args:
            question (str): Validation question text, kept in the returned diagnostic record.
            index (int): Position in the Korean-filtered validation list, not a dataset ID.

        Returns:
            QuestionScore: Mapped tokens, NLL and per-question perplexity. Empty text scores EOS only.

        Raises:
            RuntimeError: If training has not completed.
        """
        tokens = self._encode(self._tokenise(question))
        sequence = [BOS, BOS] + tokens + [EOS]
        negative_log_likelihood = -sum(
            math.log(self._probability(*sequence[i - 2:i + 1])) for i in range(2, len(sequence))
        )

        return QuestionScore(
            index=index,
            negative_log_likelihood=negative_log_likelihood,
            perplexity=math.exp(negative_log_likelihood / (len(tokens) + 1)),
            question=question,
            tokens=tokens,
        )

    ###
    # public functions
    ###

    def train(self, questions: Sequence[str]) -> None:
        """
        Trains the model on the supplied training questions.

        Args:
            questions (Sequence[str]): Training questions.

        Raises:
            ValueError: If no training questions are supplied.
            TypeError: If any question is not a string.
        """
        if not questions:
            raise ValueError("training questions must not be empty.")

        self._trained = False

        tokenised = [self._tokenise(question) for question in questions]
        frequencies = Counter(token for tokens in tokenised for token in tokens)
        self._vocabulary = {UNK, EOS}

        for token, count in frequencies.items():
            if count >= self._minimum_count:
                self._vocabulary.add(token)

        self._counts.clear()
        self._histories.clear()

        for tokens in tokenised:
            sequence = [BOS, BOS] + self._encode(tokens) + [EOS]

            for i in range(2, len(sequence)):
                older, newer, target = sequence[i - 2:i + 1]

                self._counts[(older, newer, target)] += 1
                self._histories[(older, newer)] += 1

        self._trained = True

    def summary(self, validation_questions: Sequence[str]) -> Summary:
        """
        Gets a summary of the model including the computed perplexity by pooling NLL and target counts.

        Args:
            validation_questions (Sequence[str]): Validation questions in their original order.

        Returns:
            Summary: Model summary.

        Raises:
            ValueError: If the validation split is empty.
            RuntimeError: If training has not completed.
        """
        if not self._trained:
            raise RuntimeError("model has not been trained.")

        if not validation_questions:
            raise ValueError("validation questions must not be empty.")

        negative_log_likelihoods = 0.0
        predictions = 0
        scores: list[QuestionScore] = []
        token_count = 0
        unknown_token_count = 0

        # score each question
        for index, question in enumerate(validation_questions):
            score = self._score(question, index)
            negative_log_likelihoods += score.negative_log_likelihood
            predictions += len(score.tokens) + 1
            token_count += len(score.tokens)
            unknown_token_count += score.tokens.count(UNK)

            scores.append(score)

        return Summary(
            minimum_count=self._minimum_count,
            perplexity=math.exp(negative_log_likelihoods / predictions),
            question_scores=len(scores),
            smoothing=self._smoothing,
            token_count=token_count,
            unknown_rate=unknown_token_count / token_count,
            vocabulary_size=len(self._vocabulary),
        )
