from __future__ import annotations

import csv
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from .types import ExampleScore
from ku_nlp_group_25_project.constants import LANGUAGES


class ScoreBuilder:
    """
     Scores the validation split with one frozen multilingual extractive reader.

     Attributes:
         _device (str): Torch device used for inference.
         _maximum_answer_tokens (int): Longest span considered, in context tokens.
         _maximum_length (int): Maximum tokens per window.
         _model (object): Question-answering model in evaluation mode.
         _model_name (str): Canonical name for the model.
         _stride (int): Overlap between consecutive windows.
         _tokeniser (object): Fast tokeniser loaded from the checkpoint.
    """

    ###
    # constructors
    ###

    def __init__(self, model_name: str, maximum_length: int, stride: int, maximum_answer_tokens: int) -> None:
        """
        Loads the frozen checkpoint and its tokenizer.

        Args:
            model_name (str): Hub identifier or local checkpoint path.
            maximum_length (int): Maximum tokens per window.
            stride (int): Overlap between consecutive windows.
            maximum_answer_tokens (int): Longest span considered, in context tokens.

        Raises:
            ImportError: If torch or transformers are unavailable.
            TypeError: If the model name is not a string.
        """
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")

        self._device = ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_name).eval().to(self._device)
        self._model_name = model_name
        self._maximum_answer_tokens = maximum_answer_tokens
        self._maximum_length = maximum_length
        self._stride = stride
        self._tokeniser = AutoTokenizer.from_pretrained(model_name)

    ###
    # private static functions
    ###

    @staticmethod
    def _normalise(text: str) -> str:
        text = unicodedata.normalize("NFD", str(text))
        text = "".join(character for character in text if unicodedata.category(character) != "Mn")
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

    def _context_mask(self, sequence_ids: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Marks tokens of the second sequence, excluding specials and padding."""
        separator = self._tokeniser.sep_token_id
        positions = np.flatnonzero(sequence_ids == separator)
        mask = np.zeros(len(sequence_ids), dtype=bool)

        if len(positions) >= 1:
            mask[int(positions[0]) + 1:] = True

        return mask & (offsets[:, 0] < offsets[:, 1])

    def _score(self, question: str, context: str) -> tuple[float, float, str]:
        """
        Computes the no-answer score, the best span score and the extracted span.

        Args:
            question (str): Non-English question text.
            context (str): English context.

        Returns:
            tuple[float, float, str]: Null score, best span score and predicted span text.
        """
        encoded = self._tokeniser(
            question,
            context,
            truncation="only_second",
            max_length=self._maximum_length,
            stride=self._stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        window_offsets = encoded.pop("offset_mapping").numpy()

        encoded.pop("overflow_to_sample_mapping", None)

        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with torch.no_grad():
            output = self._model(**encoded)

        start_logits = output.start_logits.cpu().numpy()
        end_logits = output.end_logits.cpu().numpy()
        sequence_ids = encoded["input_ids"].cpu().numpy()
        null_scores: list[float] = []
        best_score, best_start, best_end = -float("inf"), 0, 0

        for window in range(start_logits.shape[0]):
            offsets = window_offsets[window]
            mask = self._context_mask(sequence_ids[window], offsets)

            if not mask.any():
                continue

            null_scores.append(float(start_logits[window, 0] + end_logits[window, 0]))
            indices = np.flatnonzero(mask)

            for start in indices:
                ends = np.arange(start, min(start + self._maximum_answer_tokens, indices.max() + 1))
                totals = start_logits[window, start] + end_logits[window, ends]
                best = int(np.argmax(totals))

                if totals[best] > best_score:
                    best_score = float(totals[best])
                    best_start, best_end = int(offsets[start, 0]), int(offsets[ends[best], 1])

        null_score = min(null_scores) if null_scores else 0.0
        span = context[best_start:best_end] if np.isfinite(best_score) else ""

        return null_score, best_score, span

    ###
    # public static functions
    ###

    @staticmethod
    def read_scores_from_path(path: Path) -> list[ExampleScore]:
        """
        Reads the examples from the supplied path.

        Args:
            path (Path): The path to the scores.

        Returns:
            list[ExampleScore]: The scored examples.
        """
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        return [
            ExampleScore(
                answerable=row["answerable"] == "True",
                best_span_score=float(row["best_span_score"]),
                context_tokens=int(row["context_tokens"]),
                exact_match=float(row["exact_match"]),
                gold_answer=row["gold_answer"],
                id=row["id"],
                language=row["language"],
                null_score=float(row["null_score"]),
                overlap=float(row["overlap"]),
                predicted_span=row["predicted_span"],
                question=row["question"],
                question_tokens=int(row["question_tokens"]),
                span_tokens=int(row["span_tokens"]),
                token_f1=float(row["token_f1"]),
            )
            for row in rows
        ]

    @staticmethod
    def write_scores_to_path(scores: Sequence[ExampleScore], path: Path) -> None:
        """
        Writes the validation examples to a CSV file at the given path.

        Args:
            scores (Sequence[ExampleScore]): Scored validation examples.
            path (Path): The path to write the scores to.

        Raises:
            ValueError: If no scores are empty.
        """
        if not scores:
            raise ValueError("scores must not be empty.")

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scores[0].to_dict()))
            writer.writeheader()

            for score in scores:
                writer.writerow(score.to_dict())

    ###
    # public functions
    ###

    def scores(self, validation_data: pd.DataFrame) -> list[ExampleScore]:
        """
        Computes the validation example scores.

        Args:
            validation_data (pd.DataFrame): Validation split with lang, question, context and
                an answerable column named answerable or answertable, plus answer and id when present.

        Returns:
            list[ExampleScore]: One scored record per example, in frame order.

        Raises:
            ValueError: If the frame is empty, lacks a required column or or does not have the
                required language.
        """
        if validation_data.empty:
            raise ValueError("validation_data must not be empty.")

        frame = validation_data.copy()
        answerable_column = "answerable" if "answerable" in frame.columns else "answertable"

        for column in ("lang", "question", "context"):
            if column not in frame.columns:
                raise ValueError(f"validation_data must contain {column}.")

        if answerable_column not in frame.columns:
            raise ValueError("validation_data must contain answerable or answertable.")

        frame["answerable"] = frame[answerable_column].astype(bool)
        frame["answer"] = frame["answer"].fillna("") if "answer" in frame.columns else ""
        frame["id"] = frame["id"].astype(str) if "id" in frame.columns else [
            f"{row.lang}-{index}" for index, row in frame.iterrows()
        ]
        frame = frame[frame["lang"].isin(LANGUAGES)].reset_index(drop=True)

        if frame.empty:
            raise ValueError("validation_data contains none of the configured languages.")

        scores: list[ExampleScore] = []

        for row in frame.itertuples():
            null_score, best_score, span = self._score(row.question, row.context)
            answerable = bool(row.answerable)
            gold = "" if not answerable else row.answer
            asked = set(self._normalise(row.question).split())
            given = set(self._normalise(row.context).split())

            scores.append(
                ExampleScore(
                    answerable=answerable,
                    best_span_score=best_score,
                    context_tokens=len(self._tokeniser(row.context, add_special_tokens=False)["input_ids"]),
                    exact_match=float(self._normalise(span) == self._normalise(gold)) if answerable else float(span == ""),
                    gold_answer=gold,
                    id=row.id,
                    language=row.lang,
                    null_score=null_score,
                    overlap=len(asked & given) / len(asked) if asked else 0.0,
                    predicted_span=span,
                    question=row.question,
                    question_tokens=len(self._tokeniser(row.question, add_special_tokens=False)["input_ids"]),
                    span_tokens=len(span.split()),
                    token_f1=self._token_f1(span, gold) if answerable else float(span == ""),
                )
            )

        return scores
