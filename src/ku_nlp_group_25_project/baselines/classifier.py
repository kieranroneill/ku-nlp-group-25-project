from __future__ import annotations

import re
import unicodedata

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .prediction import Prediction
from ku_nlp_group_25_project.constants import LANGUAGES
from ku_nlp_group_25_project.types import NormalizationForm


class Classifier:
    """
     Deterministic lexical-overlap classifier for answerability prediction.

     The classifier predicts that a question is answerable when the question and context share at least
     `min_shared_tokens` distinct tokens after Unicode normalization, optional case folding, and regex-based
     tokenization.

     Attributes:
         _min_shared_tokens (int): Minimum number of distinct tokens shared by the question and context required to
            predict answerable.
         _normalization_form (NormalizationForm): Unicode normalization form applied before tokenization.
         _casefolding (bool): Whether Unicode case folding is applied after normalization.
     """
    _min_shared_tokens: int
    _normalization_form: NormalizationForm
    _casefolding: bool

    ###
    # constructors
    ###

    def __init__(
        self,
        casefolding: bool = True,
        min_shared_tokens: int = 1,
        normalization_form: NormalizationForm = "NFC",
    ) -> None:
        """
        Initializes a deterministic lexical-overlap classifier.

        Args:
            casefolding (bool): Whether to apply Unicode case folding after normalization. Defaults to True.
            min_shared_tokens (int): Minimum number of distinct tokens that must occur in both the question and context
                for the classifier to predict. Must be at least 1. Defaults to 1.
            normalization_form (NormalizationForm): Unicode normalization form. Defaults to `"NFC"`.

        Raises:
            ValueError: If `min_shared_tokens` is less than 1.
        """
        if min_shared_tokens < 1:
            raise ValueError(
                "min_shared_tokens must be at least 1."
            )

        self._min_shared_tokens = min_shared_tokens
        self._casefolding = casefolding
        self._normalization_form = normalization_form

    ###
    # private functions
    ###

    def _normalize(self, text: str) -> str:
        """
        Applies the classifier's fixed Unicode normalization policy.

        Args:
            text (str): Input text to normalize.

        Returns:
            str: The normalized text, with case folding applied if case folding is enabled.

        Raises:
            TypeError: If `text` is not a string.
        """
        normalized = unicodedata.normalize(
            self._normalization_form,
            text,
        )

        if not self._casefolding:
            return normalized

        return normalized.casefold()

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into Unicode word-character sequences.

        Args:
            text (str): Input text to normalize and tokenize.

        Returns:
            list[str]: Normalized word tokens in their original left-to-right occurrence order. Repeated tokens are
                retained at this stage; later prediction logic converts them to sets.

        Raises:
            TypeError: If `text` is not a string.
        """
        normalized = self._normalize(text)

        return re.compile(r"[^\W_]+", flags=re.UNICODE).findall(normalized)

    ###
    # public functions
    ###

    @staticmethod
    def evaluate(
        predictions_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Evaluate answerability predictions overall and by language.

        The supplied DataFrame must contain both gold labels and classifier predictions. The method filters rows by
        language and evaluates all retained examples as the `"Overall"` group, and then evaluates each language
        separately.

        The gold-label column is `answerable` must represent the dataset annotation:

        - `True` / `1`: The supplied context supports an extractable answer.
        - `False` / `0`: The supplied context does not support an extractable answer.

        The prediction column `predicted_answerable` must represent the rule classifier output:

        - `True` / `1`: The rule predicts answerable.
        - `False` / `0`: The rule predicts unanswerable.

        Precision, recall, and F1 are reported for both answerable and unanswerable labels.

        Confusion-matrix counts use the following convention:

        - TP: Gold answerable and predicted answerable.
        - TN: Gold unanswerable and predicted unanswerable.
        - FP: Gold unanswerable and predicted answerable.
        - FN: Gold answerable and predicted unanswerable.

        Args:
            predictions_dataframe (pd.DataFrame): DataFrame containing gold labels, predicted labels, and language
                codes.

        Returns:
            pd.DataFrame: The returned DataFrame contains the following columns:

            - `group`: `"Overall"` or a language code.
            - `n`: Number of evaluated examples.
            - `accuracy`: Fraction of correct predictions.
            - `answerable_precision`, `answerable_recall`,
              `answerable_f1`: Metrics for the answerable class.
            - `unanswerable_precision`, `unanswerable_recall`,
              `unanswerable_f1`: Metrics for the unanswerable class.
            - `answerable_support`: Number of gold answerable
              examples.
            - `unanswerable_support`: Number of gold unanswerable
              examples.
            - `tp`, `tn`, `fp`, `fn`: Confusion-matrix counts.

        Raises:
            ValueError: If any required column is missing.
            ValueError: If filtering to a required evaluation group produces no examples.
        """
        missing = {
            "answerable",
            "lang",
            "predicted_answerable",
        }.difference(predictions_dataframe.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        evaluation_dataframe = predictions_dataframe.copy()
        evaluation_dataframe = evaluation_dataframe.loc[
            evaluation_dataframe["lang"].isin(LANGUAGES)
        ].copy()

        groups = [("Overall", evaluation_dataframe)]

        for language in LANGUAGES:
            language_df = evaluation_dataframe.loc[
                evaluation_dataframe["lang"] == language
                ].copy()
            groups.append((language, language_df))

        rows = []

        for group_name, group_dataframe in groups:
            if group_dataframe.empty:
                raise ValueError(
                    f"Cannot evaluate empty group: {group_name}."
                )

            y_true = group_dataframe["answerable"].astype(int).to_numpy()
            y_prediction = group_dataframe["predicted_answerable"].astype(int).to_numpy()
            precision, recall, f1, support = (
                precision_recall_fscore_support(
                    y_true,
                    y_prediction,
                    labels=[1, 0],
                    average=None,
                    zero_division=0,
                )
            )

            answerable_precision = precision[0]
            unanswerable_precision = precision[1]
            answerable_recall = recall[0]
            unanswerable_recall = recall[1]
            answerable_f1 = f1[0]
            unanswerable_f1 = f1[1]
            answerable_support = support[0]
            unanswerable_support = support[1]
            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_prediction,
                labels=[0, 1],
            ).ravel()

            rows.append(
                {
                    "group": group_name,
                    "n": len(group_dataframe),
                    "accuracy": accuracy_score(y_true, y_prediction),
                    "macro_f1": (
                        answerable_f1 + unanswerable_f1
                    ) / 2.0,
                    "answerable_precision": answerable_precision,
                    "answerable_recall": answerable_recall,
                    "answerable_f1": answerable_f1,
                    "unanswerable_precision": unanswerable_precision,
                    "unanswerable_recall": unanswerable_recall,
                    "unanswerable_f1": unanswerable_f1,
                    "answerable_support": int(answerable_support),
                    "unanswerable_support": int(unanswerable_support),
                    "tp": int(tp),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                }
            )

        return pd.DataFrame(rows)

    ###
    # public functions
    ###


    def predict(
        self,
        context: str,
        question: str,
    ) -> Prediction:
        """
        Predict answerability for a question-context pair.

        The method normalizes and tokenizes the question and context, then converts each token sequence into a set of
        distinct tokens, and finally identifies their intersection.

        It predicts answerable when the number of shared distinct tokens is at least `_min_shared_tokens`.

        Args:
            context (str): English context passage in which an answer may or may not be supported.
            question (str): Question in Arabic, Korean, Telugu, or another supported question language.

        Returns:
            Prediction: Rule output and transparent lexical metrics.

        Raises:
            TypeError: If `context` or `question` is not a string.
        """
        context_tokens = set(self._tokenize(context))
        question_tokens = set(self._tokenize(question))
        shared_tokens = context_tokens & question_tokens
        question_token_count = len(question_tokens)
        shared_token_count = len(shared_tokens)

        return Prediction(
            answerable=(
                shared_token_count
                >= self._min_shared_tokens
            ),
            context_token_count=len(context_tokens),
            context_tokens=sorted(list(context_tokens)),
            question_token_count=question_token_count,
            question_tokens=sorted(list(question_tokens)),
            shared_token_count=shared_token_count,
            shared_tokens=sorted(list(shared_tokens)),
        )

    def predictions(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generates deterministic answerability predictions for a supplied DataFrame.

        Prediction and metric columns are appended to a copy of the supplied DataFrame:

        - `predicted_answerable`: Boolean rule output.
        - `shared_token_count`: Number of distinct tokens present in both question and context.
        - `question_token_count`: Number of distinct normalized question tokens.
        - `context_token_count`: Number of distinct normalized context tokens.
        - `shared_tokens`: Shared tokens separated by a pipe `" | "` for inspection in a DataFrame or CSV.

        Args:
            dataframe (pd.DataFrame): Input examples.

        Returns:
            pd.DataFrame: A copy of the input DataFrame with prediction and metric columns appended. The input DataFrame
            is not modified.

        Raises:
            ValueError: If `context` or `question` is missing from `dataframe`.
            TypeError: If a value in `context` or `question` is not a string.
        """
        missing = {
            "context",
            "question",
        }.difference(dataframe.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        results = dataframe.copy()
        predictions = [
            self.predict(
                context=context,
                question=question,
            )
            for context, question in list(
                results[
                    ["context", "question"]
                ].itertuples(index=False, name=None)
            )
        ]

        results["predicted_answerable"] = [
            prediction.answerable
            for prediction in predictions
        ]
        results["shared_token_count"] = [
            prediction.shared_token_count
            for prediction in predictions
        ]
        results["question_token_count"] = [
            prediction.question_token_count
            for prediction in predictions
        ]
        results["context_token_count"] = [
            prediction.context_token_count
            for prediction in predictions
        ]
        results["shared_tokens"] = [
            " | ".join(prediction.shared_tokens)
            for prediction in predictions
        ]

        return results
