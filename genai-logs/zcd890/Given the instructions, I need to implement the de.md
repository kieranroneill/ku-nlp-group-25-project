<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Given the instructions, I need to implement the deterministic rule-based

classifier. Can you suggest one and give me a walkthrough on how it works?

I recommend a **translation-and-overlap** classifier: translate each question into English using a fixed translation system, then predict “answerable” only when enough of its content words appear together in a short window of the English context.

This fits your assignment: Week 36 explicitly permits machine translation and lexical features, but prohibits using labelled validation examples or a trained answerability/QA model to construct the rule. Translation is preprocessing here; the answerability decision itself is a fixed heuristic.[^1_1]

## 1. The rule

Use the following procedure:

1. Translate the Arabic, Korean or Telugu question into English.
2. Lowercase and tokenize the translated question and English context.
3. Remove a fixed list of common English words, including question words.
4. Measure how many distinct question content words appear in each 40-token context window.
5. Predict answerable if the best window contains:
    - at least two distinct question content words; and
    - at least 60% of the question’s distinct content words.

For a question containing only one content word, require that word to occur. If no content words remain, predict unanswerable.

The window length and 60% cutoff are proposed design choices, not empirically established optimal values. Fix and document them before evaluating on labelled validation data; do not adjust them in response to validation results. Your instructions specifically forbid labelled validation examples in constructing this baseline.[^1_1]

Translating the question addresses the language mismatch before lexical matching. Translation followed by overlap-based matching is an established approach in cross-lingual QA, although translation ambiguity and unfamiliar vocabulary can undermine it.[^1_2]

## 2. Walkthrough

Consider this invented English context:

> Marie Curie won the Nobel Prize in Physics in 1903. She later won the Nobel Prize in Chemistry in 1911.

Suppose an Arabic question translates to:

> When did Marie Curie win the Nobel Prize in Physics?

After tokenization and stopword removal:

```text
Question content words:
{marie, curie, win, nobel, prize, physics}
```

The context contains:

```text
{marie, curie, won, nobel, prize, physics, 1903,
 she, later, chemistry, 1911}
```

Five of the six question words match:

```text
Matched:     marie, curie, nobel, prize, physics
Not matched: win
Coverage:    5 / 6 ≈ 0.83
```

The classifier predicts answerable because the best window has at least two matches and coverage exceeds 0.60.

Notice that `win` and `won` do not match. That is intentional in this simple version: it uses exact token matching, without stemming or a learned semantic representation.

### A useful failure case

Now change the question to:

> When did Marie Curie win the Nobel Prize in Medicine?

The content words become:

```text
{marie, curie, win, nobel, prize, medicine}
```

Four of six still match, so coverage is about 0.67. The rule predicts answerable—even though the context does not support the question.

This is exactly the limitation worth analysing: the rule detects shared topic words, not whether the context supports the requested relationship. Your challenge-set requirements deliberately target minimally altered questions like this.[^1_1]

## 3. Implementation

This implementation uses only the Python standard library. It expects the English translation as an input, so translation can be run separately and cached.

```python
import math
import re
import unicodedata

# Deliberately small, explicit and fixed.
STOPWORDS = frozenset("""
a an the and or but
of in on at to for from by with about into
is are was were be been being
do does did
have has had
can could will would shall should may might must
what which who whom whose when where why how
this that these those
""".split())


def tokenize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[a-z0-9]+", text)


def predict_answerability(
    question_en,
    context,
    window_size=40,
    min_coverage=0.60,
):
    """
    Returns a fixed-rule prediction and an explanation.

    question_en: English translation of the original question.
    context: Original English context.

    No answer text, answer offsets or gold labels are used.
    """
    question_terms = set(tokenize(question_en)) - STOPWORDS
    context_tokens = tokenize(context)

    if not question_terms or not context_tokens:
        return {
            "answerable": False,
            "coverage": 0.0,
            "matched_terms": [],
            "question_terms": sorted(question_terms),
            "best_window": "",
        }

    # Inspect all overlapping windows. A short context gets one window.
    n_windows = max(1, len(context_tokens) - window_size + 1)

    best_matches = set()
    best_window = []

    for start in range(n_windows):
        window = context_tokens[start:start + window_size]
        matches = question_terms.intersection(window)

        # Keep the first window in case of a tie.
        if not best_window or len(matches) > len(best_matches):
            best_matches = matches
            best_window = window

    coverage = len(best_matches) / len(question_terms)

    required_matches = max(
        min(2, len(question_terms)),
        math.ceil(min_coverage * len(question_terms)),
    )

    return {
        "answerable": len(best_matches) >= required_matches,
        "coverage": coverage,
        "matched_terms": sorted(best_matches),
        "question_terms": sorted(question_terms),
        "best_window": " ".join(best_window),
    }
```

Example usage:

```python
context = (
    "Marie Curie won the Nobel Prize in Physics in 1903. "
    "She later won the Nobel Prize in Chemistry in 1911."
)

result = predict_answerability(
    question_en=(
        "When did Marie Curie win the Nobel Prize in Physics?"
    ),
    context=context,
)

assert result["answerable"] is True
assert result["coverage"] == 5 / 6

assert predict_answerability(
    "Where is Mount Everest?",
    context,
)["answerable"] is False

assert predict_answerability("", context)["answerable"] is False
assert predict_answerability("Who?", context)["answerable"] is False
```

The returned diagnostic fields let you explain each decision without exposing gold answers to the classifier.

### Why use windows?

In this design, matching words must occur within 40 consecutive context tokens, rather than anywhere in the entire passage. That prevents some matches from accumulating across unrelated parts of a long context.

It does not require the words to form a valid supporting statement. Windowing is simply a locality constraint.

## 4. Keep translation fixed and reproducible

For this baseline, use an ordinary pretrained translation system—not a QA model asked whether the context answers the question.

A suitable pipeline is:

```text
original question + source language
                ↓
fixed translation system
                ↓
cached English translation
                ↓
fixed overlap rule + English context
                ↓
answerable / unanswerable
```

Translate the question alone. Do not provide the gold answer, answerability label or context to the translator.

For reproducibility:

- Record the translation system, model/version and decoding settings.
- Use deterministic decoding where supported.
- Cache translations and reuse the same outputs for every run.
- Treat missing translations as preprocessing failures rather than silently translating them to empty strings.
- Record your stopword list, window length and cutoff.

With cached translations, the classifier above produces identical outputs for identical inputs. This is distinct from claiming that every external translation service will always return the same translation.

Your assignment also requires acknowledging translation systems and other external resources. If you use generative AI for translation, code generation or other project work, declare that use and include the required interaction logs—including this interaction.[^1_1]

## 5. Evaluate it under the required protocol

Run the frozen rule on validation examples and compare it with the separate majority-class baseline estimated from training data.

Report:

- Number of evaluated examples.
- Precision, recall and F1 for answerable.
- Precision, recall and F1 for unanswerable.
- Macro-F1: the unweighted mean of those two class F1 scores.
- Results overall and separately for Arabic, Korean and Telugu.

These are the assignment’s required answerability metrics; accuracy alone is insufficient. Freeze the rule before evaluating it on the challenge set, which must not be used for model selection or prompt development.[^1_1]

In your error analysis, distinguish anticipated limitations such as:


| Failure type | Why this rule can fail |
| :-- | :-- |
| Unsupported relation | Most topic words match, but the context does not state the requested fact. |
| Translation mismatch | The translation uses a different English term from the context. |
| Paraphrase or morphology | Exact matching misses pairs such as `win`/`won`. |
| Critical word substitution | A changed place, person or category leaves enough other words matching. |
| Distributed evidence | Relevant words occur too far apart to fit in one window. |

These are hypotheses about the rule’s behaviour, not results to report before inspecting actual errors.

The central interpretation is straightforward: this is an interpretable baseline for local lexical compatibility, not a verifier of contextual support. That makes it useful for measuring what your later learned models achieve beyond simple translated word matching.

<span style="display:none">[^1_10][^1_11][^1_3][^1_4][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: instructions.pdf

[^1_2]: https://publikationen.sulb.uni-saarland.de/bitstream/20.500.11880/23662/1/Thesis_final_print.pdf

[^1_3]: https://par.nsf.gov/servlets/purl/10359046

[^1_4]: https://www.jair.org/index.php/jair/specialtrack-CrossLanguage

[^1_5]: https://aclanthology.org/2025.starsem-1.8.pdf

[^1_6]: https://arxiv.org/html/2505.18673v1

[^1_7]: https://hal.science/hal-04369793/document

[^1_8]: https://scholars.cityu.edu.hk/files/334969039/316817901.pdf

[^1_9]: https://www.dfki.de/fileadmin/user_upload/import/11478_XQA_christine_thesis_2020.pdf

[^1_10]: https://www.mdpi.com/2073-431x/15/2/92

[^1_11]: https://www.semanticscholar.org/paper/Cross-Lingual-Transfer-Learning-for-Question-Lee-Lee/89c2cc5489d53e7c2082855aed9380694de23e0b


---

# I was reading about a fixed lexical-overlap rule. It seems simple and requires no model training. Is this a good alternative to the one you suggested?

Yes—a fixed lexical-overlap rule is a good, assignment-compliant baseline. But it is not fundamentally different from my suggestion: that was already a fixed lexical-overlap rule, with **translation** added before matching.

The important distinction is whether you compare the original non-English question directly with the English context, or translate it first. Your assignment permits both lexical features and machine translation, without requiring either.[^2_1]

## What changes if you skip translation?

Your dataset has Arabic, Korean and Telugu questions paired with English contexts. Direct exact-token matching therefore has a structural limitation: the same concept written in different scripts will not match.[^2_1]

For example, consider this illustrative pair:

```text
Question: أين تقع باريس؟
Meaning:  Where is Paris located?

Context: Paris is the capital of France.
```

A direct overlap rule cannot match `باريس` to `Paris`. A translated-overlap rule can.

Without translation, the rule can still exploit shared surface forms, such as:

- Identically written numbers.
- Latin-script names or abbreviations retained in the question.
- Other tokens copied unchanged between question and context.

However, those signals do not establish that the requested answer is supported. In particular, a year mentioned in the question is not necessarily the year being requested.

## Which version should you choose?

| Version | Advantage | Main limitation |
| :-- | :-- | :-- |
| Direct lexical overlap | No pretrained model dependency; very simple and fully deterministic | Cannot match differently written equivalents across languages |
| Translation + fixed overlap | Makes English word matching possible after translation | Adds a translation dependency and possible translation errors |

My recommendation is:

- If your priority is the simplest baseline with no model dependency at all, use direct lexical overlap and explicitly discuss the cross-lingual limitation.
- If you already have reliable cached translations, keep the translated version. It is a more informative design for testing lexical compatibility, though whether it performs better must be established experimentally.

Neither version requires training an answerability model. But my earlier suggestion does use a pretrained translation model, so “no training required” is not the same as “no model used.”

## A minimal direct-overlap implementation

Here is a deliberately simple alternative:

```python
import re
import unicodedata


def token_set(text):
    text = unicodedata.normalize("NFKC", text).casefold()

    # Unicode-aware word tokens; exclude underscores and punctuation.
    return set(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def predict_answerability(question, context):
    question_tokens = token_set(question)
    context_tokens = token_set(context)

    shared_tokens = question_tokens & context_tokens

    return {
        "answerable": len(shared_tokens) >= 1,
        "shared_tokens": sorted(shared_tokens),
    }
```

This rule predicts answerable whenever at least one token occurs in both inputs. The one-token cutoff is a proposed fixed design choice—not a validated optimum.

One important correction to adapting my earlier code: do not use its `[a-z0-9]+` tokenizer on the original multilingual questions. That tokenizer was intended for English translations and would discard Arabic, Korean and Telugu letters.

For this direct version, the walkthrough is:

1. Normalize both strings consistently.
2. Extract Unicode word tokens.
3. Compute their set intersection.
4. Return answerable if the intersection is nonempty.
5. Save the shared tokens so each prediction is explainable.

This tokenizer is intentionally crude: it does not perform language-specific morphological segmentation.

## How to justify it in your report

You could describe the design along these lines, adapting the wording to what you actually implement:

> We use a deterministic surface-overlap baseline. After Unicode normalization and tokenization, it predicts answerable when the question and context share at least one token. The rule uses neither translation nor learned representations. Consequently, it can exploit shared written forms but cannot connect equivalent expressions written differently across languages or verify that the context supports the requested relation.

Fix the tokenizer and cutoff before inspecting labelled validation results. Week 36 specifically prohibits using labelled validation examples to construct this rule. Then evaluate it against the training-derived majority baseline, reporting macro-F1 and precision/recall/F1 for both classes, overall and separately for each language.[^2_1]

A weak result is still useful if you explain what information the rule can and cannot exploit—that discussion is explicitly required by the assignment.[^2_1]

<div align="center">⁂</div>

[^2_1]: instructions.pdf


---

# Here is my code for my classifer:

```
# prediction.py
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

```

```
# classifier.py
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
```

I have tested it and it works, would you say this follows a fixed lexical-overlap rule?

Yes—your classifier implements a **fixed** lexical-overlap rule. The decision is deterministic, uses only the question and context, and does not involve fitting a model.

## Why it qualifies

Your `predict()` method implements this rule:

$$
\hat{y}(q,c)
=
\mathbf{1}\left[
|T(q)\cap T(c)| \geq k
\right]
$$

where:

- $T(x)$ is the set of distinct tokens after your configured Unicode normalization, optional case folding and regex tokenization.
- $k$ is `min_shared_tokens`, which defaults to 1.
- A prediction of 1 means answerable.

Specifically, your code:

```python
context_tokens = set(self._tokenize(context))
question_tokens = set(self._tokenize(question))
shared_tokens = context_tokens & question_tokens
```

computes the distinct-token intersection, and:

```python
answerable=shared_token_count >= self._min_shared_tokens
```

applies a fixed decision threshold. Repeated occurrences do not increase the score: a shared token appearing ten times still contributes only one match.

Having a configurable threshold does not make the classifier learned. What matters is how you choose it: setting it in advance is a fixed rule; selecting it based on labelled validation performance would conflict with the Week 36 restriction.

## Does it satisfy the project requirement?

Based on the code shown, yes—provided you choose the configuration without using labelled validation examples.

The instructions allow a deterministic classifier based on tokenization or lexical features, using only the question and context, without a trained answerability/QA model. Your prediction path meets that description. Keeping gold labels in the DataFrame for subsequent evaluation is fine: `predictions()` passes only `context` and `question` into `predict()`, so the shown prediction logic does not use those labels.

Your evaluation output also covers the requested answerability metrics:

- Precision, recall and F1 for both classes.
- Macro-F1 as the unweighted mean of the two class F1 scores.
- Overall and per-language results, with example counts.
- Accuracy and confusion-matrix counts as additional diagnostics.

That assessment assumes `LANGUAGES` contains the required Arabic, Korean and Telugu codes and that the evaluation inputs contain valid binary labels.

## Small details worth tightening

None of these changes the classification of your method as a fixed lexical-overlap rule.

### 1. Your token counts are distinct-token counts

Because `predict()` converts token lists to sets, `question_token_count` and `context_token_count` count unique normalized tokens, not sequence lengths.

You document this correctly in `predictions()`. For consistency, consider naming the fields:

```python
question_unique_token_count
context_unique_token_count
```

Alternatively, keep your names and explicitly state their meaning in the report. Do not reuse these counts as the assignment’s tokenized question/context lengths without retaining repeated tokens.

### 2. Validate evaluation labels before converting them

This line converts values but does not establish that they are valid binary labels:

```python
group_dataframe["answerable"].astype(int)
```

For example, a value of `2` would remain `2`. Add an explicit check for missing values and values outside `{False, True, 0, 1}` for both label columns, so malformed evaluation data fails clearly.

### 3. Document the tokenizer’s limitations

Your regex is Unicode-aware, but it is not a language-specific tokenizer. Describe it as regex-based tokenization rather than implying that it performs Arabic, Korean or Telugu morphological segmentation.

Likewise, normalization and case folding do not translate words: differently written equivalents across the question and context still do not match.

### 4. Make two minor documentation corrections

- Add `macro_f1` to the list of returned columns in `evaluate()`; the code returns it, but the docstring omits it.
- In `_tokenize()`, “original string order” is clearer than “left-to-right occurrence order,” especially when discussing Arabic text.


## Suggested report description

> Our deterministic lexical-overlap baseline applies NFC Unicode normalization and case folding, then extracts tokens using the Unicode-aware regular expression `[^\W_]+`. It converts the question and context tokens into sets and predicts answerable when they share at least one distinct token. The threshold and preprocessing configuration are fixed in advance, without using labelled validation examples. The rule uses no translation or learned representations and therefore detects only identical normalized surface forms, not semantic equivalence or contextual support.

Use that wording only if you actually run the default configuration shown. Your implementation is a clear, interpretable version of the intended baseline; the main remaining task is to evaluate and explain its cross-lingual limitations.

---

# I ran it through a notebook using the validation data from the HuggingFace dataset and here is my wirite up:

```

\section{W36: Data inspection and initial baseline}

\subsection{Baseline 2: deterministic lexical-overlap rule classifier}

\subsubsection*{Implementation}

\noindent
Before comparison, both texts are normalised with Unicode NFC normalisation and case folding. It is then tokenised using a Unicode-aware regular expression that extracts word-character sequences while excluding punctuation. 

\noindent 
For each example, let $Q$ be the set of distinct question tokens and $C$ the set of distinct context tokens. The classifier computes the number of shared tokens,
\[
s = |Q \cap C|.
\]
It predicts answerable if $s \geq 1$, and unanswerable otherwise:
\[
    \hat{y} =
        \begin{cases}
            \text{answerable}, & \text{if } |Q \cap C| \geq 1, \\
            \text{unanswerable}, & \text{otherwise.}
        \end{cases}
\]
The threshold of one shared token is fixed before the labeled validation evaluation. Repeated tokens are counted only once because and punctuation-only tokens, including question marks, do not create an overlap.

\subsubsection*{Results}

\noindent
The deterministic rule obtains an overall accuracy of 0.1524 and macro-F1 of 0.1408 (Table~\ref{table:week_36_baseline_results}). It has unanswerable recall of 0.9451 but answerable recall of only 0.0212. The same pattern occurs for all three languages, although Telugu has the highest macro-F1 (0.2265).

\begin{table}[!h]
    \centering
    \small
    \begin{tabular}{lrrrrrrrrr}
        \toprule
        & & & &
        \multicolumn{3}{c}{Answerable}
        & \multicolumn{3}{c}{Unanswerable} \\
        \cmidrule(lr){5-7}\cmidrule(lr){8-10}
        Group & $N$ & Accuracy & Macro-F1
        & P & R & F1
        & P & R & F1 \\
        \midrule
            Overall & 1155 & 0.1524 & 0.1408 & 0.7000 & 0.0212 & 0.0411 & 0.1378 & 0.9451 & 0.2405 \\
            ar & 415 & 0.1398 & 0.1291 & 1.0000 & 0.0165 & 0.0325 & 0.1271 & 1.0 & 0.2256 \\
            ko & 356 & 0.0562 & 0.0547 & 0.6000 & 0.0089 & 0.0175 & 0.0484 & 0.8947 & 0.0919 \\
            te & 384 & 0.2552 & 0.2265 & 0.6316 & 0.0412 & 0.0774 & 0.2356 & 0.9247 & 0.3755 \\
        \bottomrule
    \end{tabular}
    \caption{Validation results for the deterministic lexical-overlap answerability rule. P, R, and F1 denote precision, recall, and F1, respectively. Macro-F1 is the unweighted mean of the answerable and unanswerable F1 scores.}
    \label{table:week_36_baseline_results}
\end{table}

\subsubsection*{Analysis}

\noindent
As we can see, this rule has high recall for unanswerable examples primarily because it lacks lexical evidence for most question/context pairs. Questions are in Arabic, Korean, or Telugu while contexts are English, so direct word overlap is rarely captured in translation equivalent terms. Therefore, this rule incorrectly rejects most answerable examples. 

\noindent
It can only exploit surface forms shared across languages, such as Latin-script names and some numbers. It also produces false positives when an incidental shared surface token occurs although the context does not answer the question. This simple baseline clearly demonstrates why a cross-lingual answerability model needs a representation that can relate questions to English contexts.
```

Does this correctly explain it can you sanity check it against my code?

Yes—your write-up is **consistent** with the code you shared, and the reported metrics are internally consistent. I would make a few wording changes, distinguish observed results from proposed explanations, and soften the final conclusion.

## 1. Implementation: correct, with two small edits

Your description matches the code provided you ran:

```python
Classifier(
    casefolding=True,
    min_shared_tokens=1,
    normalization_form="NFC",
)
```

In particular:

- Normalization occurs before case folding.
- The regex extracts Unicode word-character sequences, excluding underscores.
- Converting tokens to sets removes duplicates.
- The decision is exactly $|Q \cap C| \geq 1$.
- Neither gold labels nor answer offsets enter `predict()`.

The sentence below contains an unfinished clause:

> Repeated tokens are counted only once because and punctuation-only tokens…

Replace the opening implementation paragraph and that sentence with:

```latex
\noindent
Before comparison, both texts undergo Unicode NFC normalisation
followed by case folding. Each text is then tokenised using the
Unicode-aware regular expression \verb|[^\W_]+|, which extracts
word-character sequences while excluding underscores and punctuation.

...

\noindent
The threshold of one shared token was fixed before evaluation on
labelled validation data. Repeated tokens contribute only once
because the token sequences are converted into sets.
Punctuation-only strings, including question marks, do not produce
tokens and therefore cannot create overlap.
```

Only retain the claim about fixing the threshold before evaluation if that accurately describes your process.

## 2. Results: the numbers check out

I checked whether your rounded metrics could arise from consistent integer confusion matrices. They correspond to the following counts:


| Group | TP | TN | FP | FN |
| :-- | --: | --: | --: | --: |
| Overall | 21 | 155 | 9 | 970 |
| ar | 6 | 52 | 0 | 357 |
| ko | 3 | 17 | 2 | 334 |
| te | 12 | 86 | 7 | 279 |

These counts reproduce every metric in your table to the displayed precision, and the language-specific counts sum to the overall counts.

This establishes internal consistency—not that the correct dataset rows and labels were loaded. To verify that final point, compare these counts against the `tp`, `tn`, `fp`, and `fn` columns your `evaluate()` function already returns.

Two useful interpretations follow:

- The rule predicts unanswerable for 1,125 of 1,155 examples, or 97.4%.
- It rejects 970 of the 991 gold-answerable examples.

That explains how high unanswerable recall coexists with very low accuracy: most examples are answerable, but the rule almost always predicts unanswerable.

Your evaluation table includes the required counts, class-specific metrics and macro-F1, overall and by language. Those match the assignment’s reporting requirements.[^4_1]

Minor formatting fix: change Arabic unanswerable recall from `1.0` to `1.0000` for consistency.

## 3. Analysis: distinguish evidence from explanation

Your central interpretation is sensible, but three statements need qualification.

### Cross-lingual mismatch is an explanation, not directly measured by the table

The table shows that the classifier rarely predicts answerable. Your code also tells us that differently written equivalents cannot match. Together these support a plausible explanation based on the language mismatch.

However, the table alone does not establish exactly which token types caused the matches. Use your saved `shared_tokens` to inspect that before presenting Latin-script names or numbers as observed findings.

### False positives do not yet establish their cause

Your results establish that there are nine false positives. They do not establish that all nine result from an “incidental shared surface token” unless you have inspected those examples.

You can safely write:

> Shared-token overlap is insufficient to establish contextual support, as shown by the nine false-positive predictions.

Then describe the specific shared-token types if your inspection confirms them.

### The final conclusion is too strong

This sentence overreaches:

> This simple baseline clearly demonstrates why a cross-lingual answerability model needs a representation that can relate questions to English contexts.

Your experiment demonstrates a limitation of this direct exact-overlap rule, not a necessity for a particular model architecture. Translation, dictionaries or other cross-lingual processing could also address the mismatch.

A more defensible conclusion is:

> These results illustrate the limitations of direct exact-token overlap for cross-lingual answerability and motivate methods that can connect differently expressed question and context information.

## 4. Suggested revised analysis

The following version stays close to your original while separating measured outcomes from interpretation:

```latex
\subsubsection*{Analysis}

\noindent
The rule predicts unanswerable for 1,125 of the 1,155 validation
examples (97.4\%). Consequently, it achieves high unanswerable
recall (0.9451) but rejects 970 of the 991 answerable examples,
yielding answerable recall of only 0.0212. Since most validation
examples are answerable, this prediction pattern also produces
low overall accuracy.

\noindent
A plausible explanation is the mismatch between the question
languages and the English contexts. The rule compares exact
normalised token strings and uses neither translation nor
cross-lingual representations. It therefore cannot match
translation-equivalent terms written differently. It can exploit
identical surface forms, such as Latin-script strings or numbers
when these occur unchanged in both texts, but cannot determine
whether their relationship supports the requested answer.
The nine false-positive predictions further show that token
overlap alone is insufficient to establish contextual support.

\noindent
Telugu has the highest macro-F1, but this should not be interpreted
as evidence that the rule handles Telugu semantics better:
the rule has no semantic or language-specific processing, and
the language subsets differ in class proportions and shared-token
patterns. Overall, these results illustrate the limitations of
direct exact-token overlap for cross-lingual answerability.
```

One final report-level point: your baseline subsection is appropriate, but the assignment asks for the final report to be organized as a coherent investigation rather than one section per teaching week. You can keep this content while eventually replacing the `W36` heading with a thematic heading such as `Data and answerability baselines`.[^4_1]

<div align="center">⁂</div>

[^4_1]: instructions.pdf

