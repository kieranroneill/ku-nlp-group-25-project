import pandas as pd

from ku_nlp_group_25_project.baselines import Classifier


class TestClassifierConstructor:
    def test_should_fail_with_less_than_one_minimum_tokens(self):
        try:
            Classifier(
                min_shared_tokens=0
            )
        except ValueError:
            return

        raise AssertionError(
            "Expected a ValueError for min_shared_tokens=0."
        )


class TestClassifierBehaviours:
    def test_punctuation_is_omitted(self):
        classifier = Classifier()
        prediction = classifier.predict(
            context="Who?",
            question="من؟",
        )

        assert prediction.answerable is False
        assert prediction.shared_token_count == 0

    def test_uses_case_insensitively(self):
        classifier = Classifier()
        prediction = classifier.predict(
            context="nasa launched Voyager.",
            question="NASA?",
        )

        assert prediction.answerable is True
        assert prediction.shared_tokens == ["nasa"]

    def test_repeated_question_token_counts_once(self):
        classifier = Classifier()
        prediction = classifier.predict(
            question="NASA? NASA?? NASA???",
            context="NASA launched Voyager.",
        )

        assert prediction.shared_token_count == 1
        assert prediction.shared_tokens == ["nasa"]

    def test_empty_question_has_zero_overlap(self):
        classifier = Classifier()

        prediction = classifier.predict(
            question="",
            context="The answer is in this context.",
        )

        assert prediction.answerable is False
        assert prediction.question_token_count == 0
        assert prediction.question_overlap_ratio == 0.0

    def test_deterministic_predictions(self):
        classifier = Classifier()
        first_prediction = classifier.predict(
            question="NASA?",
            context="NASA launched Voyager.",
        )
        second_prediction = classifier.predict(
            question="NASA?",
            context="NASA launched Voyager.",
        )

        assert first_prediction == second_prediction


class TestClassifierDataframe:
    def test_prediction_does_not_require_labels(self):
        classifier = Classifier()
        dataframe = pd.DataFrame(
            {
                "question": ["NASA?", "Who?"],
                "context": [
                    "NASA launched Voyager.",
                    "The person was Alice.",
                ],
            }
        )
        result = classifier.predictions(dataframe=dataframe)

        assert "predicted_answerable" in result.columns
        assert "shared_token_count" in result.columns
        assert len(result) == 2
