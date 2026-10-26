import argparse

from ku_nlp_group_25_project.utilities.load_dataset import load_dataset
from ku_nlp_group_25_project.W37_Evaluate_Model import (
    LANGUAGES,
    MODEL_DIR,
    evaluate,
    get_model_predictor,
    save_predictions_csv,
)


# Evaluate all validation contexts with an empty question using the original decoder.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument(
        "--predictions-out",
        default="outputs/w43_question_ablation_no_question.csv",
    )
    args = parser.parse_args()

    _, validation = load_dataset()
    validation = validation[validation["lang"].isin(LANGUAGES)].copy()
    assert len(validation) == 1155, "Validation split has changed"
    assert set(validation["lang"]) == set(LANGUAGES)

    frozen_predictor = get_model_predictor(args.model_dir)

    # Remove only the question; keep context, checkpoint and decoder unchanged.
    def predict_without_question(question, context):
        return frozen_predictor("", context)

    records = evaluate(validation, predict_without_question)
    save_predictions_csv(records, args.predictions_out)


if __name__ == "__main__":
    main()
