from datasets import load_dataset as load
from pandas import DataFrame

from ku_nlp_group_25_project.constants import LANGUAGES


def load_dataset() -> tuple[DataFrame, DataFrame]:
    """
    Loads the TyDi XOR RC dataset and returns the training and validation dataframes.

    Returns:
        tuple[DataFrame, DataFrame]: The training and validation dataframes.
    """
    dataset = load("coastalcph/tydi_xor_rc")
    training_data_frame = dataset["train"].to_pandas()
    validation_data_frame = dataset["validation"].to_pandas()

    return (
        training_data_frame[training_data_frame["lang"].isin(LANGUAGES)].reset_index(drop=True),
        validation_data_frame[validation_data_frame["lang"].isin(LANGUAGES)].reset_index(drop=True)
    )
