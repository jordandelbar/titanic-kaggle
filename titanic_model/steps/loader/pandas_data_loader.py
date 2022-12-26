from pathlib import Path

import pandas
from zenml.steps import Output, step

from titanic_model.config.core import config


@step
def data_loader() -> Output(
    train=pandas.DataFrame, target=pandas.Series, test=pandas.DataFrame
):
    """Load the data from titanic files

    Returns:
        train (pandas.DataFrame): train data without the target column
        target (pandas.Series): target columns of our train dataframe.
        test (pandas.DataFrame): test data (to return to the Titanic competition).
    """

    train = pandas.read_csv(
        Path(__file__).resolve().parents[3] / f"datasets/{config.training_data}"
    )
    test = pandas.read_csv(
        Path(__file__).resolve().parents[3] / f"datasets/{config.testing_data}"
    )

    target = train[config.target]
    train.drop(config.target, axis=1, inplace=True)

    return train, target, test
