import os
import zipfile
from pathlib import Path

import pandas
from kaggle.api.kaggle_api_extended import KaggleApi
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

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    kaggle_api.competition_download_files(
        config.kaggle_competition,
        path=Path(__file__).resolve().parents[2] / "datasets/",
    )
    with zipfile.ZipFile(f"./{config.kaggle_competition}.zip", "r") as zipref:
        zipref.extractall(Path(__file__).resolve().parents[2] / "datasets/")

    os.remove(
        Path(__file__).resolve().parents[2]
        / "datasets/"
        / f"{config.kaggle_competition}.zip"
    )

    train = pandas.read_csv(
        Path(__file__).resolve().parents[3] / f"datasets/{config.training_data}"
    )
    test = pandas.read_csv(
        Path(__file__).resolve().parents[3] / f"datasets/{config.testing_data}"
    )

    target = train[config.target]
    train.drop(config.target, axis=1, inplace=True)

    return train, target, test
