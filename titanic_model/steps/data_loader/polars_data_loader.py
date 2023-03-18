"""Download & Load the data."""
import polars
from config.core import config
from utils.files_management import (
    check_if_files_exists,
    download_files_from_kaggle,
    return_datasets_path,
)
from zenml.steps import Output, step


@step
def data_loader() -> (
    Output(train=polars.DataFrame, target=polars.DataFrame, test=polars.DataFrame)
):
    """Load the data from titanic files.

    Returns:
        train (pandas.DataFrame): train data without the target column
        target (pandas.Series): target columns of our train dataframe.
        test (pandas.DataFrame): test data (to return to the Titanic competition).
    """
    if not check_if_files_exists(config.data_files.values()):
        download_files_from_kaggle(kaggle_competition=config.kaggle_competition)

    train = polars.read_csv(return_datasets_path() / config.data_files["training_data"])
    test = polars.read_csv(return_datasets_path() / config.data_files["testing_data"])

    target = train.select(config.target)
    train = train.drop(config.target)

    return train, target, test
