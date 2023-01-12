import pandas
from zenml.steps import Output, step

from titanic_model.config.core import config
from titanic_model.utils.files_management import (
    check_if_files_exists,
    download_files_from_kaggle,
    return_datasets_path,
)


@step
def data_loader() -> Output(
    train=pandas.DataFrame, target=pandas.Series, test=pandas.DataFrame
):
    """Loads the data from titanic files

    Returns:
        train (pandas.DataFrame): train data without the target column
        target (pandas.Series): target columns of our train dataframe.
        test (pandas.DataFrame): test data (to return to the Titanic competition).
    """

    if not check_if_files_exists(config.data_files.values()):
        download_files_from_kaggle(kaggle_competition=config.kaggle_competition)

    train = pandas.read_csv(return_datasets_path() / config.data_files["training_data"])
    test = pandas.read_csv(return_datasets_path() / config.data_files["testing_data"])

    target = train[config.target]
    train.drop(config.target, axis=1, inplace=True)

    return train, target, test
