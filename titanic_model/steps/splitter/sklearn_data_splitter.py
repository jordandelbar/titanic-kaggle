import pandas
from sklearn.model_selection import train_test_split
from zenml.steps import Output, step

from titanic_model.config.core import config


@step
def data_splitter(
    train: pandas.DataFrame, target: pandas.Series
) -> Output(
    X_train=pandas.DataFrame,
    X_test=pandas.DataFrame,
    y_train=pandas.Series,
    y_test=pandas.Series,
):
    """Split the data into training and testing dataframes

    Args:
        train (pandas.DataFrame): train data without the target column
        target (pandas.Series): target columns of our train dataframe.

    Returns:
        X_train (pandas.DataFrame): train dataframe to be used for model training
        X_test (pandas.DataFrame): test dataframe to be used for model evaluation
        y_train (pandas.Series): target series to be used for model training
        y_test (pandas.Series): target series to be used for model evaluation
    """

    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=config.test_size, random_state=config.random_state
    )

    return X_train, X_test, y_train, y_test
