"""Split the data in train/test sets."""
import pandas
import polars
from config.core import config
from sklearn.model_selection import train_test_split
from zenml.steps import Output, step


@step
def data_splitter(
    train: polars.DataFrame, target: polars.DataFrame
) -> Output(
    X_train=pandas.DataFrame,
    X_test=pandas.DataFrame,
    y_train=pandas.DataFrame,
    y_test=pandas.DataFrame,
):
    """Split the data into training and testing dataframes.

    Args:
        train (polars.DataFrame): train data without the target column
        target (polars.Series): target column of our train dataframe.

    Returns:
        X_train (polars.DataFrame): train dataframe to be used for model training
        X_test (polars.DataFrame): test dataframe to be used for model evaluation
        y_train (polars.DataFrame): target series to be used for model training
        y_test (polars.DataFrame): target series to be used for model evaluation
    """
    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=config.test_size, random_state=config.random_state
    )

    return (
        X_train.to_pandas(),
        X_test.to_pandas(),
        y_train.to_pandas(),
        y_test.to_pandas(),
    )
