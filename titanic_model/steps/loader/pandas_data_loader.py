from pathlib import Path

import pandas
from zenml.steps import Output, step


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

    train = pandas.read_csv(Path(__file__).resolve().parents[3] / "datasets/train.csv")
    test = pandas.read_csv(Path(__file__).resolve().parents[3] / "datasets/test.csv")

    target = train["Survived"]
    train.drop("Survived", axis=1, inplace=True)

    return train, target, test
