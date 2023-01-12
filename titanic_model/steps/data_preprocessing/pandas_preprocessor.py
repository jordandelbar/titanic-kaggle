import numpy
import pandas
from zenml.steps import Output, step

from titanic_model.config.core import config


@step
def preprocessor(X: pandas.DataFrame) -> Output(X_pp=pandas.DataFrame):
    """Preprocesses the data before the training of our model

    Args:
        X (pandas.DataFrame): train or test dataframe

    Returns:
        X_pp (pandas.DataFrame): preprocessed train or test dataframe
    """
    X = X.copy()

    # Is the passenger a baby
    X["is_baby"] = numpy.where(X["Age"] < 5, 1, 0)

    # Was the passenger travelling alone
    X["alone"] = numpy.where((X["SibSp"] == 0) & (X["Parch"] == 0), 1, 0)

    # Family member total
    X["family"] = X["SibSp"] + X["Parch"]

    # Create a title column
    X["title"] = X["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)  # noqa: W605
    X["title"] = X["title"].replace("Mlle", "Miss")
    X["title"] = X["title"].replace("Ms", "Miss")
    X["title"] = X["title"].replace("Mme", "Mrs")
    X["title"] = X["title"].replace("Don", "Mr")
    X["title"] = X["title"].replace("Dona", "Mrs")

    # Drop features not useful anymore
    X.drop(config.features_to_drop, axis=1, inplace=True)

    return X
