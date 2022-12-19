from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import Output, step


@step
def data_splitter(
    train: pd.DataFrame, target: pd.Series
) -> Output(
    X_train=pd.DataFrame, y_train=pd.Series, X_test=pd.DataFrame, y_test=pd.Series
):
    """Load the data from titanic files"""

    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.33)

    return X_train, y_train, X_test, y_test
