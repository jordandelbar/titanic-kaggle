from pathlib import Path

import pandas as pd
from zenml.steps import Output, step


@step
def data_loader() -> Output(train=pd.DataFrame, target=pd.Series, test=pd.DataFrame):
    """Load the data from titanic files"""

    train = pd.read_csv(Path(__file__).resolve().parents[3] / "datasets/train.csv")
    test = pd.read_csv(Path(__file__).resolve().parents[3] / "datasets/test.csv")

    target = train["Survived"]
    train.drop("Survived", axis=1, inplace=True)

    return train, target, test
