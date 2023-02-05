"""Infer the data to the service."""
import os

import pandas
import requests
from zenml.steps import Output, step

from titanic_model.utils.files_management import return_datasets_path


@step
def inferer(test: pandas.DataFrame) -> Output(infered_test=pandas.DataFrame):
    """Infer the test dataset by using our web api service.

    Args:
        test (pandas.DataFrame): test data (to return to the Titanic competition).

    Returns:
        infered_test (pandas.DataFrame): test data with predicted column added
    """
    url = os.getenv("WEB_SERVICE_URL")
    data = test.to_json(orient="records")
    response_data = requests.post(url=url, data=data)
    test["surviving_probability_prediction"] = response_data.json()["prediction"]

    test.to_csv(return_datasets_path() / "infered_test.csv")

    return test
