"""Infer the data to the service."""
import pandas
from zenml.steps import Output, step

from titanic_model.utils import get_api_response, return_datasets_path


@step
def inferer(test: pandas.DataFrame) -> Output(infered_test=pandas.DataFrame):
    """Infer the test dataset by using our web api service.

    Args:
        test (pandas.DataFrame): test data (to return to the Titanic competition).

    Returns:
        infered_test (pandas.DataFrame): test data with predicted column added
    """
    final = test.copy()
    final["prediction"] = ""
    final["prediction"] = test.apply(get_api_response, axis=1)

    final.to_csv(return_datasets_path() / "infered_test.csv")

    return final
