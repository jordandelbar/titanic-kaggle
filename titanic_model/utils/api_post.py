"""Manage the post request to the web api service."""
import os

import pandas
import requests


def get_api_response(row: pandas.Series) -> float:
    """Get the response from the web api service.

    Args:
        row (pandas.Series): row of the test dataset
        url (str): url of the web api service

    Returns:
        response (dict): response from the web api service
    """
    url = os.getenv("WEB_SERVICE_URL")
    data = row.to_json()
    api_response = requests.post(url=url, data=data).json()
    api_response = api_response["prediction"][0]
    return api_response
