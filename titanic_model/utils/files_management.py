import os
import zipfile
from pathlib import Path
from typing import List

from kaggle.api.kaggle_api_extended import KaggleApi


def return_datasets_path() -> str:
    """Returns the datasets folder path

    Returns:
        str: path of the datasets folder
    """
    return Path(__file__).resolve().parents[1] / "datasets/"


def check_if_files_exists(file_list: List[str]) -> bool:
    """Returns True if all the files in the list exists

    Args:
        path (str): path to the directory to be checked
        file_list (List[str]): list of files to check

    Returns:
        bool: True if the files exist, False otherwise
    """
    return all([os.path.isfile(return_datasets_path() / file) for file in file_list])


def download_files_from_kaggle(kaggle_competition: str) -> None:
    """Download the dataset of a kaggle competition, unzip it
       and remove the zip file afterwards

    Args:
        kaggle_competition (str): name of the kaggle competition to
                                  be downloaded
    """
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    kaggle_api.competition_download_files(
        kaggle_competition,
        path=return_datasets_path(),
    )

    with zipfile.ZipFile(
        return_datasets_path() / f"{kaggle_competition}.zip",
        "r",
    ) as zipref:
        zipref.extractall(return_datasets_path())

    os.remove(return_datasets_path() / f"{kaggle_competition}.zip")

    return None
