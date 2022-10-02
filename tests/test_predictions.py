import numpy as np
from titanic_model import __version__ as _version
from titanic_model.processing.data_manager import (load_dataset,
                                                   load_pipeline)
from titanic_model.predict import make_prediction
from titanic_model.config.core import config


def test_if_any_null_after_fe():

    pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    _titanic_pipe = load_pipeline(file_name=pipeline_file_name)
    test_dataset = load_dataset(file_name=config.app_config.testing_data)

    results = _titanic_pipe[:-2].transform(test_dataset)

    is_NaN = results.isnull()

    row_has_NaN = is_NaN.any(axis=1)

    rows_with_NaN = results[row_has_NaN]
    print(rows_with_NaN)

    assert results.isnull().values.any() is not True


def test_if_any_null_before_pred():

    pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    _titanic_pipe = load_pipeline(file_name=pipeline_file_name)
    test_dataset = load_dataset(file_name=config.app_config.testing_data)
    print(test_dataset.head())

    results = _titanic_pipe[:-1].transform(test_dataset)
    print(results)

    assert np.isnan(results).any() is not True


def test_prediction():

    test_dataset = load_dataset(file_name=config.app_config.testing_data)

    results = make_prediction(test_dataset)

    assert len(results["predictions"]) == test_dataset.shape[0]
    assert type(results["predictions"][0]) == np.float64
