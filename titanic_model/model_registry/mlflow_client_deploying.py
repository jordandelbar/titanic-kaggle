"""Mlflow functions for model deployment."""
import importlib
import json
import logging
import os
from typing import Any, Dict

import mlflow
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)

MODEL_CONFIG_NAME = "model_config.json"


def check_model_exists(model_name: str) -> bool:
    """Check if a model exists under that name.

    Args:
        model_name (str): model name to be checked

    Returns:
        bool: True if the model exists, False otherwise
    """
    client = mlflow.MlflowClient()
    if len(client.search_registered_models(f"name='{model_name}'")):
        return True
    else:
        logger.warning(f"There is no registered model with name: {model_name}!")
        models = client.search_registered_models()
        logger.debug(f"Existing models: {models}!")
        return False


def check_model_stage_exist(model_name: str, stage: str) -> bool:
    """Check if a model is available on a certain stage.

    Args:
        model_name (str): model name to be checked
        stage (str): stage of the model to be checked

    Returns:
        bool: True if the model exists at that stage, False otherwise
    """
    client = mlflow.MlflowClient()
    if check_model_exists(model_name=model_name):
        if len(client.get_latest_versions(name=model_name, stages=[stage])):
            return True
    else:
        logger.warning(f"There is no model on stage: {stage}!")
        return False


def get_meta(model_name: str, stage: str) -> ModelVersion:
    """Get the meta data of a model.

    Args:
        model_name (str): model name to get metadata from
        stage (str): stage of the model

    Raises:
        mlflow.MlflowException: raises an error if there is no
            model at this stage

    Returns:
        ModelVersion: return ModelVersion class with model metadata
    """
    client = mlflow.MlflowClient()
    if check_model_stage_exist(model_name=model_name, stage=stage):
        return client.get_latest_versions(name=model_name, stages=[stage])[0]
    else:
        err_msg = f"There is no model on stage: {stage}!"
        raise mlflow.MlflowException(message=err_msg)


def get_model(model_name: str, stage: str) -> Any:
    """Download the model from MLflow.

    Args:
        model_name (str): name of the model to download
        stage (str): stage of the model to download

    Returns:
        Any: model
    """
    version_meta = get_meta(model_name=model_name, stage=stage)
    run_id = version_meta.run_id
    model_uri = version_meta.source

    run = mlflow.get_run(run_id=run_id)
    run_data = json.loads(run.data.tags["mlflow.log-model.history"])[0]
    loader_module = run_data["flavors"]["python_function"]["loader_module"]
    mlflow_loader = importlib.import_module(loader_module)
    model = mlflow_loader.load_model(model_uri)

    return model


def get_inputs_example(model_name: str, stage: str) -> Dict[str, Any]:
    """Get inputs example for the service.

    Args:
        model_name (str): name of the model to get inputs example from
        stage (str): stage of the model

    Raises:
        ValueError: raised if the format for handling the inputs example is
            not implemented
        mlflow.MlflowException: raised if there is no inputs example in MLflow

    Returns:
        Dict[str, Any]: returns a Json (dict) of the inputs example
    """
    version_meta = get_meta(model_name=model_name, stage=stage)
    run_id = version_meta.run_id
    path_artifacts = version_meta.source

    run = mlflow.get_run(run_id=run_id)
    run_data = json.loads(run.data.tags["mlflow.log-model.history"])[0]
    example_info = run_data.get("saved_input_example_info", None)
    if example_info is not None:
        file_name = example_info["artifact_path"]
        example = mlflow.artifacts.load_dict(os.path.join(path_artifacts, file_name))
        example_type = example_info["type"]
        if example_type == "dataframe":
            columns = example["columns"]
            data = example["data"][0]
            example_json = {}
            for i, col in enumerate(columns):
                # workaround for pydantic model bug
                val = data[i] + 0.1 if isinstance(data[i], float) else data[i]
                example_json[col] = val
            return [example_json]
        else:
            err_msg = "Example type: {example_type}, handling is not implemented"
            raise ValueError(err_msg)
    else:
        raise mlflow.MlflowException("Data example is not provided")


def get_requirements_path(model_uri: str) -> str:
    """Get the requirements of a requirements.txt for model dependencies.

    Args:
        model_uri (str): uri of the model

    Returns:
        str: path of the dowloaded requirements.txt
    """
    return mlflow.pyfunc.get_model_dependencies(model_uri)


def get_model_config(model_name: str, stage: str) -> Dict[str, Any]:
    """Get the model config.

    Args:
        model_name (str): name of the model to get the config from
        stage (str): stage of the model

    Returns:
        Dict: dictionnary with the model config
    """
    meta = get_meta(model_name=model_name, stage=stage)
    model_config_uri = os.path.join(meta.source.replace("model", ""), MODEL_CONFIG_NAME)
    return mlflow.artifacts.load_dict(artifact_uri=model_config_uri)
