from typing import Dict

from sklearn.pipeline import Pipeline
from zenml.steps import Output, step

from titanic_model.model_registry.mlflow_client_deploying import (
    get_inputs_example,
    get_meta,
    get_model,
    get_requirements_path,
)


@step
def model_fetcher() -> Output(
    model=Pipeline,
    model_metadata=Dict,
    model_requirements=str,
):
    """Fetches the model from model registry

    Returns:
        model (sklearn.pipeline.Pipeline): model classifier
        model_metadata (Dict): metadata of the model
        model_requirements (str): path to the requirements.txt
            (dependency file) to run the model
    """
    model_name = "titanic_model"  # TODO: config
    stage = "Production"  # TODO: config

    model = get_model(model_name=model_name, stage=stage)
    model_requirements = get_requirements_path(
        model_uri=f"models:/{model_name}/{stage}"
    )
    model_inputs_example = get_inputs_example(model_name=model_name, stage=stage)
    model_metadata = get_meta(model_name=model_name, stage=stage)

    model_metadata = {
        "bento_model_name": model_metadata.name,
        "mlflow_model_version": model_metadata.version,
        "mlflow_model_stage": model_metadata.current_stage,
        "inputs_example": model_inputs_example,
    }

    return model, model_metadata, model_requirements
