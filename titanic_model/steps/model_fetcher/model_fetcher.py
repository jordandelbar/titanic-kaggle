from typing import Dict

from sklearn.pipeline import Pipeline
from zenml.steps import Output, step

from titanic_model.mlflow_model_management.mlflow_metadata import (
    get_model,
    get_model_config,
    get_requirements_path,
)


@step
def model_fetcher() -> Output(
    model=Pipeline, model_config=Dict, model_requirements=str
):

    model_name = "titanic-model"  # TODO: config
    stage = "Production"  # TODO: config

    model = get_model(model_name=model_name, stage=stage)
    model_config = get_model_config(model_name=model_name, stage=stage)
    model_requirements = get_requirements_path(
        model_uri=f"models:/{model_name}/{stage}"
    )

    print(model_config)
    print(model_requirements)

    return model, model_config, model_requirements
