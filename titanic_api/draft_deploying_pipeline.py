import os
from typing import Dict, NoReturn

import bentoml
import mlflow
from sklearn.pipeline import Pipeline


def exec_command(command: str):
    if os.system(command) != 0:
        msg = f"Command excecution failed! Command: {command}"
        raise RuntimeError(msg)


# 1. Retrieve the model and metadata from mlflow


def import_mlflow_model(mlflow_model_name, model_stage, **kwargs) -> Pipeline:

    model = mlflow.sklearn.load_model(
        model_uri=f"models:/{mlflow_model_name}/{model_stage}"
    )
    return model


# 2. Save the model as a bento model with correct predict signature


def save_bento_model(model: Pipeline, signatures: Dict, name: str) -> NoReturn:

    bentoml.sklearn.save_model(
        model=model,
        signatures=signatures,
        name=name,
    )


# 3. Build a bento


def build_bento_service(model_name: str) -> NoReturn:

    bentoml.bentos.build(
        service="service.py:svc",
        python={
            "packages": [
                "scikit-learn",
                "pandas",
                "numpy",
                "xgboost",
                "feature_engine",
                "pydantic",
            ]
        },
    )


# 4. Containerize as a docker image


def build_docker_image(service_name: str) -> NoReturn:

    exec_command(f"bentoml containerize {service_name}")


def main() -> NoReturn:
    mlflow_model_name = "titanic-model"
    bento_model_name = "titanic_model"
    model = import_mlflow_model(
        mlflow_model_name=mlflow_model_name, model_stage="Production"
    )
    save_bento_model(
        model=model,
        signatures={"predict_proba": {"batchable": True, "batch_dim": 0}},
        name=bento_model_name,
    )
    build_bento_service(model_name=bento_model_name)
    build_docker_image(service_name="titanic-service")


if __name__ == "__main__":
    main()
