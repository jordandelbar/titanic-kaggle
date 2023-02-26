"""Build the BentoML service image."""
import json
import os
from pathlib import Path

import bentoml
import mlflow.pyfunc
from model_registry.mlflow_client_deploying import (
    get_inputs_example,
    get_meta,
    get_requirements_path,
)
from zenml.steps import step


@step
def bento_builder() -> None:
    """Save a bentoml model and build a bento service from that model.

    Args:
        model (sklearn.pipeline.Pipeline): model classifier
        model_metadata (Dict): metadata of the model
        model_requirements (str): path to the requirements.txt
            (dependency file) to run the model
    Returns:
        None
    """
    model_name = "titanic_model"  # TODO: config
    stage = "Production"  # TODO: config

    mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
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

    bento_model_name = model_metadata["bento_model_name"]
    bentoml_service_name = f"{bento_model_name}_service"
    signatures = {"predict": {"batchable": True, "batch_dim": 0}}

    # Saving bentoml model
    bentoml.mlflow.import_model(
        name=bento_model_name,
        model_uri=f"models:/{model_name}/{stage}",
        signatures=signatures,
    )

    # Persist model metadata into bento workdir
    model_metadata_file_path = str(
        Path(__file__).resolve().parents[2] / "bento_service/meta.json"
    )

    with open(model_metadata_file_path, "w") as fp:
        json.dump(model_metadata, fp)

    # Build bento service
    bentoml.bentos.build(
        service="service.py:svc",
        python={"requirements_txt": model_requirements},
        build_ctx=str(Path(__file__).resolve().parents[2] / "bento_service"),
    )

    # Remove metadata file
    os.remove(model_metadata_file_path)

    # Containerize the bento service
    bentoml.container.build(bento_tag=bentoml_service_name)
