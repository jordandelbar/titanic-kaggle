import json
import os
from pathlib import Path
from typing import Dict

import bentoml
from sklearn.pipeline import Pipeline
from zenml.steps import Output, step


@step
def bento_builder(
    model: Pipeline, model_metadata: Dict, model_requirements: str
) -> Output(bentoml_service_name=str):
    """Saves a bentoml model and build a bento service from that model

    Args:
        model (sklearn.pipeline.Pipeline): model classifier
        model_metadata (Dict): metadata of the model
        model_requirements (str): path to the requirements.txt
            (dependency file) to run the model
    Returns:
        bentoml_service_name (str): name of the bentoml service builded
    """
    bento_model_name = model_metadata["bento_model_name"]
    bentoml_service_name = f"{bento_model_name}_service"
    signatures = {"predict_proba": {"batchable": True, "batch_dim": 0}}

    # Saving bentoml model
    bentoml.sklearn.save_model(
        model=model,
        signatures=signatures,
        name=bento_model_name,
    )

    # Persist input examples into bento workdir
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

    # Remove input examples file
    os.remove(model_metadata_file_path)

    return bentoml_service_name
