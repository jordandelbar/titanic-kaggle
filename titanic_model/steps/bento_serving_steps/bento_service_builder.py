import json
import os
from pathlib import Path
from typing import Dict

import bentoml
from sklearn.pipeline import Pipeline
from zenml.steps import Output, step


@step
def bento_builder(
    model: Pipeline, model_input_example: Dict, model_requirements: str
) -> Output(bentoml_service_name=str):

    bento_model_name = "titanic_model"
    bentoml_service_name = "titanic-service"
    signatures = {"predict_proba": {"batchable": True, "batch_dim": 0}}

    # Saving bentoml model
    bentoml.sklearn.save_model(
        model=model,
        signatures=signatures,
        name=bento_model_name,
    )

    # Persist input examples into bento workdir
    input_example_file_path = str(
        Path(__file__).resolve().parents[2] / "bento_service/input_examples.json"
    )

    with open(input_example_file_path, "w") as fp:
        json.dump(model_input_example, fp)

    # Build bento service
    bentoml.bentos.build(
        service="service.py:svc",
        python={"requirements_txt": model_requirements},
        build_ctx=str(Path(__file__).resolve().parents[2] / "bento_service"),
    )

    # Remove input examples file
    os.remove(input_example_file_path)

    return bentoml_service_name
