import json
import os
from pathlib import Path
from typing import Dict

import bentoml
from zenml.steps import Output, step


@step
def bento_builder(
    model_name: str, model_input_example: Dict, model_requirements: str
) -> Output(bentoml_service_name=str):

    bentoml_service_name = "titanic-service"

    input_example_file_path = str(
        Path(__file__).resolve().parents[2] / "bento_service/meta.json"
    )

    with open(input_example_file_path, "w") as fp:
        json.dump(model_input_example, fp)

    bentoml.bentos.build(
        service="service.py:svc",
        python={"requirements_txt": model_requirements},
        build_ctx=str(Path(__file__).resolve().parents[2] / "bento_service"),
    )

    os.remove(input_example_file_path)

    return bentoml_service_name
