import json
from typing import Dict

import bentoml
import pandas
from bentoml.io import JSON

with open("meta.json", "r") as file:
    metadata = json.load(file)

model_name = metadata["bento_model_name"]
version = metadata["mlflow_model_version"]
inputs_example = metadata["inputs_example"]
description = "Titanic model"

DOC = f"""
    Model name: {model_name}.
    Model version mlflow: {version}.
    Description: {description}.
"""

runner = bentoml.sklearn.get(f"{model_name}:latest").to_runner()
svc = bentoml.Service(f"{model_name}_service", runners=[runner])
Input = JSON.from_sample(inputs_example)


@svc.api(input=Input, output=JSON(), doc=DOC, name=model_name)
def predict_bentoml(input_data: Dict) -> Dict[str, float]:
    input_df = pandas.DataFrame([input_data])
    return {"prediction": runner.predict_proba.run(input_df)[:, 1]}
