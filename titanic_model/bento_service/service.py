import json
from typing import Dict

import bentoml
import numpy
import pandas
from bentoml.io import JSON, NumpyNdarray

with open("meta.json", "r") as file:
    serving_meta = json.load(file)


titanic_runner = bentoml.sklearn.get("titanic_model:latest").to_runner()

svc = bentoml.Service("titanic-service", runners=[titanic_runner])

Input = JSON.from_sample(serving_meta)


@svc.api(input=Input, output=NumpyNdarray())
def predict_bentoml(input_data: Dict) -> numpy.ndarray:
    input_df = pandas.DataFrame([input_data])
    return titanic_runner.predict_proba.run(input_df)[:, 1]
