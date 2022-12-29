import numpy
import bentoml
from bentoml.io import NumpyNdarray

titanic_runner = bentoml.mlflow.get("titanic_model:latest").to_runner()

svc = bentoml.Service("titanic-service", runners=[titanic_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: numpy.ndarray) -> numpy.ndarray:
    result = titanic_runner.predict.run(input_series)
    return result
