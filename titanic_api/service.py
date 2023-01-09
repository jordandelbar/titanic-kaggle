import bentoml
import numpy
import pandas
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel

titanic_runner = bentoml.sklearn.get("titanic_model:latest").to_runner()

svc = bentoml.Service("titanic-service", runners=[titanic_runner])


class TitanicFeatures(BaseModel):
    Pclass: int = 3
    Sex: str = "male"
    Age: float = 33.0
    Fare: float = 8.0
    Embarked: str = "Q"
    is_baby: int = 0
    alone: int = 0
    family: int = 2
    title: str = "Mr"


@svc.api(input=JSON(pydantic_model=TitanicFeatures), output=NumpyNdarray())
def predict_bentoml(input_data: TitanicFeatures) -> numpy.ndarray:
    input_df = pandas.DataFrame([input_data.dict()])
    return titanic_runner.predict_proba.run(input_df)[:, 1]
