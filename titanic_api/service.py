import bentoml
import numpy
import pandas
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel

titanic_runner = bentoml.sklearn.get("titanic_model:latest").to_runner()

svc = bentoml.Service("titanic-service", runners=[titanic_runner])


class TitanicFeatures(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    Embarked: str
    is_baby: int
    alone: int
    family: int
    title: str


@svc.api(input=JSON(pydantic_model=TitanicFeatures), output=NumpyNdarray())
def predict_bentoml(input_data: TitanicFeatures) -> numpy.ndarray:
    input_df = pandas.DataFrame([input_data.dict()])
    return titanic_runner.predict.run(input_df)
