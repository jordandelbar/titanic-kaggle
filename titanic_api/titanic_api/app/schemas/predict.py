from typing import Any, List, Optional

from pydantic import BaseModel
from titanic_model.processing.validation import TitanicInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "PassengerId": 892,
                        "Pclass": 3,
                        "Name": "Kelly, Mr. James",
                        "Sex": "male",
                        "Age": 34.5,
                        "SibSp": 0,
                        "Parch": 0,
                        "Ticket": "330911",
                        "Fare": 7.83,
                        "Cabin": "B49",
                        "Embarked": "Q",
                    }
                ]
            }
        }
