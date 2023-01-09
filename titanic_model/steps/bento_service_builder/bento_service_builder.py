from pathlib import Path

import bentoml
from zenml.steps import Output, step


@step
def bento_builder(model_name: str) -> Output(bentoml_service_name=str):

    bentoml_service_name = "titanic-service"

    bentoml.bentos.build(
        service="service.py:svc",
        python={
            "packages": [
                "scikit-learn",
                "pandas",
                "numpy",
                "xgboost",
                "feature_engine",
                "pydantic",
            ]
        },
        build_ctx=str(Path(__file__).resolve().parents[2] / "bento_service"),
    )

    return bentoml_service_name
