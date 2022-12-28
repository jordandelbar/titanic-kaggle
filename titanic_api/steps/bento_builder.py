from pathlib import Path

from zenml.integrations.bentoml.steps import (
    BentoMLBuilderParameters,
    bento_builder_step,
)

bento_builder = bento_builder_step(
    params=BentoMLBuilderParameters(
        model_name="titanic-model",
        model_type="sklearn",
        service="service.py:svc",
        labels={
            "framework": "scikit-learn",
            "dataset": "titanic",
            "zenml_version": "0.30.0",
        },
        exclude=["titanic_model/"],
        python={
            "packages": ["zenml", "scikit-learn", "pandas", "numpy"],
            "lock_packages": False,
        },
        working_dir=str(Path(__file__).resolve().parents[1]),
    )
)
