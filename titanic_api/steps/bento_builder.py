from zenml.integrations.bentoml.steps import (
    BentoMLBuilderParameters,
    bento_builder_step,
)

bento_builder = bento_builder_step(
    params=BentoMLBuilderParameters(
        model_name="titanic_model",
        model_type="sklearn",
        service="service.py:svc",
        labels={
            "framework": "scikit-learn",
            "dataset": "titanic",
            "zenml_version": "0.30.0",
        },
        python={
            "packages": [
                "zenml",
                "scikit-learn",
                "pandas",
                "numpy",
                "xgboost",
                "feature_engine",
            ],
            "lock_packages": False,
        },
    )
)
