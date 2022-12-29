from pathlib import Path
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
from zenml.steps import Output, step

from titanic_model.config.core import config
from titanic_model.model_definition.sklearn_model_pipeline import titanic_pipeline

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError("Your active stack needs to contain a MLFlow experiment tracker")

mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name=config.experiment_name
)


@step(
    experiment_tracker=experiment_tracker.name,
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def trainer(X_train: pd.DataFrame, y_train: pd.Series) -> Output(clf_pipeline=Pipeline):
    """Train the model on the training dataframe

    Args:
        X_train (pandas.DataFrame): train dataframe to be used for model training
        y_train (pandas.Series): target series to be used for model training

    Returns:
        clf_pipeline(sklearn.pipeline.Pipeline): classifier sklearn pipeline
    """

    titanic_pipeline.fit(X=X_train, y=y_train)

    mlflow.sklearn.log_model(
        sk_model=titanic_pipeline,
        artifact_path="model",
        code_paths=[str(Path(__file__).resolve().parents[2])],
    )

    return titanic_pipeline
