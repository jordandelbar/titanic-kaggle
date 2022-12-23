import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.steps import Output, step

from titanic_model.model_definition.sklearn_model_pipeline import titanic_pipeline

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )


@step(experiment_tracker=experiment_tracker.name)
def trainer(X_train: pd.DataFrame, y_train: pd.Series) -> Output(clf_pipeline=Pipeline):
    """Train the model on the training dataframe"""

    mlflow.sklearn.autolog(log_input_examples=True, log_post_training_metrics=False)
    titanic_pipeline.fit(X=X_train, y=y_train)

    return titanic_pipeline
