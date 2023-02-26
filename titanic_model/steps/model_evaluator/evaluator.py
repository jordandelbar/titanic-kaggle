"""Evaluate the model."""
from typing import Dict

import mlflow
import pandas
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
from zenml.steps import Output, step

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError("Your active stack needs to contain a MLFlow experiment tracker")

mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name="titanic_training_pipeline"
)


@step(
    experiment_tracker=experiment_tracker.name,
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def model_evaluator(
    model: mlflow.pyfunc.PythonModel,
    X_test: pandas.DataFrame,
    y_test: pandas.Series,
) -> Output(metrics=Dict):
    """Evaluate model training.

    Args:
        model(None): test
        X_test (pandas.DataFrame): test dataframe to be used for model evaluation
        y_test (pandas.Series): target series to be used for model evaluation

    Returns:
        metrics (Dict): dictionnary with the different evaluation metrics
    """
    y_predict = model.predict(model_input=X_test, context="")
    y_predict_proba = model.predict_proba(model_input=X_test, context="")

    metrics = {
        "precision": precision_score(y_true=y_test, y_pred=y_predict),
        "recall": recall_score(y_true=y_test, y_pred=y_predict),
        "f1 score": f1_score(y_true=y_test, y_pred=y_predict),
        "accuracy": accuracy_score(y_true=y_test, y_pred=y_predict),
        "roc auc score": roc_auc_score(y_true=y_test, y_score=y_predict_proba),
    }
    mlflow.log_metrics(metrics=metrics)
    return metrics
