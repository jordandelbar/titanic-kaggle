from typing import Dict

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.steps import Output, step

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )


@step(experiment_tracker=experiment_tracker.name)
def model_evaluator(
    clf_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Output(metrics=Dict):
    """Evaluate model training"""

    y_predict_proba = clf_pipeline.predict_proba(X_test)[:, 1]
    y_predict = clf_pipeline.predict(X_test)

    metrics = {
        "precision": precision_score(y_true=y_test, y_pred=y_predict),
        "recall": recall_score(y_true=y_test, y_pred=y_predict),
        "f1 score": f1_score(y_true=y_test, y_pred=y_predict),
        "accuracy": accuracy_score(y_true=y_test, y_pred=y_predict),
        "roc auc score": roc_auc_score(y_true=y_test, y_score=y_predict_proba),
    }
    mlflow.log_metrics(metrics=metrics)
    return metrics
