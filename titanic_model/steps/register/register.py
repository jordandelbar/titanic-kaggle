from typing import Dict

import mlflow
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
from zenml.steps import step

from titanic_model.mlflow_model_management.mlflow_model_management import (
    promote_models,
    registering_model_decision,
)

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )

mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name="titanic_training_pipeline"
)


@step(
    experiment_tracker=experiment_tracker.name,
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def model_register(metrics: Dict) -> None:
    """Register the model"""

    if registering_model_decision(
        model_name="titanic-model",
        model_accuracy=metrics["accuracy"],
        model_f1_score=["f1 score"],
    ):
        mlflow_active_run = mlflow.active_run()
        model_uri = "runs:/{}/model".format(mlflow_active_run.info.run_id)
        mlflow.register_model(model_uri, "titanic-model")
        promote_models(model_name="titanic-model", metric_to_check="accuracy")
    return None
