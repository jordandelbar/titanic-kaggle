from typing import Dict

import mlflow
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
from zenml.steps import step

from titanic_model.model_register.model_register_decision import model_register_decision

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

    client = mlflow.MlflowClient()

    staging_model = None

    for mv in client.search_model_versions("name='titanic-model'"):
        if dict(mv)["current_stage"] == "Staging":
            staging_model = dict(mv)

    mlflow_active_run = mlflow.active_run()

    staging_model_metrics = client.get_run(
        staging_model["run_id"]
    ).data.to_dictionary()["metrics"]

    staging_model_metrics = {
        key: value
        for key, value in staging_model_metrics.items()
        if not "training" in key.lower()
    }

    if model_register_decision(
        current_model_metrics=staging_model_metrics, new_model_metrics=metrics
    ):
        model_uri = "runs:/{}/model".format(mlflow_active_run.info.run_id)
        create_new_version = mlflow.register_model(model_uri, "titanic-model")
        client.transition_model_version_stage(
            name="titanic-model",
            version=create_new_version.version,
            stage="Staging",
        )
        client.transition_model_version_stage(
            name="titanic-model",
            version=staging_model["version"],
            stage="Archived",
        )
    return None
