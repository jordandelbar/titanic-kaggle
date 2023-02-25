"""Train the model."""
import mlflow
import pandas
from sklearn.pipeline import Pipeline
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
from zenml.steps import Output, step

from titanic_model.config.core import config
from titanic_model.model_definition.pytorch_model_pipeline import (
    LogisticRegression,
    MeanImputer,
    TargetEncoding,
)

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
def trainer(
    X_train: pandas.DataFrame, y_train: pandas.Series
) -> Output(clf_pipeline=Pipeline):
    """Train the model on the training dataframe.

    Args:
        X_train (pandas.DataFrame): train dataframe to be used for model training
        y_train (pandas.Series): target series to be used for model training
    Returns:
        clf_pipeline(sklearn.pipeline.Pipeline): classifier sklearn pipeline
    """
    mlflow.log_dict(config.dict(), "model_config.json")
    # mlflow.sklearn.autolog(log_input_examples=True)

    mean_imputer = MeanImputer()
    target_encoder = TargetEncoding(m=300)
    clf = LogisticRegression(input_dim=9, output_dim=1, epochs=5000)

    x = target_encoder.fit_transform(
        x=X_train,
        y=y_train,
        features_list=[
            "Sex",
            "Embarked",
            "Pclass",
            "is_baby",
            "alone",
            "family",
            "title",
        ],
    )
    x = mean_imputer.fit_transform(x=x, features_list=["Age"])
    clf.fit(x=x.to_numpy(), y=y_train)

    class CustomModel(mlflow.pyfunc.PythonModel):
        def __init__(self, mean_imputer, target_encoder, clf):
            self.mean_imputer = mean_imputer
            self.target_encoder = target_encoder
            self.clf = clf

        def predict(self, context, model_input):
            x_test = self.target_encoder.transform(x=model_input)
            x_test = self.mean_imputer.transform(x=x_test)
            return self.clf.predict(x=x_test.to_numpy())

    model = CustomModel(
        mean_imputer=mean_imputer, target_encoder=target_encoder, clf=clf
    )

    mlflow.pyfunc.log_model(
        python_model=model,
        artifact_path="model",
        input_example=X_train.loc[0].to_dict(),
    )

    return model
