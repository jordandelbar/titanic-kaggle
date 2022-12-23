import mlflow
import pandas as pd
from feature_engine.encoding import MeanEncoder, RareLabelEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.steps import Output, step

from processing.features import preprocessing

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

    titanic_pipeline = Pipeline(
        [
            ("preprocessing", preprocessing()),
            (
                "categorical_imputer_frequent",
                CategoricalImputer(
                    imputation_method="frequent",
                    variables=["Embarked"],  # TODO: in config
                ),
            ),
            (
                "categorical_imputer_missing",
                CategoricalImputer(
                    imputation_method="missing",
                    variables=["title"],  # TODO: in config
                ),
            ),
            (
                "median_imputer",
                MeanMedianImputer(
                    imputation_method="median",
                    variables=["Age", "Fare"],  # TODO: in config
                ),
            ),
            (
                "rare_label_encoder",
                RareLabelEncoder(variables=["title"]),  # TODO: in config
            ),
            (
                "mean_target_encoder",
                MeanEncoder(
                    ignore_format=True,
                    variables=["Pclass", "Sex", "Embarked", "title"],  # TODO: in config
                ),
            ),
            (
                "last_imputer",
                MeanMedianImputer(
                    imputation_method="mean",
                    variables=["Pclass", "Sex", "Embarked", "title"],  # TODO: in config
                ),
            ),
            (
                "scaling",
                ColumnTransformer(
                    [
                        (
                            "standard_scaler",
                            StandardScaler(),
                            ["Fare"],  # TODO: in config
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
            ("clf", RandomForestClassifier()),
        ]
    )

    mlflow.sklearn.autolog()
    titanic_pipeline.fit(X=X_train, y=y_train)

    # signature = infer_signature(X_train, titanic_pipeline.predict(X_train))

    return titanic_pipeline
