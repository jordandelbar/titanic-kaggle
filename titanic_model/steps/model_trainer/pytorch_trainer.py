"""Train the model."""
import logging
from typing import List, Union

import mlflow
import numpy
import polars
import torch
from config.core import config
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
    experiment_name=config.experiment_name
)


@step(
    experiment_tracker=experiment_tracker.name,
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def trainer(
    X_train: polars.DataFrame, y_train: polars.Series
) -> Output(model=mlflow.pyfunc.PythonModel):
    """Train the model on the training dataframe.

    Args:
        X_train (pandas.DataFrame): train dataframe to be used for model training
        y_train (pandas.Series): target series to be used for model training
    Returns:
        clf_pipeline(sklearn.pipeline.Pipeline): classifier sklearn pipeline
    """
    mlflow.log_dict(config.dict(), "model_config.json")

    class TargetEncoder:
        def __init__(self, smoothing: int, features_to_encode: Union[str, List]):
            """Init.

            Args:
                smoothing (int): smoothing to apply
                features_to_encode (list): list of features to encode
            """
            self.smoothing = smoothing
            self.features_to_encode = features_to_encode
            self.global_mean = None
            self.mapping = dict()

        def fit(
            self, x: polars.DataFrame, y: Union[polars.Series, polars.DataFrame]
        ) -> None:
            """Fit the target encoder.

            Args:
                x (polars.DataFrame): features table
                y (y: Union[polars.Series, polars.DataFrame]): target

            Returns:
                None
            """
            if isinstance(y, polars.DataFrame):
                on = y.columns[0]
            else:
                on = y.absname

            x = x.with_columns(y)

            # Compute the global mean
            mean = x[on].mean()
            self.global_mean = mean

            if isinstance(self.features_to_encode, str):
                self.features_to_encode = [self.features_to_encode]

            for feature in self.features_to_encode:
                # Compute the count and mean of each group
                agg = x.groupby(feature).agg(
                    [
                        polars.count().cast(polars.Float64),
                        polars.col(on).mean().cast(polars.Float64).alias("mean"),
                    ]
                )
                # Compute the smoothed mean
                smooth = agg.with_columns(
                    encoding=(
                        polars.col("count") * polars.col("mean") + self.smoothing * mean
                    )
                    / (polars.col("count") + self.smoothing)
                ).select([polars.col(feature), polars.col("encoding")])
                self.mapping[feature] = {
                    "table": smooth.to_dict(as_series=False),
                    "dtype": x.get_column(feature).dtype,
                }
            return None

        def transform(self, x: polars.DataFrame) -> polars.DataFrame:
            features_with_unseen = list()
            for feature in self.mapping.keys():
                mapping_table = polars.from_dict(self.mapping[feature]["table"])
                mapping_table = mapping_table.with_columns(
                    polars.col(feature).cast(self.mapping[feature]["dtype"])
                )
                temp = x.join(mapping_table, on=feature, how="left")
                x = temp.replace(feature, temp["encoding"]).select(x.columns)
                # Handling of unseen data
                if x.select(polars.col(feature).is_null().any()).to_numpy().squeeze():
                    features_with_unseen.append(feature)
                    x = x.with_columns(
                        polars.col(feature).fill_null(self.global_mean).alias(feature)
                    )
            if features_with_unseen:
                logging.debug(
                    f"""Feature(s) {features_with_unseen} has unseen values,
                      defaults to global mean"""
                )
            return x

        def fit_transform(
            self, x: polars.DataFrame, y: Union[polars.Series, polars.DataFrame]
        ) -> polars.DataFrame:
            self.fit(x=x, y=y)
            return self.transform(x=x)

    class MeanImputer:
        def __init__(self, features_to_impute: List):
            """Init.

            Args:
                features_to_impute (list): list of feature to impute
            """
            self.features_to_impute = features_to_impute
            self.mapping = dict()

        def fit(self, x: polars.DataFrame):
            """Fit.

            Args:
                x (polars.DataFrame): feature dataset

            Returns:
                None
            """
            for features in self.features_to_impute:
                self.mapping[features] = x[features].mean()
            return None

        def transform(self, x: polars.DataFrame):
            """Transform.

            Args:
                x (polars.DataFrame): feature dataset

            Returns:
                polars.DataFrame: transformed dataset
            """
            for feature in self.mapping.keys():
                x = x.with_columns(
                    polars.col(feature).fill_null(
                        polars.lit(self.mapping[feature]),
                    )
                )
            return x

        def fit_transform(self, x: polars.DataFrame):
            """Fit & transform.

            Args:
                x (polars.DataFrame): feature dataset
                features_list (list): list of features to mean impute

            Returns:
                polars.DataFrame: transformed dataset
            """
            self.fit(x=x)
            return self.transform(x=x)

    class LogisticRegression(torch.nn.Module):
        """Logistic Regression in PyTorch."""

        def __init__(
            self,
            input_dim: int = 9,
            output_dim: int = 1,
            epochs: int = 5000,
            loss_func=torch.nn.BCELoss(),
        ):
            """Init.

            Args:
                input_dim (int): input dimension
                output_dim (int): output dimension
                epochs (int, optional): Number of training epochs.
                                        Defaults to 5000.
                loss_func: Loss function.
            """
            super(LogisticRegression, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.loss_func = loss_func
            self.epochs = epochs
            self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        def forward(self, x):
            """Forward pass."""
            y_pred = torch.sigmoid(self.linear(x))
            return y_pred

        def fit(self, x: polars.DataFrame, y: polars.Series):
            """Fit.

            Args:
                x (polars.DataFrame): training dataframe
                y (polars.Series): target series

            Returns:
                None
            """
            x = x.to_numpy()
            y = y.to_numpy().squeeze()

            x = torch.from_numpy(x.astype(numpy.float32))
            y = torch.from_numpy(y.astype(numpy.float32))[:, None]

            iter = 0
            epochs = self.epochs
            for epoch in range(0, epochs):
                pred_y = self.forward(x)

                # Compute and print loss
                loss = self.loss_func(pred_y, y)

                # Zero gradients, perform a backward pass,
                # and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                iter += 1
                if iter % 500 == 0:
                    print(f"epoch {epoch}, loss {loss.item()}")
            return None

        def predict_proba(self, x: numpy.ndarray):
            """Return probability of class.

            Args:
                x (numpy.ndarray): dataframe to infer

            Returns:
                numpy.ndarray: probability of survival
            """
            x = torch.from_numpy(x.astype(numpy.float32))

            y_proba = self.forward(x)
            return y_proba.flatten().detach().numpy()

        def predict(self, x: numpy.ndarray, threshold: float = 0.5):
            """Predict survival score.

            Args:
                x (numpy.ndarray): dataframe to infer
                threshold (float): threshold to apply for classes

            Returns:
                numpy.ndarray: score prediction
            """
            y_pred = self.predict_proba(x)
            y_pred[y_pred > threshold] = 1
            y_pred[y_pred <= threshold] = 0
            return y_pred

    mean_imputer = MeanImputer(features_to_impute=["Age", "Fare"])
    target_encoder = TargetEncoder(
        smoothing=300,
        features_to_encode=[
            "Sex",
            "Embarked",
            "Pclass",
            "is_baby",
            "alone",
            "family",
            "title",
        ],
    )
    clf = LogisticRegression(input_dim=9, output_dim=1, epochs=5000)
    x = target_encoder.fit_transform(
        x=X_train,
        y=y_train,
    )
    x = mean_imputer.fit_transform(x=x)
    clf.fit(x=x, y=y_train)

    class CustomModel(mlflow.pyfunc.PythonModel):
        def __init__(self, mean_imputer, target_encoder, clf):
            self.mean_imputer = mean_imputer
            self.target_encoder = target_encoder
            self.clf = clf

        def predict(self, context, model_input):
            x_test = self.target_encoder.transform(x=model_input)
            x_test = self.mean_imputer.transform(x=x_test)
            return self.clf.predict(x=x_test.to_numpy())

        def predict_proba(self, context, model_input):
            x_test = self.target_encoder.transform(x=model_input)
            x_test = self.mean_imputer.transform(x=x_test)
            return self.clf.predict_proba(x=x_test.to_numpy())

    model = CustomModel(
        mean_imputer=mean_imputer, target_encoder=target_encoder, clf=clf
    )

    mlflow.pyfunc.log_model(
        python_model=model,
        artifact_path="model",
        input_example={
            key: info[0] for key, info in X_train[0, :].to_dict(as_series=False).items()
        },
    )

    return model
