"""Train the model."""
from typing import List

import mlflow
import numpy
import pandas
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
    X_train: pandas.DataFrame, y_train: pandas.Series
) -> Output(model=mlflow.pyfunc.PythonModel):
    """Train the model on the training dataframe.

    Args:
        X_train (pandas.DataFrame): train dataframe to be used for model training
        y_train (pandas.Series): target series to be used for model training
    Returns:
        clf_pipeline(sklearn.pipeline.Pipeline): classifier sklearn pipeline
    """
    mlflow.log_dict(config.dict(), "model_config.json")
    # mlflow.sklearn.autolog(log_input_examples=True)

    class TargetEncoding:
        """Target encoding."""

        def __init__(self, m: int):
            """Init.

            Args:
                m (int): weight for the overall mean
            """
            self.mapping = dict()
            self.m = m

        def fit(self, x: pandas.DataFrame, y: pandas.Series, features_list: List):
            """Fit.

            Args:
                x (pandas.DataFrame): feature dataset
                y (pandas.Series): target dataset
                features_list (list): list of features to target encode
            """
            x = x.copy()
            # Compute the global mean
            mean = y.mean()

            x["target"] = y

            # Compute the number of values and the mean of each group
            for feature in features_list:
                agg = x.groupby(feature)["target"].agg(["count", "mean"])
                counts = agg["count"]
                means = agg["mean"]

                # Compute the "smoothed" means
                smooth = (counts * means + self.m * mean) / (counts + self.m)

                # Replace each value by the according smoothed mean
                self.mapping[feature] = smooth
            return "Added to the mapping"

        def transform(self, x):
            """Transform.

            Args:
                x (pandas.DataFrame): feature dataset

            Returns:
                pandas.DataFrame: transformed dataset
            """
            x = x.copy()
            for feature in self.mapping.keys():
                x[feature] = x[feature].map(self.mapping[feature])

                if x[feature].isnull().any():
                    print(f"{feature} has unseen categories")
                    x[feature] = x[feature].fillna(x[feature].mean())

            return x

        def fit_transform(self, x, y, features_list):
            """Fit & transform.

            Args:
                x (pandas.DataFrame): feature dataset
                y (pandas.Series): target dataset
                features_list (list): list of features to target encode

            Returns:
                pandas.DataFrame: transformed dataset
            """
            self.fit(x=x, y=y, features_list=features_list)
            return self.transform(x=x)

    class MeanImputer:
        """Mean imputing."""

        def __init__(self):
            """Init."""
            self.mapping = dict()

        def fit(self, x, features_list):
            """Fit.

            Args:
                x (pandas.DataFrame): feature dataset
                features_list (list): list of features to mean impute
            """
            for feature in features_list:
                self.mapping[feature] = x[feature].mean()

        def transform(self, x):
            """Transform.

            Args:
                x (pandas.DataFrame): feature dataset

            Returns:
                pandas.DataFrame: transformed dataset
            """
            for feature in self.mapping.keys():
                x[feature] = x[feature].fillna(self.mapping[feature])
            return x

        def fit_transform(self, x, features_list):
            """Fit & transform.

            Args:
                x (pandas.DataFrame): feature dataset
                features_list (list): list of features to mean impute

            Returns:
                pandas.DataFrame: transformed dataset
            """
            self.fit(x=x, features_list=features_list)
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
                output_dim (_type_): output dimension
                epochs (int, optional): Number of training epochs. Defaults to 5000.
                loss_func (_type_, optional): Loss function.
                             Defaults to torch.nn.BCELoss().
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

        def fit(self, x: pandas.DataFrame, y: pandas.Series):
            """Fit function to accomodate sklearn pipeline API."""
            y = y.to_numpy()

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
                    print("epoch {}, loss {}".format(epoch, loss.item()))

        def predict_proba(self, x):
            """Return probability of class."""
            x = torch.from_numpy(x.astype(numpy.float32))

            y_proba = self.forward(x)
            return y_proba.flatten().detach().numpy()

        def predict(self, x: numpy.ndarray, threshold: float = 0.5):
            """Predict survival score.

            Args:
                x (numpy.ndarray): features
                threshold (float, optional): Threshold to determine label.
                                             Defaults to 0.5.

            Returns:
                numpy.ndarray: score prediction
            """
            y_pred = self.predict_proba(x)
            y_pred[y_pred > threshold] = 1
            y_pred[y_pred <= threshold] = 0
            return y_pred

        def fit_transform(self, x, y):
            """Fit & transform to accomodate sklearn pipeline API."""
            self.fit(x, y)
            return self.predict(x)

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
        input_example=X_train.loc[0].to_dict(),
    )

    return model
