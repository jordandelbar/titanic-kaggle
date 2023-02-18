"""Model definition."""
import numpy
import pandas
import torch
from feature_engine.encoding import MeanEncoder, RareLabelEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from titanic_model.config.core import config


class LogisticRegression(torch.nn.Module, BaseEstimator, TransformerMixin):
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
            loss_func (_type_, optional): Loss function. Defaults to torch.nn.BCELoss().
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

    def fit(self, X: pandas.DataFrame, y: pandas.Series):
        """Fit function to accomodate sklearn pipeline API."""
        y = y.to_numpy()

        X = torch.from_numpy(X.astype(numpy.float32))
        y = torch.from_numpy(y.astype(numpy.float32))[:, None]

        iter = 0
        epochs = self.epochs
        for epoch in range(0, epochs):

            pred_y = self.forward(X)

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

    def predict_proba(self, X):
        """Return probability of class."""
        X = torch.from_numpy(X.astype(numpy.float32))

        y_proba = self.forward(X)
        return y_proba.flatten().detach().numpy()

    def predict(self, X: numpy.ndarray, threshold: float = 0.5):
        """Predict survival score.

        Args:
            X (numpy.ndarray): features
            threshold (float, optional): Threshold to determine label. Defaults to 0.5.

        Returns:
            numpy.ndarray: score prediction
        """
        y_pred = self.predict_proba(X)
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        return y_pred

    def fit_transform(self, X, y):
        """Fit & transform to accomodate sklearn pipeline API."""
        self.fit(X, y)
        return self.predict(X)


titanic_pipeline = Pipeline(
    [
        (
            "categorical_imputer_frequent",
            CategoricalImputer(
                imputation_method="frequent",
                variables=config.cat_to_impute_frequent,
            ),
        ),
        (
            "categorical_imputer_missing",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.cat_to_impute_missing,
            ),
        ),
        (
            "median_imputer",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.num_to_impute,
            ),
        ),
        (
            "rare_label_encoder",
            RareLabelEncoder(variables=config.rare_label_to_group),
        ),
        (
            "mean_target_encoder",
            MeanEncoder(
                ignore_format=True,
                variables=config.target_label_encoding,
            ),
        ),
        (
            "last_imputer",
            MeanMedianImputer(
                imputation_method="mean",
                variables=config.target_label_encoding,
            ),
        ),
        (
            "scaling",
            ColumnTransformer(
                [
                    (
                        "standard_scaler",
                        StandardScaler(),
                        config.features_to_scale,
                    )
                ],
                remainder="passthrough",
            ),
        ),
        ("logistic_regression", LogisticRegression(input_dim=9, output_dim=1)),
    ]
)
