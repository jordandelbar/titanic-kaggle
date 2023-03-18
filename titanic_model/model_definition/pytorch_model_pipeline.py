"""Model definition."""
from typing import List

import numpy
import pandas
import torch


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
            threshold (float, optional): Threshold to determine label. Defaults to 0.5.

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
