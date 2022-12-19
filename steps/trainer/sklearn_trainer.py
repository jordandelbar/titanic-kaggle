import pandas as pd
from feature_engine.encoding import MeanEncoder, RareLabelEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from zenml.steps import Output, step

from processing.features import preprocessing


@step
def trainer(X_train: pd.DataFrame, y_train: pd.Series) -> Output(clf=Pipeline):
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

    titanic_pipeline.fit(X=X_train, y=y_train)

    return titanic_pipeline
