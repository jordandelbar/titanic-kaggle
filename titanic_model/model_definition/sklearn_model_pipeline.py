from feature_engine.encoding import MeanEncoder, RareLabelEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from titanic_model.config.core import config
from titanic_model.model_definition.preprocessing import Preprocessing

titanic_pipeline = Pipeline(
    [
        ("preprocessing", Preprocessing()),
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
        ("clf", XGBClassifier()),
    ]
)
