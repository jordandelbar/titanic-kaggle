from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import (CategoricalImputer,
                                       MeanMedianImputer)
from feature_engine.encoding import (MeanEncoder,
                                     RareLabelEncoder)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from titanic_model.config.core import config
from titanic_model.processing.features import preprocessing


titanic_pipeline = Pipeline([
        ('preprocessing', preprocessing()),
        ('categorical_imputer_frequent',
         CategoricalImputer(imputation_method='frequent',
                            variables=config.model_config.cat_to_impute_frequent)),
        ('categorical_imputer_missing',
         CategoricalImputer(imputation_method='missing',
                            variables=config.model_config.cat_to_impute_missing)),
        ('median_imputer',
         MeanMedianImputer(imputation_method='median',
                           variables=config.model_config.num_to_impute)),
        ('rare_label_encoder',
         RareLabelEncoder(variables=config.model_config.rare_label_to_group)),
        ('mean_target_encoder',
         MeanEncoder(ignore_format=True,
                     variables=config.model_config.target_label_encoding)),
        ('last_imputer',
         MeanMedianImputer(imputation_method='mean',
                           variables=config.model_config.target_label_encoding)),
        ('scaling',
         ColumnTransformer([
            ('standard_scaler', StandardScaler(),
             config.model_config.feature_to_scale)
                ], remainder='passthrough')),
        ('clf',
         RandomForestClassifier())
    ]
)
