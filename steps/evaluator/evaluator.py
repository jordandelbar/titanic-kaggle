from typing import Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from zenml.steps import Output, step


@step
def model_evaluator(
    clf_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Output(metrics=Dict):
    """Evaluate model training"""

    y_predict_proba = clf_pipeline.predict_proba(X_test)[:, 1]
    y_predict = clf_pipeline.predict(X_test)

    metrics = {
        "precision": precision_score(y_true=y_test, y_pred=y_predict),
        "recall": recall_score(y_true=y_test, y_pred=y_predict),
        "f1 score": f1_score(y_true=y_test, y_pred=y_predict),
        "accuracy": accuracy_score(y_true=y_test, y_pred=y_predict),
        "roc auc score": roc_auc_score(y_true=y_test, y_score=y_predict_proba),
    }

    print(metrics)
    return metrics
