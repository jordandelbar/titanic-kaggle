"""Preprocess the data."""
import polars
from config.core import config
from zenml.steps import Output, step


@step
def preprocessor(X: polars.DataFrame) -> Output(X_pp=polars.DataFrame):
    """Preprocess the data before the training of our model.

    Args:
        X (polars.DataFrame): train or test dataframe

    Returns:
        X_pp (polars.DataFrame): preprocessed train or test dataframe
    """
    # Is the passenger a baby
    X = X.with_columns(
        polars.when(polars.col("Age") < 5).then(1).otherwise(0).alias("is_baby")
    )

    # Was the passenger travelling alone
    X = X.with_columns(
        polars.when((polars.col("SibSp") == 0) & (polars.col("Parch") == 0))
        .then(1)
        .otherwise(0)
        .alias("alone")
    )

    # Family member total
    X = X.with_columns((polars.col("SibSp") + polars.col("Parch")).alias("family"))

    # Create a title column
    X = X.with_columns(
        polars.col("Name")
        .str.extract("([A-Za-z]+)\.")
        .str.replace("Mlle", "Miss")
        .str.replace("Ms", "Miss")
        .str.replace("Mme", "Mrs")
        .str.replace("Don", "Mr")
        .str.replace("Dona", "Mrs")
        .alias("title")
    )
    # Drop features not useful anymore
    X = X.drop(config.features_to_drop)

    return X
