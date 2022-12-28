import mlflow
from zenml.steps import step, Output
from sklearn.pipeline import Pipeline


@step
def model_fetcher() -> Output(model=Pipeline):

    model_name = "titanic-model"
    stage = "Production"

    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")

    return model
