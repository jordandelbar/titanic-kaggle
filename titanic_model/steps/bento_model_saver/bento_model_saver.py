import bentoml
from sklearn.pipeline import Pipeline
from zenml.steps import Output, step


@step
def model_saver(model: Pipeline) -> Output(bentoml_model_name=str):

    bento_model_name = "titanic_model"
    signatures = {"predict_proba": {"batchable": True, "batch_dim": 0}}

    bentoml.sklearn.save_model(
        model=model,
        signatures=signatures,
        name=bento_model_name,
    )
    return bento_model_name
