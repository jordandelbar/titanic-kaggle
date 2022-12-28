from zenml.pipelines import pipeline


@pipeline
def deploying_pipeline(model_fetcher, bento_builder):
    # Fetch the model from mlflow
    model = model_fetcher()

    # Build the bento
    _ = bento_builder(model=model)
