from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def deploying_pipeline(model_fetcher, bento_builder):
    # Retrieve model from MLflow and its metadata
    model, model_metadata, model_requirements = model_fetcher()

    # Build a Bento service and containerize it
    _ = bento_builder(
        model=model,
        model_metadata=model_metadata,
        model_requirements=model_requirements,
    )
