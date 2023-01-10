from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def deploying_pipeline(model_fetcher, bento_builder, service_containerizer):
    # Retrieve model from MLflow and its metadata
    model, model_input_example, model_requirements = model_fetcher()

    # Build a Bento service
    bento_service_name = bento_builder(
        model=model,
        model_input_example=model_input_example,
        model_requirements=model_requirements,
    )

    # Containerize the Bento Service in a Docker image
    _ = service_containerizer(bentoml_service_name=bento_service_name)
