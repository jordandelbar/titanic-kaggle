from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def deploying_pipeline(
    model_fetcher, model_saver, bento_builder, service_containerizer
):
    # Retrieve model from MLflow and its metadata
    model, model_input_example, model_requirements = model_fetcher()
    bento_model_name = model_saver(model=model)
    bento_service_name = bento_builder(
        model_name=bento_model_name,
        model_input_example=model_input_example,
        model_requirements=model_requirements,
    )
    _ = service_containerizer(bentoml_service_name=bento_service_name)
