from zenml.config import DockerSettings
from zenml.pipelines import pipeline

docker_settings = DockerSettings(replicate_local_python_environment="pip_freeze")


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def infering_pipeline(loader, preprocessor, inferer):
    # Load the data
    train, target, test = loader()

    # Preprocess test
    preprocessed_test = preprocessor(X=test)

    # Infer test dataframe to our web service
    _ = inferer(test=preprocessed_test)
