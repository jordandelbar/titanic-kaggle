"""Infering pipeline."""
from zenml.config import DockerSettings
from zenml.pipelines import pipeline

docker_settings = DockerSettings(replicate_local_python_environment="pip_freeze")


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def infering_pipeline(loader, preprocessor, inferer):
    """Pipeline to infer our test data.

    Args:
        loader: loads the data for training
        preprocessor: preprocesses the data
        inferer: passes the data to the service image and gets predictions
    """
    # Load the data
    train, target, test = loader()

    # Preprocess test
    preprocessed_test = preprocessor(X=test)

    # Infer test dataframe to our web service
    _ = inferer(test=preprocessed_test)
