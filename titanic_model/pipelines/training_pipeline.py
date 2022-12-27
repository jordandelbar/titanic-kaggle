from zenml.config import DockerSettings
from zenml.pipelines import pipeline

docker_settings = DockerSettings(replicate_local_python_environment="pip_freeze")


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def training_pipeline(loader, splitter, trainer, evaluator, register):
    # Load the data
    train, target, test = loader()

    # Split the data in train and test dataset
    X_train, X_test, y_train, y_test = splitter(train, target)

    # Train the model
    clf_pipeline = trainer(X_train, y_train)

    # Evaluate the model
    metrics = evaluator(clf_pipeline, X_test, y_test)

    # Register the model
    _ = register(metrics)
