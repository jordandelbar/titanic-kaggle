"""Training pipeline."""
from zenml.config import DockerSettings
from zenml.pipelines import pipeline

docker_settings = DockerSettings(replicate_local_python_environment="pip_freeze")


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def training_pipeline(loader, preprocessor, splitter, trainer, evaluator, register):
    """Pipeline to train our model.

    Args:
        loader: loads the data for training
        preprocessor: preprocesses the data
        splitter: split the data into train & test data
        trainer: trains the model
        evaluator: evaluate the model metrics
        register: registers the model to the MLflow server
    """
    # Load the data
    train, target, test = loader()

    # Preprocess train
    preprocessed_train = preprocessor(X=train)

    # Split the data in train and test dataset
    X_train, X_test, y_train, y_test = splitter(preprocessed_train, target)

    # Train the model
    model = trainer(X_train, y_train)

    # Evaluate the model
    metrics = evaluator(model, X_test, y_test)

    # Register the model
    _ = register(metrics)
