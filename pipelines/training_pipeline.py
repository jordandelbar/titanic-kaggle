from zenml.config import DockerSettings
from zenml.pipelines import pipeline

docker_settings = DockerSettings(replicate_local_python_environment="pip_freeze")


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def training_pipeline(loader, splitter, trainer, evaluator):
    # Link all the steps artifacts together
    train, target, test = loader()
    X_train, X_test, y_train, y_test = splitter(train, target)
    clf_pipeline = trainer(X_train, y_train)
    metrics = evaluator(clf_pipeline, X_test, y_test)  # noqa: F841
