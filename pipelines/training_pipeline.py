from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def training_pipeline(
    loader,
    splitter,
):
    # Link all the steps artifacts together
    train, target, test = loader()
    X_train, X_test, y_train, y_test = splitter(train, target)
