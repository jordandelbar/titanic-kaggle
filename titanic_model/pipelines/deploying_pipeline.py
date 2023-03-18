"""Deploying pipeline."""
from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def deploying_pipeline(bento_builder):
    """Pipeline to deploy our model.

    Args:
        bento_builder: builds the model docker image for service
    """
    # Build a Bento service and containerize it
    _ = bento_builder()
