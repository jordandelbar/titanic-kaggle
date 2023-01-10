from pipelines.deploying_pipeline import deploying_pipeline
from steps.model_fetcher.model_fetcher import model_fetcher

from titanic_model.steps.bento_serving_steps.bento_service_builder import bento_builder
from titanic_model.steps.bento_serving_steps.bento_service_containerizer import (
    bento_containerizer,
)

run = deploying_pipeline(
    model_fetcher=model_fetcher(),
    bento_builder=bento_builder(),
    service_containerizer=bento_containerizer(),
)

if __name__ == "__main__":
    run.run()
