from pipelines.deploying_pipeline import deploying_pipeline
from steps.bento_model_saver.bento_model_saver import model_saver
from steps.bento_service_builder.bento_service_builder import bento_builder
from steps.bento_service_containerizer.bento_service_containerizer import (
    bento_containerizer,
)
from steps.model_fetcher.model_fetcher import model_fetcher

run = deploying_pipeline(
    model_fetcher=model_fetcher(),
    model_saver=model_saver(),
    bento_builder=bento_builder(),
    service_containerizer=bento_containerizer(),
)

if __name__ == "__main__":
    run.run()
