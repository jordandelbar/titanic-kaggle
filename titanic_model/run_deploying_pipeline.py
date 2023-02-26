"""Run the deploying pipeline."""
from pipelines.deploying_pipeline import deploying_pipeline
from steps.bento_serving.bento_service_containerizer import bento_builder

run = deploying_pipeline(
    bento_builder=bento_builder(),
)

if __name__ == "__main__":
    run.run()
