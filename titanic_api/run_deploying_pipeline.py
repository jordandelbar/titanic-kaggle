from pipelines.deploying_pipeline import deploying_pipeline
from steps.model_fetcher import model_fetcher
from steps.bento_builder import bento_builder


run = deploying_pipeline(model_fetcher=model_fetcher(), bento_builder=bento_builder)


if __name__ == "__main__":
    run.run()