from pipelines.training_pipeline import training_pipeline
from steps.loader.pandas_data_loader import data_loader
from steps.splitter.sklearn_data_splitter import data_splitter
from steps.trainer.sklearn_trainer import trainer

run = training_pipeline(
    loader=data_loader(), splitter=data_splitter(), trainer=trainer()
)

if __name__ == "__main__":
    run.run()
