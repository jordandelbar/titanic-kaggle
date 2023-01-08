from pipelines.training_pipeline import training_pipeline
from steps.data_loader.pandas_data_loader import data_loader
from steps.data_preprocessing.pandas_preprocessor import preprocessor
from steps.data_splitter.sklearn_data_splitter import data_splitter
from steps.model_evaluator.evaluator import model_evaluator
from steps.model_register.register import model_register
from steps.model_trainer.sklearn_trainer import trainer

run = training_pipeline(
    loader=data_loader(),
    preprocessor=preprocessor(),
    splitter=data_splitter(),
    trainer=trainer(),
    evaluator=model_evaluator(),
    register=model_register(),
)

if __name__ == "__main__":
    run.run()
