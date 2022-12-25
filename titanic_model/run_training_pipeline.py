from pipelines.training_pipeline import training_pipeline
from steps.evaluator.evaluator import model_evaluator
from steps.loader.pandas_data_loader import data_loader
from steps.register.register import model_register
from steps.splitter.sklearn_data_splitter import data_splitter
from steps.trainer.sklearn_trainer import trainer

run = training_pipeline(
    # Load the data
    loader=data_loader(),
    # Split the data in train and test dataset
    splitter=data_splitter(),
    # Train the model
    trainer=trainer(),
    # Evaluate the model
    evaluator=model_evaluator(),
    # Register the model
    register=model_register(),
)

if __name__ == "__main__":
    run.run()
