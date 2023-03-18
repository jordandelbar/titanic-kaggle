"""Run the training pipeline."""
from mlflow_python_model_materializer import PythonModelMaterializer
from pipelines.training_pipeline import training_pipeline
from polars_materializer import PolarsMaterializer
from steps.data_loader.polars_data_loader import data_loader
from steps.data_processing.polars_processor import preprocessor
from steps.data_splitter.sklearn_polars_data_splitter import data_splitter
from steps.model_evaluator.evaluator import model_evaluator
from steps.model_register.register import model_register
from steps.model_trainer.pytorch_trainer import trainer

run = training_pipeline(
    loader=data_loader().configure(output_materializers=PolarsMaterializer),
    preprocessor=preprocessor().configure(output_materializers=PolarsMaterializer),
    splitter=data_splitter(),
    trainer=trainer().configure(output_materializers=PythonModelMaterializer),
    evaluator=model_evaluator(),
    register=model_register(),
)

if __name__ == "__main__":
    run.run()
