"""Run the infering pipeline."""
from pipelines.infering_pipeline import infering_pipeline
from steps.data_inferer.pandas_inferer import inferer
from steps.data_loader.pandas_data_loader import data_loader
from steps.data_preprocessing.pandas_preprocessor import preprocessor

run = infering_pipeline(
    loader=data_loader(),
    preprocessor=preprocessor(),
    inferer=inferer(),
)

if __name__ == "__main__":
    run.run()
