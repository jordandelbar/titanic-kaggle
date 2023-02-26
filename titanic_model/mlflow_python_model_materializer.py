"""Materializer for Polars."""
import os
from typing import Any, Type

import cloudpickle
import mlflow
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.logger import get_logger
from zenml.materializers.base_materializer import BaseMaterializer

logger = get_logger(__name__)

PICKLE_FILENAME = "model.pkl"


class PythonModelMaterializer(BaseMaterializer):
    """Materializer to read data to and from polars."""

    ASSOCIATED_TYPES = [mlflow.pyfunc.PythonModel]
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def __init__(self, uri: str):
        """Define `self.data_path`.

        Args:
            uri: The URI where the artifact data is stored.
        """
        super().__init__(uri)
        self.pickle_path = os.path.join(self.uri, PICKLE_FILENAME)

    def load(self, data_type: Type[Any]) -> mlflow.pyfunc.PythonModel:
        """Read `polars.DataFrame` or `polars.Series` from a `.pickle` file.

        Args:
            data_type: The type of the data to read.

        Returns:
            The polars dataframe or series.
        """
        super().load(data_type)
        with fileio.open(self.pickle_path, mode="rb") as f:
            model = cloudpickle.load(f)

        return model

    def save(self, model: mlflow.pyfunc.PythonModel) -> None:
        """Blabla.

        Args:
            model: blabla.
        """
        super().save(model)

        with fileio.open(self.pickle_path, mode="wb") as f:
            cloudpickle.dump(model, f)
