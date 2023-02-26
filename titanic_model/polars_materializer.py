"""Materializer for Polars."""
import os
from typing import Any, Type, Union

import polars
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.logger import get_logger
from zenml.materializers.base_materializer import BaseMaterializer

logger = get_logger(__name__)

PARQUET_FILENAME = "df.parquet.gzip"
COMPRESSION_TYPE = "gzip"


class PolarsMaterializer(BaseMaterializer):
    """Materializer to read data to and from polars."""

    ASSOCIATED_TYPES = (polars.DataFrame, polars.Series)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def __init__(self, uri: str):
        """Define `self.data_path`.

        Args:
            uri: The URI where the artifact data is stored.
        """
        super().__init__(uri)
        self.parquet_path = os.path.join(self.uri, PARQUET_FILENAME)

    def load(self, data_type: Type[Any]) -> Union[polars.DataFrame, polars.Series]:
        """Read `polars.DataFrame` or `polars.Series` from a `.parquet` file.

        Args:
            data_type: The type of the data to read.

        Returns:
            The polars dataframe or series.
        """
        super().load(data_type)
        with fileio.open(self.parquet_path, mode="rb") as f:
            df = polars.read_parquet(f)

        return df

    def save(self, df: Union[polars.DataFrame, polars.Series]) -> None:
        """Write a polars dataframe or series to the specified filename.

        Args:
            df: The polars dataframe or series to write.
        """
        super().save(df)

        with fileio.open(self.parquet_path, mode="wb") as f:
            df.write_parquet(f)
