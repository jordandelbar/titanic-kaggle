from pathlib import Path
from typing import List

import yaml
from zenml.steps import BaseParameters


class TrainingConfig(BaseParameters):
    model_name: str
    experiment_name: str
    training_data: str
    testing_data: str
    target: str
    features: List[str]
    test_size: float
    random_state: int
    features_to_drop: List[str]
    categorical_vars: List[str]
    numerical_vars: List[str]
    vars_with_na: List[str]
    cat_to_impute_frequent: List[str]
    cat_to_impute_missing: List[str]
    num_to_impute: List[str]
    rare_label_to_group: List[str]
    target_label_encoding: List[str]
    features_to_scale: List[str]


# Package directories
MLPIPELINE_ROOT = Path(__file__).resolve().parents[1]
ROOT = MLPIPELINE_ROOT.parent
CONFIG_FILE_PATH = MLPIPELINE_ROOT / "config.yaml"


def find_config_file() -> Path:
    """Locate config file

    Raises:
        Exception: when the config file is not found

    Returns:
        Path: Path of the config file
    """

    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config file not found at {CONFIG_FILE_PATH!r}")


def fetch_config_yaml(cfg_path: Path = None) -> TrainingConfig:
    """Parse the yaml config file

    Args:
        cfg_path (Path, optional): Path of the config file. Defaults to None.

    Raises:
        OSError: if a config file is not found at the path mentioned

    Returns:
        TrainingConfig: training configuration
    """

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = yaml.safe_load(conf_file)
            return TrainingConfig(**parsed_config)
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: TrainingConfig = None) -> TrainingConfig:
    """Create and validate config

    Args:
        parsed_config (TrainingConfig, optional): Config parsed. Defaults to None.

    Returns:
        TrainingConfig: training configuration
    """
    if parsed_config is None:
        parsed_config = fetch_config_yaml()

    return parsed_config


config = create_and_validate_config()
