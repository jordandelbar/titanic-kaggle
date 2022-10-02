from pathlib import Path
from typing import List

from pydantic import BaseModel
from strictyaml import YAML, load

import titanic_model

# Package directories
PACKAGE_ROOT = Path(titanic_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / 'config.yml'
DATASET_DIR = PACKAGE_ROOT / 'datasets'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'


class AppConfig(BaseModel):
    """
    Application config
    """

    package_name: str
    training_data: str
    testing_data: str
    model_name: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    Model Config
    """

    target: str
    features: List[str]
    features_to_drop: List[str]
    vars_with_na: List[str]
    categorical_vars: List[str]
    numerical_vars: List[str]
    cat_to_impute_frequent: List[str]
    cat_to_impute_missing: List[str]
    num_to_impute: List[str]
    rare_label_to_group: List[str]
    target_label_encoding: List[str]
    feature_to_scale: List[str]
    test_size: float
    random_state: int


class Config(BaseModel):
    """
    Global config class
    """

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """
    Locate config file
    """

    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config file not found at {CONFIG_FILE_PATH!r}")


def fetch_config_yaml(cfg_path: Path = None) -> YAML:
    """
    Parse the yaml config file
    """

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, 'r') as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """
    Run validation on the parsed config
    """

    if parsed_config is None:
        parsed_config = fetch_config_yaml()

        _config = Config(
            app_config=AppConfig(**parsed_config.data),
            model_config=ModelConfig(**parsed_config.data),
        )

    return _config


config = create_and_validate_config()
