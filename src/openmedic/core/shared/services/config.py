import logging
from typing import Dict, List, Optional, Union, Literal

import yaml
from pydantic import BaseModel


class ConfigException(Exception):
    """Custom exception"""

    def __init__(self, message: str = "An error occurred in ConfigReader"):
        self.message: str = message
        super().__init__(self.message)

# Pydantic Schemas for Sections
class DataConfig(BaseModel):
    image_dir: str
    coco_annotation_path: str


class ResizeTransform(BaseModel):
    target_w: int
    target_h: int
    interpolation: Literal["INTER_LINEAR", "INTER_NEAREST"]


class FlipTransform(BaseModel):
    is_vertical: bool


class TransformConfig(BaseModel):
    Resize: Optional[ResizeTransform] = None
    Flip: Optional[FlipTransform] = None


class ModelParams(BaseModel):
    n_channels: int
    n_classes: int
    model_checkpoint: Optional[str] = None


class ModelConfig(BaseModel):
    name: str
    params: ModelParams


class PipelineConfig(BaseModel):
    seed: Optional[int] = None
    batch_size: int
    is_shuffle: Optional[bool] = None
    train_ratio: Optional[float] = None
    n_epochs: Optional[int] = None
    is_gpu: Optional[bool] = False
    verbose: Optional[bool] = False


class OptimizationConfig(BaseModel):
    name: str
    params: Dict[str, Union[float, int]]


class LossFunctionConfig(BaseModel):
    name: str
    type: str
    params: Dict[str, Union[str, List[float]]]


class MetricParams(BaseModel):
    epsilon: float
    include_background: bool
    n_classes: int


class MetricConfig(BaseModel):
    name: str
    params: MetricParams


class MonitorSaveBest(BaseModel):
    target_score: str
    patience: int


class MonitorCheckpoint(BaseModel):
    mode: Optional[Literal["eval", "train"]] = None
    save_best: Optional[MonitorSaveBest] = None


class MonitorConfig(BaseModel):
    CheckPoint: MonitorCheckpoint


# Full Config Schema
class AppConfig(BaseModel):
    data: DataConfig
    transform: Optional[TransformConfig] = None
    model: ModelConfig
    pipeline: PipelineConfig
    optimization: Optional[OptimizationConfig] = None
    loss_function: LossFunctionConfig
    metric: MetricConfig
    monitor: Optional[MonitorConfig] = None


# ConfigReader Class
class ConfigReader:
    _config: Optional[AppConfig] = None

    @classmethod
    def initialize(cls, config_path: str):
        if not config_path.endswith((".yaml", ".yml")):
            raise ConfigException("Only support `yaml` or `yml` file.")

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        try:
            cls._config = AppConfig(**raw_config)
        except Exception as e:
            raise ConfigException(f"Config validation failed: {str(e)}")

    @classmethod
    def get_field(cls, name: str):
        if not cls._config:
            raise ConfigException("ConfigReader not initialized.")

        try:
            return getattr(cls._config, name)
        except AttributeError:
            logging.warning(f"[ConfigReader][get_field]: The field `{name}` is not set in the config file.")
            return None
