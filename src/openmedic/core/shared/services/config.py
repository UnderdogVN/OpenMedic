import logging
from typing import Dict, List, Optional, Union, Literal, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


"""CONFIGURATION FIELDS"""
# DATA FIELD
class DataField(BaseModel):
    image_dir: str
    coco_annotation_path: str


# TRANSFORM FIELD
class TransformField(BaseModel):
    # Behaviour of pydantic can be controlled via the model_config attribute on a BaseModel. https://docs.pydantic.dev/2.0/usage/model_config/
    model_config = ConfigDict(extra="allow")


# MODEL FIELD
class ModelParams(BaseModel):
    n_channels: int
    n_classes: int
    # Add the extra configuration: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.extra
    # Behaviour of pydantic can be controlled via the model_config attribute on a BaseModel. https://docs.pydantic.dev/2.0/usage/model_config/
    model_config = ConfigDict(extra="allow")


class ModelFieldTrainer(BaseModel):
    name: str
    params: ModelParams
    # In training progress, if use `model_checkpoint` then it will apply transfer learning technique.
    model_checkpoint: Optional[str] = None


class ModelFieldEvaluator(BaseModel):
    name: str
    params: ModelParams
    # In evaluation progress, `model_checkpoint` is requisite.
    model_checkpoint: str


class ModelFieldInferencer(BaseModel):
    # TODO: need to check
    name: str
    params: ModelParams
    # In evaluation progress, `model_checkpoint` is requisite.
    model_checkpoint: str


# PIPELINE FIELD
class PipelineFieldTrainer(BaseModel):
    batch_size: int
    n_epochs: int
    train_ratio: float = Field(..., gt=0, le=1)
    # Optional attributes
    seed: int = 1
    is_shuffle: bool = False
    num_workers: int = 1
    is_gpu: bool = True
    verbose: bool = True


class PipelineFieldEvaluator(BaseModel):
    batch_size: int
    # Optional attributes
    num_workers: int = 1
    is_gpu: bool = True
    verbose: bool = True
    is_shuffle: bool = False


class PipelineFieldInferencer(BaseModel):
    # TODO: Need to impelement here
    pass


# OPTIMIZATION FIELD
class OptimizationField(BaseModel):
    name: str
    params: Dict[str, Any]
    # Behaviour of pydantic can be controlled via the model_config attribute on a BaseModel. https://docs.pydantic.dev/2.0/usage/model_config/
    model_config = ConfigDict(extra="allow")


# LOSS FUNCTION FIELD
class LossFunctionField(BaseModel):
    name: str
    type: str
    params: Dict[str, Any]
    # Behaviour of pydantic can be controlled via the model_config attribute on a BaseModel. https://docs.pydantic.dev/2.0/usage/model_config/
    model_config = ConfigDict(extra="allow")


# METRIC FIELD
class MetricField(BaseModel):
    name: str
    params: Dict[str, Any]


# MONITOR FIELD
class MonitorField(BaseModel):
    # Behaviour of pydantic can be controlled via the model_config attribute on a BaseModel. https://docs.pydantic.dev/2.0/usage/model_config/
    model_config = ConfigDict(extra="allow")


"""MANIFEST ANATOMY"""
class ManifestTrainer(BaseModel):
    data: DataField
    model: ModelFieldTrainer
    pipeline: PipelineFieldTrainer
    optimization: OptimizationField
    loss_function: LossFunctionField
    metric: MetricField

    # Optional fields
    transform: Optional[TransformField] = None
    monitor: Optional[MonitorField] = None


class ManifestEvaluator(BaseModel):
    data: DataField
    model: ModelFieldEvaluator
    pipeline: PipelineFieldEvaluator
    loss_function: LossFunctionField
    metric: MetricField

    # Optional fields
    transform: Optional[TransformField] = None
    monitor: Optional[MonitorField] = None


class ManifestInferencer(BaseModel):
    # TODO: Need to implement here
    pass



"""CONFIGURATION READNING"""
# TODO: Need to implement / modify the logics below.
class ConfigException(Exception):
    """Custom exception"""
    def __init__(self, message: str = "An error occurred in ConfigReader"):
        self.message: str = message
        super().__init__(self.message)


# ConfigReader Class
class ConfigReader:
    _manifest: Optional[Union[ManifestTrainer, ManifestEvaluator, ManifestInferencer]] = None

    @classmethod
    def initialize(cls, config_path: str, mode: str):
        if not config_path.endswith((".yaml", ".yml")):
            raise ConfigException("Only support `yaml` or `yml` file.")

        with open(config_path, "r") as f:
            config: dict = yaml.safe_load(f)

        if mode not in ["train", "eval", "infer"]:
            raise ConfigException(f"Does not support with `mode` {ManifestInferencer}")
        
        try:
            if mode == "train":
                cls._manifest = ManifestTrainer(**config)
            elif mode == "eval":
                cls._manifest = ManifestEvaluator(**config)
            else:
                cls._manifest = ManifestInferencer(**config)
        except Exception as e:
            raise ConfigException(f"Config validation failed: {str(e)}")

    @classmethod
    def get_field(cls, name: str) -> dict:
        if not cls._manifest:
            raise ConfigException("ConfigReader is not initialized.")

        try:
            field: BaseModel = getattr(cls._manifest, name)
            return field.model_dump(exclude_none=True)
        except AttributeError:
            logging.warning(f"[ConfigReader][get_field]: The field `{name}` is not set in the config file.")
            return None