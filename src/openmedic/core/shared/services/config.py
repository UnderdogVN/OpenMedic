import logging
import os
from typing import Any, Dict, Optional, Union

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
    input_path: str
    batch_size: int = 1
    output_dir: str = ""
    is_gpu: bool = True
    verbose: bool = True
    mask_threshold: float = 0.5


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
    model: ModelFieldEvaluator
    pipeline: PipelineFieldInferencer
    transform: Optional[TransformField] = None


"""CONFIGURATION READNING"""


# TODO: Need to implement / modify the logics below.
class ConfigException(Exception):
    """Custom exception"""

    def __init__(self, message: str = "An error occurred in ConfigReader"):
        self.message: str = message
        super().__init__(self.message)


# ConfigReader Class
class ConfigReader:
    _manifest: Optional[
        Union[ManifestTrainer, ManifestEvaluator, ManifestInferencer]
    ] = None

    @classmethod
    def init_manifest(cls, config: dict, mode: str):
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
    def initialize(cls, config_path: str, mode: str):
        if not config_path.endswith((".yaml", ".yml")):
            raise ConfigException("Only support `yaml` or `yml` file.")

        with open(config_path, "r") as f:
            config: dict = yaml.safe_load(f)

        cls.init_manifest(config=config, mode=mode)

    @classmethod
    def get_field(cls, name: str) -> dict:
        if not cls._manifest:
            raise ConfigException("ConfigReader is not initialized.")

        try:
            field: BaseModel = getattr(cls._manifest, name)
            return field.model_dump(exclude_none=True)
        except AttributeError:
            logging.warning(
                f"[ConfigReader][get_field]: The field `{name}` is not set in the config file."
            )
            return None


class ExperimentLogger:
    """Centralized experiment logger setup.

    This class configures root logging for the current OpenMedic run so that:
    - All log messages from the training and management modules are written to a
      file named `training.log` under the current experiment directory
      `.openmedic/experiment_YYYYMMDD.HHMMSS/`.
    - Console logging remains enabled (if already configured) while a file
      handler is added exactly once to avoid duplicate messages.

    The experiment directory name is synchronized with the pipeline by lazily
    importing the pipeline runtime state to compute the exact path.
    """

    _initialized: bool = False
    _log_file_path: Optional[str] = None

    @classmethod
    def initialize(cls, log_filename: str = "training.log") -> str:
        """Initialize a file logger under the current experiment directory.

        Input:
        ------
            log_filename: str - The log file name to create in the experiment directory.

        Output:
        -------
            str - Absolute path to the created log file.

        Notes:
        ------
            This method is idempotent; subsequent calls will return the same path
            without re-adding handlers.
        """
        if cls._initialized and cls._log_file_path:
            return cls._log_file_path

        # Lazy import to avoid circular imports during module load
        from openmedic.core.shared.services.plans.management import (
            OpenMedicOSEnv,
            OpenMedicPipelineResult,
        )

        experiment_dir: str = os.path.join(
            OpenMedicOSEnv.home,
            OpenMedicPipelineResult.get_current_experiment(),
        )
        os.makedirs(experiment_dir, exist_ok=True)

        log_path: str = os.path.join(experiment_dir, log_filename)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Avoid adding duplicate file handlers pointing to the same path
        existing_same_file = False
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                try:
                    if os.path.abspath(getattr(handler, "baseFilename", "")) == os.path.abspath(log_path):
                        existing_same_file = True
                        break
                except Exception:
                    # If handler does not expose baseFilename, ignore
                    pass

        if not existing_same_file:
            file_handler = logging.FileHandler(log_path, mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            root_logger.addHandler(file_handler)

        cls._initialized = True
        cls._log_file_path = log_path
        logging.info(f"[ExperimentLogger]: Logging to {log_path}")
        return log_path

    @classmethod
    def get_log_path(cls) -> Optional[str]:
        """Return the absolute path to the log file if initialized."""
        return cls._log_file_path


class TrainingConsole:
    """Pretty, minimal console output for training with optional colors.

    This console prints a compact header and per-epoch lines using metrics
    available after each epoch. It does not write to files; file logging is
    handled by ExperimentLogger.
    """

    _RESET: str = "\x1b[0m"
    _BOLD: str = "\x1b[1m"
    _CYAN: str = "\x1b[36m"
    _GREEN: str = "\x1b[32m"
    _YELLOW: str = "\x1b[33m"
    _MAGENTA: str = "\x1b[35m"

    def _supports_color(self) -> bool:
        """Check if stdout likely supports ANSI colors."""
        try:
            import sys

            return sys.stdout.isatty()
        except Exception:
            return False

    def _c(self, text: str, color: str) -> str:
        """Colorize text if supported; otherwise return plain text."""
        if self._supports_color():
            return f"{color}{text}{self._RESET}"
        return text

    def _get_gpu_mem(self) -> str:
        """Return current GPU reserved memory as string like '11.9G'."""
        try:
            import torch

            if torch.cuda.is_available():
                mem_bytes: int = torch.cuda.memory_reserved(0)
                mem_gb: float = mem_bytes / 1e9
                return f"{mem_gb:.1f}G"
        except Exception:
            pass
        return "-"

    def print_header(self) -> None:
        """Print a two-line header similar to the demo in testlog.py."""
        line1: str = f"{'Epoch':>10} {'GPU_mem':>10} {'train_loss':>12} {'eval_loss':>12}"
        line2: str = f"{'':>10} {'':>10} {'train_metric':>12} {'eval_metric':>12}"
        print(self._c(self._BOLD + line1, self._CYAN))
        print(self._c(self._BOLD + line2, self._CYAN))

    def print_epoch(
        self,
        *,
        epoch_idx: int,
        num_epochs: int,
        train_loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        train_metric: Optional[float] = None,
        eval_metric: Optional[float] = None,
    ) -> None:
        """Print one epoch's summary as two aligned lines with colors.

        Input:
        ------
            epoch_idx: int - Current epoch (1-based)
            num_epochs: int - Total epochs
            train_loss/eval_loss: Optional[float] - Mean loss values
            train_metric/eval_metric: Optional[float] - Mean metric values
        """
        # Clear any in-place progress line before printing final epoch summary
        try:
            print("\r" + (" " * 160) + "\r", end="", flush=True)
        except Exception:
            pass
        epoch_str: str = f"{epoch_idx}/{num_epochs}"
        epoch_cell: str = self._c(f"{epoch_str:>10}", self._MAGENTA)
        gpu_mem: str = self._get_gpu_mem()

        train_loss_s: str = "-" if train_loss is None else f"{train_loss:.4f}"
        eval_loss_s: str = "-" if eval_loss is None else f"{eval_loss:.4f}"
        train_metric_s: str = "-" if train_metric is None else f"{train_metric:.4f}"
        eval_metric_s: str = "-" if eval_metric is None else f"{eval_metric:.4f}"

        epoch_line: str = (
            f"{epoch_cell} "
            f"{gpu_mem:>10} "
            f"{self._c(f'{train_loss_s:>12}', self._YELLOW)} "
            f"{self._c(f'{eval_loss_s:>12}', self._YELLOW)}"
        )
        val_line: str = (
            f"{'':>10} {'':>10} "
            f"{self._c(f'{train_metric_s:>12}', self._GREEN)} "
            f"{self._c(f'{eval_metric_s:>12}', self._GREEN)}"
        )
        print(epoch_line)
        print(val_line)

    def print_step_progress(
        self,
        *,
        epoch_idx: int,
        num_epochs: int,
        step_idx: int,
        total_steps: int,
        train_loss_running: Optional[float] = None,
        train_metric_running: Optional[float] = None,
    ) -> None:
        """Print a single updating progress line for training steps.

        The line overwrites itself using a carriage return and no newline, so it
        continuously updates until the epoch completes.
        """
        epoch_str: str = f"{epoch_idx}/{num_epochs}"
        gpu_mem: str = self._get_gpu_mem()
        tl_s: str = "-" if train_loss_running is None else f"{train_loss_running:.4f}"
        tm_s: str = "-" if train_metric_running is None else f"{train_metric_running:.4f}"
        line: str = (
            f"{self._c(f'{epoch_str:>10}', self._MAGENTA)} "
            f"{gpu_mem:>10} "
            f"{self._c(f'{tl_s:>12}', self._YELLOW)} "
            f"{'-':>12} "
            f"  [step {step_idx}/{total_steps}]  "
            f"metric={self._c(tm_s, self._GREEN)}"
        )
        print(f"\r{line}", end="", flush=True)
