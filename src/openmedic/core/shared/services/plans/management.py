import datetime
import logging
import os
import statistics
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import openmedic.core.shared.services.utils as utils
from openmedic.core.shared.services.config import ConfigReader
from openmedic.core.shared.services.objects.model import OpenMedicModelBase
from openmedic.core.shared.services.plans.custom_dataset import OpenMedicDataset
from openmedic.core.shared.services.plans.custom_eval import OpenMedicEvaluator
from openmedic.core.shared.services.plans.custom_infer import OpenMedicInferencer
from openmedic.core.shared.services.plans.custom_train import OpenMedicTrainer


class OpenMedicExeception(Exception):
    """Custom exception"""

    def __init__(self, message: str = "An error occurred"):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicPipelineBase(ABC):
    """The abstract class for all OpenMedic pipelines."""

    @abstractmethod
    def init_arguments():
        pass

    @abstractmethod
    def run() -> dict:
        pass


class OpenMedicPipelineResult:
    """OpenMedicPipelineResult manages all the results (loss, metrics, ...) and informations when running pipelines."""

    train_losses: List[float] = []
    train_metric_scores: List[float] = []
    eval_losses: List[float] = []
    eval_metric_scores: List[float] = []
    metadata: dict = {}
    current_time: datetime = utils.get_current_time()
    prefix_exp: str = "experiment"
    open_model: Optional[OpenMedicModelBase] = None

    @classmethod
    def get_result(cls, attr_name: str) -> any:
        return getattr(cls, attr_name, None)

    @classmethod
    def update(cls, attr_name: str, val: any):
        attr: any = getattr(cls, attr_name, None)
        try:
            if isinstance(attr, list):
                attr.append(val)
            elif isinstance(attr, dict):
                print("yes")
                attr.update(val)
            else:
                setattr(cls, attr_name, val)
        except Exception as e:
            raise OpenMedicExeception(f"[OpenMedicPipelineResult][update]: {e}")

    @classmethod
    def init_metadata(cls, mode: str):
        method_name = f"_init_{mode}_metadata"
        method: any = getattr(cls, method_name, None)
        assert method is not None, f"The method ({method_name}) does not exist."
        cls.metadata = method()

    @classmethod
    def get_metadata(cls) -> dict:
        return cls.metadata

    @classmethod
    def get_model(cls) -> Optional[OpenMedicModelBase]:
        return cls.open_model

    @classmethod
    def _init_train_metadata(cls) -> dict:
        # Get all informations
        return {
            "data": ConfigReader.get_field(name="data"),
            "transform": ConfigReader.get_field(name="transform"),
            "model": ConfigReader.get_field(name="model"),
            "pipeline": ConfigReader.get_field(name="pipeline"),
            "optimization": ConfigReader.get_field(name="optimization"),
            "loss_function": ConfigReader.get_field(name="loss_function"),
            "metric": ConfigReader.get_field(name="metric"),
        }

    @classmethod
    def _init_eval_metadata(cls) -> dict:
        # Get all informations
        return {
            "data": ConfigReader.get_field(name="data"),
            "transform": ConfigReader.get_field(name="transform"),
            "model": ConfigReader.get_field(name="model"),
            "pipeline": ConfigReader.get_field(name="pipeline"),
            "loss_function": ConfigReader.get_field(name="loss_function"),
            "metric": ConfigReader.get_field(name="metric"),
        }

    @classmethod
    def get_scores(cls) -> dict:
        return {
            "train_losses": cls.train_losses,
            "train_metric_scores": cls.train_metric_scores,
            "eval_losses": cls.eval_losses,
            "eval_metric_scores": cls.eval_metric_scores,
        }

    @classmethod
    def get_current_experiment(cls) -> str:
        if not cls.current_time:
            raise OpenMedicExeception("`current_time` is note initalized.")
        return f'{cls.prefix_exp}_{cls.current_time.strftime("%Y%m%d.%H%M%S")}'


class OpenMedicOSEnv:
    """OpenMedicOSEnv manages all OS envs"""

    home: str = os.path.join(
        os.environ.get("OPENMEDIC_HOME", os.getcwd()),
        ".openmedic",
    )


class OpenMedicManager:
    """OpenMedicManager manages train/eval/inference pipeline."""

    ### CLASS METHODS ###
    @classmethod
    def _get_train_objects(cls) -> List[Union[OpenMedicDataset, OpenMedicTrainer]]:
        """Gets OpenMedic objects for training pipeline."""
        return [
            OpenMedicDataset.initialize_with_config(),
            OpenMedicTrainer.initialize_with_config(),
        ]

    @classmethod
    def _get_eval_objects(cls) -> List[Union[OpenMedicDataset, OpenMedicEvaluator]]:
        """Gets OpenMedic objects for evaluation pipeline."""
        return [
            OpenMedicDataset.initialize_with_config(),
            OpenMedicEvaluator.initialize_with_config(),
        ]

    @classmethod
    def _get_inference_objects(cls) -> OpenMedicInferencer:
        # TODO: Need to implement logics
        """Gets OpenMedic objects for inference pipeline."""
        return OpenMedicInferencer.initialize_with_config()

    @classmethod
    def _get_objects(cls, mode: str = "") -> list:
        """Gets OpenMedic objects.

        Input:
        ------
            mode: str - The mode to get OpenMedic objects.
                Accept values ("eval", "train", "inference")

        Output:
        -------
            list - List of OpenMedic objects.
        """
        method_name = f"_get_{mode}_objects"
        method: any = getattr(cls, method_name, None)
        if not method:
            raise OpenMedicExeception(
                f"[OpenMedicManager][_get_objects]: No method found `{method_name}`",
            )
        return method()

    ### OBJECT METHODS ###
    def __init__(self, *args, **kwargs):
        """Initializes the object with bare-bones attributes."""
        self.open_dataset: Optional[OpenMedicDataset] = None
        self.open_trainer: Optional[OpenMedicTrainer] = None
        self.open_evaluator: Optional[OpenMedicEvaluator] = None
        self.open_model: Optional[OpenMedicModelBase] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.train_loader: Optional[DataLoader] = None
        self.eval_loader: Optional[DataLoader] = None
        self.pipeline_info: dict = {}
        self.data_info: dict = {}
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mode: str = ""

    def _train_val_split(self) -> List[OpenMedicDataset]:
        """Splits dataset to train/val datasets.

        Output:
        -------
            train_dataset, val_dataset: List[OpenMedicDataset] - List of OpenMedicDataset (train_dataset and val_dataset)
        """
        generator: torch.Generator = torch.Generator().manual_seed(
            self.pipeline_info["seed"],
        )
        train_size: int = int(
            self.pipeline_info["train_ratio"] * len(self.open_dataset),
        )
        val_size: int = len(self.open_dataset) - train_size

        return random_split(
            dataset=self.open_dataset,
            lengths=[train_size, val_size],
            generator=generator,
        )

    def _permute_batch_images(self, images: torch.Tensor) -> torch.Tensor:
        """Permute image from (B, H, W, C) to (B, C, H, W).

        Input:
        ------
            images: torch.Tensor - Tensor of batch image with shape (B, H, W, C).
        """
        return images.permute(0, 3, 1, 2).float()

    def _process_batch(
        self,
        images: torch.Tensor,
        gts: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Process batch images (tensor).

        Input:
        ------
            images: torch.Tensor - Tensor of batch image with shape (B, H, W, C).
            gts: torch.Tensor - Tensor of batch ground truths with shape (B, H, W).
        Output:
        ------
            images, gts: List[torch.Tensor] - List of tensor of processed batch images.
        """

        """Steps:
            1. Permute image from (B, H, W, C) to (B, C, H, W).
            2. Convert Tensor data type.
            3. Apply device for Tensor.
        """
        images = self._permute_batch_images(images=images)
        gts = gts.long()

        if self.pipeline_info["is_gpu"]:
            images, gts = images.to(device=self.device), gts.to(device=self.device)

        return images, gts

    def plan_train(self):
        """Plans and prepares objects for traininig pipeline.
        Note: The ConfigReader need to be initialized before.
        """
        self._mode = "train"
        self.open_dataset, self.open_trainer = self._get_objects(mode=self._mode)
        self.pipeline_info = ConfigReader.get_field(name="pipeline")
        self.data_info = ConfigReader.get_field(name="data")
        logging.info(
            f"[OpenMedicManager][plan_train]: Target dataset with: \n\tImage Directory: {self.data_info['image_dir']}\n\tCOCO File: {self.data_info['coco_annotation_path']}",
        )

        train_dataset: OpenMedicDataset
        val_dataset: OpenMedicDataset
        train_dataset, val_dataset = self._train_val_split()
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.pipeline_info["batch_size"],
            shuffle=self.pipeline_info["is_shuffle"],
            num_workers=self.pipeline_info["num_workers"],
        )
        self.val_loader: DataLoader = DataLoader(
            dataset=val_dataset,
            batch_size=self.pipeline_info["batch_size"],
            shuffle=self.pipeline_info["is_shuffle"],
            num_workers=self.pipeline_info["num_workers"],
        )

        self.open_model, self.optimizer = self.open_trainer.get_object(
            names=["model", "optimizer"],
        )
        if self.pipeline_info["is_gpu"]:
            self.open_model = self.open_model.to(device=self.device)

        OpenMedicPipelineResult.init_metadata(mode=self._mode)

    def activate_train(self):
        """Activate train mode."""
        self.open_model.train()

    def activate_eval(self):
        """Activate evaluation mode."""
        self.open_model.eval()

    def execute_train_per_epoch(self, epoch: int):
        """Execute training process per epoch.
        --> Update `train_losses` and `train_metric_scores` to OpenMedicPipelineResult

        Input:
        ------
            epoch: int - The current epoch.

        Usage:
        ------
        ```
            open_manager.plan_train(
                config_path=config_path
            )
            n_epochs: int = open_manager.pipeline_info["n_epochs"]
            for epoch in range(1, n_epochs + 1):
                open_manager.activate_train()
                train_loss: float
                train_metric_score: float
                train_loss, train_metric_score = open_manager.execute_train_per_epoch(epoch=epoch)
        ```
        """
        step: int
        images: torch.Tensor
        gts: torch.Tensor
        loss: torch.Tensor
        metric_score: float
        train_metric_scores: list = []
        train_losses: list = []
        for step, (images, gts) in enumerate(self.train_loader, 1):
            images, gts = self._process_batch(
                images=images,
                gts=gts,
            )
            self.optimizer.zero_grad()
            loss, metric_score = self.open_trainer.feedforward(images=images, gts=gts)
            train_metric_scores.append(metric_score)
            train_losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if step % 10 == 0 and self.pipeline_info["verbose"]:
                print(
                    f"\r\tTraining in step {step} with loss {statistics.mean(train_losses):.5f} and metric score: {statistics.mean(train_metric_scores):.5f}",
                    end="",
                    flush=True,
                )
        train_loss_per_step: float = statistics.mean(train_losses)
        train_metric_score_per_step: float = statistics.mean(train_metric_scores)

        if self.pipeline_info["verbose"]:
            print(
                f"\r\tCompleted trainning at epoch {epoch} with loss {train_loss_per_step:.5f} and metric score: {train_metric_score_per_step:.5f}",
                flush=True,
            )

        # Update to OpenMedicPipelineResult
        OpenMedicPipelineResult.update(
            attr_name="train_losses",
            val=train_loss_per_step,
        )
        OpenMedicPipelineResult.update(
            attr_name="train_metric_scores",
            val=train_metric_score_per_step,
        )

    def monitor_per_epoch(self, **kwargs):
        """Execute monitor process per epoch.
        Update latest state `open_model` to OpenMedicPipelineResult.
        """
        # Updated to OpenMedicPipelineResult
        OpenMedicPipelineResult.update(attr_name="open_model", val=self.open_model)
        if self._mode == "train":
            self.open_trainer.execute_monitor(**kwargs)
        elif self._mode == "eval":
            self.open_evaluator.execute_monitor(**kwargs)

    def plan_eval(self):
        """Plans and prepares objects for evaluation pipeline.
        Note: The ConfigReader need to be initialized before.
        """
        self._mode = "eval"
        self.open_dataset, self.open_evaluator = self._get_objects(mode=self._mode)
        self.pipeline_info = ConfigReader.get_field(name="pipeline")
        self.data_info = ConfigReader.get_field(name="data")
        logging.info(
            f"[OpenMedicManager][plan_eval]: Target dataset with: \n\tImage Directory: {self.data_info['image_dir']}\n\tCOCO File: {self.data_info['coco_annotation_path']}",
        )

        self.eval_loader: DataLoader = DataLoader(
            dataset=self.open_dataset,
            batch_size=self.pipeline_info["batch_size"],
            shuffle=self.pipeline_info["is_shuffle"],
            num_workers=self.pipeline_info["num_workers"],
        )

        self.open_model = self.open_evaluator.get_object(names=["model"])[0]
        if self.pipeline_info["is_gpu"]:
            self.open_model = self.open_model.to(device=self.device)

        OpenMedicPipelineResult.init_metadata(mode=self._mode)

    def execute_eval_per_epoch(self, epoch: int):
        """Execute evaluation process per epoch.
        --> Update `eval_losses` and `eval_metric_scores` to OpenMedicPipelineResult

        Input:
        ------
            epoch: int - The current epoch.

        Usage:
        ------
        ```
            open_manager.plan_eval(
                config_path=config_path
            )

            open_manager.activate_eval()
            open_manager.execute_eval_per_epoch(epoch=1)

            # Monitor progress (if monitors are configured)
            open_manager.monitor_per_epoch()
        ```
        """
        step: int
        images: torch.Tensor
        gts: torch.Tensor
        loss: torch.Tensor
        metric_score: float
        eval_metric_scores: list = []
        eval_losses: list = []

        # Validate that evaluator is available
        if self.open_evaluator is None:
            raise OpenMedicExeception(
                "[OpenMedicManager][execute_eval_per_epoch]: open_evaluator is None. Please ensure plan_eval() is called first.",
            )

        # Use eval_loader for evaluation
        data_loader = self.eval_loader

        with torch.no_grad():
            for step, (images, gts) in enumerate(data_loader, 1):
                images, gts = self._process_batch(
                    images=images,
                    gts=gts,
                )
                loss, metric_score = self.open_evaluator.feedforward(
                    images=images,
                    gts=gts,
                )
                eval_metric_scores.append(metric_score)
                eval_losses.append(loss.item())

                if step % 10 == 0 and self.pipeline_info["verbose"]:
                    print(
                        f"\r\tEvaluating in step {step} with loss {statistics.mean(eval_losses):.5f} and metric score: {statistics.mean(eval_metric_scores):.5f}",
                        end="",
                        flush=True,
                    )
            eval_loss_per_step: float = statistics.mean(eval_losses)
            eval_metric_score_per_step: float = statistics.mean(eval_metric_scores)
            if self.pipeline_info["verbose"]:
                print(
                    f"\r\tCompleted evaluation at epoch {epoch} with loss {eval_loss_per_step:.5f} and metric score: {eval_metric_score_per_step:.5f}",
                    flush=True,
                )

        # Update to OpenMedicPipelineResult
        OpenMedicPipelineResult.update(attr_name="eval_losses", val=eval_loss_per_step)
        OpenMedicPipelineResult.update(
            attr_name="eval_metric_scores",
            val=eval_metric_score_per_step,
        )
