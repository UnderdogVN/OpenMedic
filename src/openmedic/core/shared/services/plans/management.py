import argparse
import logging
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


class OpenMedicPipeline:
    MODULE_TEMPLATE: str = "openmedic.core.pipelines.{pipeline_name}"

    @classmethod
    def execute(cls, pipeline_name: str) -> dict:
        """The abstract method for all pipelines in `openmedic.core.pipelines`."""
        args: argparse.Namespace
        parser: argparse.ArgumentParser
        pipeline: OpenMedicPipelineBase = utils.import_module(
            module_name=cls.MODULE_TEMPLATE.format(
                pipeline_name=pipeline_name,
            ),
        )
        parser = pipeline.init_arguments()
        args, _ = parser.parse_known_args()
        kwargs = {
            attr: getattr(args, attr) for attr in dir(args) if not attr.startswith("_")
        }
        kwargs["pipeline_name"] = pipeline_name
        return pipeline.run(**kwargs)

    @classmethod
    def _get_all_pipeline_info(cls) -> dict:
        return ConfigReader.get_field(name="pipeline")

    @classmethod
    def _get_train_pipeline_info(cls, pipeline_info: dict) -> dict:
        # Required fields
        batch_size: int = pipeline_info["batch_size"]
        n_epochs: int = pipeline_info["n_epochs"]
        train_ratio: float = pipeline_info["train_ratio"]
        if train_ratio > 1.0:
            OpenMedicExeception(
                f"[OpenMedicManager][_populate_objects]: train_ratio need to lower than 1.0",
            )

        # Optional fields
        seed: int = pipeline_info.get("seed", 1)
        is_shuffle: bool = pipeline_info.get("is_shuffle", False)
        num_workers: int = pipeline_info.get("num_workers", 1)
        is_gpu: bool = pipeline_info.get("is_gpu", True)

        return {
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "train_ratio": train_ratio,
            "is_shuffle": is_shuffle,
            "num_workers": num_workers,
            "seed": seed,
            "is_gpu": is_gpu,
        }

    @classmethod
    def get_pipeline_info(cls, mode: Optional[str] = None) -> dict:
        pipline_info: dict = cls._get_all_pipeline_info()
        if not mode:
            return pipline_info

        method_name = f"_get_{mode}_pipeline_info"
        method: any = getattr(cls, method_name, None)
        if not method:
            raise OpenMedicExeception(
                f"[OpenMedicPipeline][get_pipeline_config]: No method found `{method_name}`",
            )
        return method(pipline_info=pipline_info)


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
    def _get_eval_objects(cls) -> any:
        # TODO: Need to implement logics
        """Gets OpenMedic objects for evaluation pipeline."""
        pass

    @classmethod
    def _get_inference_objects(cls) -> any:
        # TODO: Need to implement logics
        """Gets OpenMedic objects for inference pipeline."""
        pass

    @classmethod
    def _get_objects(cls, mode: str) -> list:
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
        self.open_model: Optional[OpenMedicModelBase] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.train_loader: Optional[DataLoader] = None
        self.eval_loader: Optional[DataLoader] = None
        self.pipeline_info: dict = {}
        self.data_info: dict = {}
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.open_dataset, self.open_trainer = self._get_objects(mode="train")
        self.pipeline_info = OpenMedicPipeline.get_pipeline_info(mode="train")
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

        self.model, self.optimizer = self.open_trainer.get_object(
            names=["model", "optimizer"],
        )
        if self.pipeline_info["is_gpu"]:
            model = model.to(device=self.device)  # noqa: F841

    def activate_train(self):
        """Activate train mode."""
        self.open_model.train()

    def activate_eval(self):
        """Activate evaluation mode."""
        self.open_model.eval()

    def execute_train_per_epoch(self, epoch: int, verbose: bool = True) -> List[float]:
        """Execute training process per epoch.

        Input:
        ------
            epoch: int - The current epoch.
            verbose: bool - Manifests the log if True.
                By default is True.

        Output:
        -------
            train_loss_per_step, train_metric_score_per_step: List[float].
                The loss and metric score per step.
        Usage:
        ------
        ```
            open_manager.plan_train(
                config_path=config_path
            )

            n_epochs: int = open_manager.pipeline_info["n_epochs"]
            train_losses: list = []
            train_metric_scores: list = []
            for epoch in range(1, n_epochs + 1):
                open_manager.activate_train()
                result: List[float] = open_manager.execute_train_per_epoch(epoch=epoch, verbose=True)
                train_losses.append(result[0])
                train_metric_scores.append(result[1])
        ```
        """
        step: int
        images: torch.Tensor
        gts: torch.Tensor
        losses: torch.Tensor
        metric_score: float
        step_train_metric_scores: list = []
        step_train_losses: list = []
        for step, (images, gts) in enumerate(self.train_loader, 1):
            images, gts = self._process_batch(
                images=images,
                gts=gts,
            )
            self.optimizer.zero_grad()
            losses, metric_score = self.open_trainer.feedforward(images=images, gts=gts)
            step_train_metric_scores.append(metric_score)
            step_train_losses.append(losses.item())
            losses.backward()
            self.optimizer.step()

            if step % 10 == 0 and verbose:
                print(
                    f"\r\tTraining in step {step} with loss {statistics.mean(step_train_losses):.5f} and metric score: {statistics.mean(step_train_metric_scores):.5f}",
                    end="",
                    flush=True,
                )
        train_loss_per_step: float = statistics.mean(step_train_losses)
        train_metric_score_per_step: float = statistics.mean(step_train_metric_scores)
        if verbose:
            print(
                f"\r\tCompleted trainning at epoch {epoch} with loss {train_loss_per_step:.5f} and metric score: {train_metric_score_per_step:.5f}",
                flush=True,
            )

        return train_loss_per_step, train_metric_score_per_step

    def monitor_per_epoch():
        pass

    def plan_eval(self, config_path: str):
        pass

    def execute_eval(self):
        pass
