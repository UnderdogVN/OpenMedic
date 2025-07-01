import argparse
import logging
import statistics
import warnings
from typing import List, Union

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import openmedic.core.shared.helper as helper
import openmedic.core.shared.services as services
from openmedic.core.shared.services.plans import OpenMedicDataset, OpenMedicTrainer

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")


def _init_objects(config_path: str) -> List[Union[OpenMedicDataset, OpenMedicTrainer]]:
    services.ConfigReader.initialize(config_path=config_path)
    return (
        services.OpenMedicDataset.initialize_with_config(),
        services.OpenMedicTrainer.initialize_with_config(),
    )


def _get_pipeline_config(pipeline_info: dict) -> dict:
    # Required fields
    batch_size: int = pipeline_info["batch_size"]
    n_epochs: int = pipeline_info["n_epochs"]
    train_ratio: float = pipeline_info["train_ratio"]
    assert train_ratio < 1.0, "train_ratio need to lower than 1.0"

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


def _process_batch_images(images: torch.Tensor, device: str) -> torch.Tensor:
    """Process batch images (tensor)

    Input:
    ------
        images: torch.Tensor - Tensor of batch image with shape (B, H, W, C).
        device: str - device to use Tensor.

    Output:
    ------
        images: torch.Tensor - Tensor of processed batch images.
    """

    """Steps:
        1. Permute image from (B, H, W, C) to (B, C, H, W).
        2. Convert to float Tensor.
        3. Apply device for Tensor.
    """
    return images.permute(0, 3, 1, 2).float()


def _process_batch(images: torch.Tensor, gts: torch.Tensor, device: str):
    images = _process_batch_images(images=images, device=device)
    gts = gts.long()

    return images, gts


def train_val_split(
    custom_dataset: OpenMedicDataset,
    train_ratio: float,
    seed: int,
) -> List[OpenMedicDataset]:
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    train_size: int = int(train_ratio * len(custom_dataset))
    val_size: int = len(custom_dataset) - train_size

    return random_split(
        dataset=custom_dataset,
        lengths=[train_size, val_size],
        generator=generator,
    )


def _execute_monitor(monitor_info: dict):
    checkpoint_info: dict = monitor_info.get("CheckPoint", {})
    save_best: dict = {}
    if checkpoint_info:
        save_best: dict = checkpoint_info.get("save_best", {})

    if save_best:
        target_on_dataset: str = save_best["on"]
        target_score_type: str = save_best["target"]

    return


### MAIN PIPELINE ###
def init_arguments():
    parser = argparse.ArgumentParser(prog="")

    parser.add_argument("--config-path", dest="config_path", required=True, default="")
    return parser


@helper.montior
def run(*, pipeline_name: str, config_path: str):
    logging.info(f"[{pipeline_name}][run]: Initializing training object...")
    custom_dataset: OpenMedicDataset
    custom_trainer: OpenMedicTrainer
    custom_dataset, custom_trainer = _init_objects(config_path=config_path)
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline_info: dict = services.ConfigReader.get_field(name="pipeline")
    pipeline_config: dict = _get_pipeline_config(pipeline_info=pipeline_info)
    data_info: dict = services.ConfigReader.get_field(name="data")
    monitor_info: dict = services.ConfigReader.get_field(name="monitor")
    logging.info(
        f"[{pipeline_name}][run]: Target dataset with: \n\tImage Directory: {data_info['image_dir']}\n\tCOCO File: {data_info['coco_annotation_path']}",
    )

    if pipeline_config["is_gpu"] and device == "cpu":
        logging.warning(
            "User set to use GPU but the found the only CPU avaiable. Stopping processing...",
        )
        return {}

    train_dataset: OpenMedicDataset
    val_dataset: OpenMedicDataset
    train_dataset, val_dataset = train_val_split(
        custom_dataset=custom_dataset,
        train_ratio=pipeline_config["train_ratio"],
        seed=pipeline_config["seed"],
    )
    train_loader: DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=pipeline_config["batch_size"],
        shuffle=pipeline_config["is_shuffle"],
        num_workers=pipeline_config["num_workers"],
    )

    val_loader: DataLoader = DataLoader(
        dataset=val_dataset,
        batch_size=pipeline_config["batch_size"],
        shuffle=pipeline_config["is_shuffle"],
        num_workers=pipeline_config["num_workers"],
    )

    step: int
    images: torch.Tensor
    gts: torch.Tensor
    epoch: int
    model: services.OpenMedicModelBase
    optimizer: optim.Optimizer

    model, optimizer = custom_trainer.get_object(names=["model", "optimizer"])

    if pipeline_config["is_gpu"]:
        model = model.to(device=device)

    n_epochs: int = pipeline_config["n_epochs"]
    for epoch in range(1, n_epochs + 1):
        logging.info(f"[{pipeline_name}][run]: Running epoch: {epoch}/{n_epochs}...")
        model.train()  # mode training

        step_train_metric_scores: list = []
        step_train_losses: list = []
        for step, (images, gts) in enumerate(train_loader, 1):
            images, gts = _process_batch(
                images=images,
                gts=gts,
                device=device,
            )

            if pipeline_config["is_gpu"]:
                images, gts = images.to(device=device), gts.to(device=device)

            losses: torch.Tensor
            metric_score: float
            optimizer.zero_grad()

            losses, metric_score = custom_trainer.feedforward(images=images, gts=gts)
            step_train_metric_scores.append(metric_score)
            step_train_losses.append(losses.item())
            losses.backward()
            optimizer.step()

            if step % 10 == 0:
                print(
                    f"\r\tTraining in step {step} with loss {statistics.mean(step_train_losses):.5f} and metric score: {statistics.mean(step_train_metric_scores):.5f}",
                    end="",
                    flush=True,
                )
        train_losses: float = statistics.mean(step_train_losses)
        train_metric_scores: float = statistics.mean(step_train_metric_scores)
        print(
            f"\r\tCompleted trainning at epoch {epoch} with loss {train_losses:.5f} and metric score: {train_metric_scores:.5f}",
            flush=True,
        )

        # Activate evaluation mode
        model.eval()
        with torch.no_grad():
            step_eval_metric_scores: list = []
            step_eval_losses: list = []
            for step, (images, gts) in enumerate(val_loader, 1):
                images, gts = _process_batch(
                    images=images,
                    gts=gts,
                    device=device,
                )

                if pipeline_config["is_gpu"]:
                    images, gts = images.to(device=device), gts.to(device=device)

                losses, metric_score = custom_trainer.feedforward(
                    images=images,
                    gts=gts,
                )
                step_eval_metric_scores.append(metric_score)
                step_eval_losses.append(losses.item())

                if step % 10 == 0:
                    print(
                        f"\r\tEvaluating in step {step} with loss {statistics.mean(step_eval_losses):.5f} and metric score: {statistics.mean(step_eval_metric_scores):.5f}",
                        end="",
                        flush=True,
                    )
            eval_losses: float = statistics.mean(step_eval_losses)
            eval_metric_scores: float = statistics.mean(step_eval_metric_scores)
            print(
                f"\r\tCompleted evaluation at epoch {epoch} with loss {eval_losses:.5f} and metric score: {eval_metric_scores:.5f}",
                flush=True,
            )

        # Execute monitor
        custom_trainer.execute_monitor(
            train_losses=train_losses,
            train_metric_scores=train_metric_scores,
            eval_losses=eval_losses,
            eval_metric_scores=eval_metric_scores,
        )
    return {}


@helper.montior
def run(*, pipeline_name: str, config_path: str):
    logging.info(f"[{pipeline_name}][run]: Planning train pipeline...")
    services.ConfigReader.initialize(config_path=config_path)
    open_manager: services.plans.OpenMedicManager = services.plans.OpenMedicManager()
    open_manager.plan_train()

    logging.info(f"[{pipeline_name}][run]: Executing train pipeline...")
    n_epochs: int = open_manager.pipeline_info["n_epochs"]
    train_losses: list = []
    train_metric_scores: list = []
    for epoch in range(1, n_epochs + 1):
        logging.info(f"[{pipeline_name}][run]: Running epoch: {epoch}/{n_epochs}...")
        open_manager.activate_train()

        results: List[float] = open_manager.execute_train_per_epoch(
            epoch=epoch,
            verbose=True,
        )
        train_losses.append(results[0])
        train_metric_scores.append(results[1])
