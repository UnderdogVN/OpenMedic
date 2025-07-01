import argparse
import logging
from typing import List, Union, Dict
import warnings
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
import statistics

import openmedic.core.shared.helper as helper
import openmedic.core.shared.services as services
import openmedic.core.shared.services.plans as plans

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")


### MAIN PIPELINE ###
@helper.montior
def run(*, config_path: str):
    logging.info(f"[train][run]: Planning train pipeline...")
    services.ConfigReader.initialize(config_path=config_path)
    open_manager: plans.OpenMedicManager = plans.OpenMedicManager()
    open_manager.plan_train()

    logging.info(f"[train][run]: Executing train pipeline...")
    n_epochs: int = open_manager.pipeline_info["n_epochs"]

    map_results: Dict[str, list]  = {
        "train_losses": [],
        "train_metric_scores": [],
        "eval_losses": [],
        "eval_metric_scores": []
    }
    for epoch in range(1, n_epochs + 1):
        logging.info(f"[train][run]: Running epoch: {epoch}/{n_epochs}...")

        # Training progress
        open_manager.activate_train()
        train_loss: float
        train_metric_score: float
        train_loss, train_metric_score = open_manager.execute_train_per_epoch(epoch=epoch, verbose=True)
        map_results["train_losses"].append(train_loss)
        map_results["train_metric_scores"].append(train_metric_score)

        # Evaluation progress
        open_manager.activate_eval()
        eval_loss: float
        eval_metric_score: float
        eval_loss, eval_metric_score = open_manager.execute_eval_per_epoch(epoch=epoch, verbose=True)
        map_results["eval_losses"].append(eval_loss)
        map_results["eval_metric_scores"].append(eval_metric_score)



