import logging
from typing import Dict
import warnings
import datetime

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
    now: datetime = plans.OpenMedicPipelineResult.current_time

    logging.info(f"[train][run]: Executing train pipeline...")
    n_epochs: int = open_manager.pipeline_info["n_epochs"]
    # breakpoint()
    for epoch in range(1, n_epochs + 1):
        logging.info(f"[train][run]: Running epoch: {epoch}/{n_epochs}...")

        # Training progress
        open_manager.activate_train()
        open_manager.execute_train_per_epoch(epoch=epoch)

        # Evaluation progress
        open_manager.activate_eval()
        open_manager.execute_eval_per_epoch(epoch=epoch)

        # Monitor progress
        open_manager.monitor_per_epoch()




