import datetime
import logging
import warnings

import openmedic.core.shared.helper as helper
import openmedic.core.shared.services as services
import openmedic.core.shared.services.plans as plans

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")


### MAIN PIPELINE ###
@helper.montior
def run(*, config_path: str) -> dict:
    """Runs the evaluation pipeline.
    
    Input:
    ------
        config_path: str - Path to the configuration file.
        
    Output:
    -------
        dict - Returns evaluation results with timestamp.
    """
    logging.info(f"[eval][run]: Planning evaluation pipeline...")
    services.ConfigReader.initialize(config_path=config_path)
    open_manager: plans.OpenMedicManager = plans.OpenMedicManager()
    open_manager.plan_eval()
    now: datetime = plans.OpenMedicPipelineResult.current_time
    ts: int = int(now.timestamp())

    logging.info(f"[eval][run]: Executing evaluation pipeline...")
    n_epochs: int = open_manager.pipeline_info.get("n_epochs", 1)
    
    for epoch in range(1, n_epochs + 1):
        logging.info(f"[eval][run]: Running evaluation epoch: {epoch}/{n_epochs}...")

        # Evaluation progress
        open_manager.activate_eval()
        open_manager.execute_eval_per_epoch(epoch=epoch)

        # Monitor progress (if monitors are configured)
        open_manager.monitor_per_epoch()

    return {
        "timestamp": ts,
    } 