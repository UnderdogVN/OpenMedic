import datetime
import warnings

import openmedic.core.shared.helper as helper
import openmedic.core.shared.services as services
import openmedic.core.shared.services.plans as plans
from openmedic.core.shared.services.logger import logger

warnings.filterwarnings("ignore")

### MAIN PIPELINE ###
@helper.montior
def run(*, config_path: str) -> dict:
    logger.info(f"[train][run]: Planning train pipeline...")
    services.ConfigReader.initialize(config_path=config_path, mode="train")
    open_manager: plans.OpenMedicManager = plans.OpenMedicManager()
    open_manager.plan_train()
    now: datetime = plans.OpenMedicPipelineResult.current_time
    ts: int = int(now.timestamp())

    logger.info(f"[train][run]: Executing train pipeline...")
    n_epochs: int = open_manager.pipeline_info["n_epochs"]
    # breakpoint()
    logger.header()
    for epoch in range(1, n_epochs + 1):

        # Training progress
        open_manager.activate_train()
        open_manager.execute_train_per_epoch(epoch=epoch)

        # Evaluation progress
        open_manager.activate_eval()
        open_manager.execute_eval_per_epoch(epoch=epoch)

        # Console summary (single-line per epoch)
        scores = plans.OpenMedicPipelineResult.get_scores()
        train_losses = scores.get("train_losses", []) or []
        eval_losses = scores.get("eval_losses", []) or []
        train_metrics = scores.get("train_metric_scores", []) or []
        eval_metrics = scores.get("eval_metric_scores", []) or []
        train_loss = train_losses[-1] if len(train_losses) > 0 else None
        eval_loss = eval_losses[-1] if len(eval_losses) > 0 else None
        train_metric = train_metrics[-1] if len(train_metrics) > 0 else None
        eval_metric = eval_metrics[-1] if len(eval_metrics) > 0 else None
        logger.print_epoch(
            epoch_idx=epoch,
            num_epochs=n_epochs,
            train_loss=train_loss,
            eval_loss=eval_loss,
            train_metric=train_metric,
            eval_metric=eval_metric,
        )

        # Monitor progress
        open_manager.monitor_per_epoch()

    return {
        "timestamp": ts,
    }
