"""
OpenMedic Inference Pipeline
This module provides the functionality to run the OpenMedic inference pipeline.
Usage:
    openmedic infer --config-path <path> --input-path <path>
"""

import datetime
import warnings

import openmedic.core.shared.services as services
import openmedic.core.shared.services.plans as plans
from openmedic.core.shared.services.logger import logger

warnings.filterwarnings("ignore")


def run(*, config_path: str) -> dict:
    logger.info("[inference][run]: Planning inference pipeline...")
    services.ConfigReader.initialize(config_path=config_path, mode="infer")
    open_inferencer = plans.OpenMedicInferencer.initialize_with_config()
    open_inferencer.run_inference()
    now: datetime = plans.OpenMedicPipelineResult.current_time
    ts: int = int(now.timestamp())
    logger.info(f"[inference][run]: Inference completed at {now} (timestamp: {ts})")
    return {
        "timestamp": ts,
    }
