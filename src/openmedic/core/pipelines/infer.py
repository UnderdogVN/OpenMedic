"""
OpenMedic Inference Pipeline
This module provides the functionality to run the OpenMedic inference pipeline.
Usage:
    openmedic infer --config-path <path> --input-path <path>
"""

import datetime
import logging
import warnings

import openmedic.core.shared.services as services
import openmedic.core.shared.services.plans as plans

warnings.filterwarnings("ignore")


def run(*, config_path: str) -> dict:
    services.setup_experiment_logger(
        log_filename="inference.log",
        options=services.LoggerOptions(filename="inference.log", enable_console=True, enable_color=True),
    )
    logging.info("[inference][run]: Planning inference pipeline...")
    services.ConfigReader.initialize(config_path=config_path, mode="infer")
    open_inferencer = plans.OpenMedicInferencer.initialize_with_config()
    open_inferencer.run_inference()
    now: datetime = plans.OpenMedicPipelineResult.current_time
    ts: int = int(now.timestamp())
    logging.info(f"[inference][run]: Inference completed at {now} (timestamp: {ts})")
    return {
        "timestamp": ts,
    }
