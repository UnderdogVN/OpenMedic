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

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")


def run(*, config_path: str, input_path: str) -> dict:
    logging.info(f"[inference][run]: Planning inference pipeline...")
    services.ConfigReader.initialize(config_path=config_path)
    open_inferencer = plans.OpenMedicInferencer.initialize_with_config()
    open_inferencer.run_inference(input_path=input_path)
    now: datetime = plans.OpenMedicPipelineResult.current_time
    ts: int = int(now.timestamp())
    logging.info(f"[inference][run]: Inference completed at {now} (timestamp: {ts})")
    return {
        "timestamp": ts,
    }
