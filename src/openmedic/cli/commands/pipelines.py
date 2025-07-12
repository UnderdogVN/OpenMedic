import logging

import click

logging.basicConfig(level=logging.INFO)

import openmedic.core.pipelines.eval as eval_pipeline
import openmedic.core.pipelines.infer as inference_pipeline
import openmedic.core.pipelines.train as train_pipeline
from openmedic.cli.cli_management import cli


@cli.command()
@click.option("--config-path", required=True, help="The configure path", type=str)
def train(config_path: str):
    train_pipeline.run(
        config_path=config_path,
    )


@cli.command()
@click.option("--config-path", required=True, help="The configure path", type=str)
def eval(config_path: str):
    eval_pipeline.run(
        config_path=config_path,
    )


@click.option("--config-path", required=True, help="The configure path", type=str)
@click.option(
    "--input-path",
    required=True,
    help="The input path for inference",
    type=str,
)
def infer(config_path: str, input_path: str):
    inference_pipeline.run(
        config_path=config_path,
        input_path=input_path,
    )
