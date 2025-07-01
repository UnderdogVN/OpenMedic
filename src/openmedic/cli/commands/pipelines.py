import click
import logging
logging.basicConfig(level=logging.INFO)

from openmedic.cli.cli_management import cli
import openmedic.core.pipelines.train as train_pipeline

@cli.command()
@click.option("--config-path", required=True, help="The configure path", type=str)
def train(config_path: str):
    train_pipeline.run(
        config_path=config_path
    )