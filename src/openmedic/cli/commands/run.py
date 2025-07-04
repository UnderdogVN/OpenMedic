import logging

import click

logging.basicConfig(level=logging.INFO)

import openmedic.core.shared.services as services
from openmedic.cli.cli_management import cli


@cli.command()
@click.option("--pipeline", required=True, help="Pipeline to run")
def run(pipeline):
    click.echo(f"Running pipeline: {pipeline}")
    logging.info("Test")
    print(dir(services.objects.OpenMedicModel))
