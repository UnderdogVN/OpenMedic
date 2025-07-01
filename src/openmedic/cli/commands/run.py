import click
import logging
logging.basicConfig(level=logging.INFO)

from openmedic.cli.cli_management import cli
import openmedic.core.shared.services as services

@cli.command()
@click.option("--pipeline", required=True, help="Pipeline to run")
def run(pipeline):
    click.echo(f"Running pipeline: {pipeline}")
    logging.info("Test")
    print(dir(services.objects.OpenMedicModel))