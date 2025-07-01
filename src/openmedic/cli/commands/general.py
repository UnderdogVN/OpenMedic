import click
from openmedic.cli.cli_management import cli
from openmedic import __version__

# Add --version option
cli = click.version_option(version=__version__)(cli)