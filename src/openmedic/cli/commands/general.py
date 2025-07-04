import click

from openmedic import __version__
from openmedic.cli.cli_management import cli

# Add --version option
cli = click.version_option(version=__version__)(cli)
