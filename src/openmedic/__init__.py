# openmedic/__init__.py
from .core import pipelines
from .core.shared import services

__version__ = "0.0.1"

__all__ = [
    "cli",
    "pipelines",
    "services"
]
