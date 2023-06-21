"""Utility tool to co-register IMS data with microscopy modality."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ims2micro")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Lukasz G. Migas"
__email__ = "lukas.migas@yahoo.com"

from ims2micro.event_loop import get_app

# force application creation
get_app()
