"""Utility tool to co-register IMS data with microscopy modality."""
# from importlib.metadata import PackageNotFoundError, version
#
# try:
#     __version__ = version("image2image")
# except PackageNotFoundError:
#     __version__ = "uninstalled"
from loguru import logger

__version__ = "0.1.6a0"
__author__ = "Lukasz G. Migas"
__email__ = "lukas.migas@yahoo.com"

try:
    from image2image.qt.event_loop import get_app

    # force application creation
    get_app()
except TypeError:
    pass

logger.disable("image2image")
