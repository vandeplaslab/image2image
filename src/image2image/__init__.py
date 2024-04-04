"""Utility tool to co-register IMS data with microscopy modality."""

from loguru import logger

__version__ = "0.1.6a2"
__author__ = "Lukasz G. Migas"
__email__ = "lukas.migas@yahoo.com"

try:
    from image2image.qt.event_loop import get_app

    # force application creation
    get_app()
except TypeError:
    pass

logger.disable("image2image")
