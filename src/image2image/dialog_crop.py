"""Viewer dialog."""
from qtextra._napari.mixins import ImageViewMixin
from qtextra.mixins import IndicatorMixin
from qtpy.QtWidgets import QMainWindow

# need to load to ensure all assets are loaded properly
import image2image.assets  # noqa: F401


class ImageCropWindow(QMainWindow, IndicatorMixin, ImageViewMixin):
    """Image viewer dialog."""
