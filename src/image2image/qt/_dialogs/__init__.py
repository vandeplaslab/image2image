"""Various dialogs."""

from image2image.qt._dialogs._about import AboutDialog, open_about
from image2image.qt._dialogs._dataset import CloseDatasetDialog, DatasetDialog, SelectChannelsToLoadDialog
from image2image.qt._dialogs._extract import ExtractChannelsDialog
from image2image.qt._dialogs._locate import LocateFilesDialog
from image2image.qt._dialogs._spectrum import SpectrumDialog
from image2image.qt._dialogs._sysinfo import open_sysinfo
from image2image.qt._register._fiducials import FiducialsDialog

__all__ = [
    "AboutDialog",
    "CloseDatasetDialog",
    "DatasetDialog",
    "ExtractChannelsDialog",
    "FiducialsDialog",
    "LocateFilesDialog",
    "SelectChannelsToLoadDialog",
    "SpectrumDialog",
    "open_about",
    "open_sysinfo",
]
