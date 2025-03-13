"""Various dialogs."""

from image2image.qt._dialogs._about import AboutDialog, open_about
from image2image.qt._dialogs._channels import OverlayChannelsDialog
from image2image.qt._dialogs._dataset import (
    CloseDatasetDialog,
    DatasetDialog,
    ExtractChannelsDialog,
    SelectChannelsToLoadDialog,
)
from image2image.qt._dialogs._locate import LocateFilesDialog
from image2image.qt._dialogs._misc import ImportSelectDialog
from image2image.qt._dialogs._sysinfo import open_sysinfo
from image2image.qt._register._fiducials import FiducialsDialog
from image2image.qt._viewer._transform import SelectTransformDialog

__all__ = [
    "AboutDialog",
    "CloseDatasetDialog",
    "DatasetDialog",
    "ExtractChannelsDialog",
    "FiducialsDialog",
    "ImportSelectDialog",
    "LocateFilesDialog",
    "OverlayChannelsDialog",
    "SelectChannelsToLoadDialog",
    "SelectTransformDialog",
    "open_about",
    "open_sysinfo",
]
