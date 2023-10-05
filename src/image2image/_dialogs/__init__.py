"""Various dialogs."""
from image2image._dialogs._about import AboutDialog, open_about
from image2image._dialogs._channels import OverlayChannelsDialog
from image2image._dialogs._dataset import (
    CloseDatasetDialog,
    ExtractChannelsDialog,
    SelectChannelsToLoadDialog,
    SelectImagesDialog,
)
from image2image._dialogs._fiducials import FiducialsDialog
from image2image._dialogs._locate import LocateFilesDialog
from image2image._dialogs._misc import ImportSelectDialog
from image2image._dialogs._transform import SelectTransformDialog

__all__ = [
    "AboutDialog",
    "open_about",
    "CloseDatasetDialog",
    "ExtractChannelsDialog",
    "SelectChannelsToLoadDialog",
    "SelectImagesDialog",
    "FiducialsDialog",
    "LocateFilesDialog",
    "ImportSelectDialog",
    "SelectTransformDialog",
    "OverlayChannelsDialog",
]

if __name__ == "__main__":  # pragma: no cover
    import sys

    from qtextra.utils.dev import apply_style, qapplication, qdev

    import image2image.assets  # noqa: F401
    from image2image.models.data import DataModel  # noqa

    model = DataModel()
    model.add_paths([r"/Users/lgmigas/Downloads/ims2micro-1.png"])

    app = qapplication()
    dlg = SelectImagesDialog(None, model)
    apply_style(dlg)
    dlg.show()
    qdev(dlg, ("qtextra", "image2image"))
    sys.exit(app.exec_())
