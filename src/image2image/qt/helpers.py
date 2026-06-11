"""Helpers."""

import qtextra.helpers as hp
from qtpy.QtWidgets import QWidget


def warn_if_uint8(parent: QWidget) -> None:
    """Warn if a widget is uint8."""
    if not hasattr(parent, "as_uint8") and not hasattr(parent, "CONFIG"):
        return
    if parent.CONFIG.as_uint8 and not hp.confirm(
        parent,
        "parent you aware that you are exporting the image(s) with <b>uint8</b> data type? This might reduce the dynamic range of your images."
        "<br>Would you like to continue?<br><br>(If you say<b>No, we will change the value to <b>False</b>"
        " and continue)",
        title="Data Type Warning",
    ):
        parent.as_uint8.setChecked(False)
