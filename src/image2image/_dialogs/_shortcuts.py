"""Shortcuts helper for image2image"""
from qtextra.widgets.qt_dialog import QtFramelessTool
import qtextra.helpers as hp


class ShortcutsDialog(QtFramelessTool):
    """Provide shortcuts information."""

    SHORTCUTS = ""

    def __init__(self, parent):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMinimumHeight(200)

    # noinspection PyAttributeOutsideInit
    def make_panel(self):
        """Make panel."""

        return hp.make_v_layout(
            self._make_hide_handle("Shortcuts")[1],
            hp.make_label(self, self.SHORTCUTS),
            # stretch_id=1,
        )


class RegisterShortcutsDialog(ShortcutsDialog):
    """Provide shortcuts information."""

    SHORTCUTS = """
    <h3>image2register shortcuts</h3>
    <br>
    <br><b>F1</b> - Open documentation in the browser.
    <br>
    <br><b>Ctrl+T</b> - Open the IPython console.
    <br><b>Ctrl+C</b> - Import configuration file.
    <br><b>Ctrl+F</b> - Add fixed image.
    <br><b>Ctrl+M</b> - Add moving image.
    <br>
    <br><b>1</b> - Activate zoom mode in both images.
    <br><b>2</b> - Activate add-points mode in both images.
    <br><b>3</b> - Activate move-points mode in both images.
    <br><b>L</b> - Use currently shown area in the 'Fixed' image as the 'Area of interest'.
    <br><b>Z</b> - Zoom-in on the currently set 'Area of interest'.
    """
    # <br><b>A</b> - Zoom-in on previous point in both images.
    # <br><b>D</b> - Zoom-in on next point in both images.
