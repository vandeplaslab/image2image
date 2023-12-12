"""Shortcuts helper for image2image."""
import qtextra.helpers as hp
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtpy.QtWidgets import QWidget


class ShortcutsDialog(QtFramelessTool):
    """Provide shortcuts information."""

    TITLE = "Shortcuts"
    SHORTCUTS = ""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMinimumHeight(200)

    # noinspection PyAttributeOutsideInit
    def make_panel(self):
        """Make panel."""
        return hp.make_v_layout(
            self._make_hide_handle(self.TITLE)[1],
            hp.make_label(self, self.SHORTCUTS),
            # stretch_id=1,
        )


class RegisterShortcutsDialog(ShortcutsDialog):
    """Provide shortcuts information."""

    TITLE = "Registration App Shortcuts"
    SHORTCUTS = """
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
    <br><b>T</b> - Toggle between each of the transformed 'moving' images.
    <br><b>V</b> - Toggle visibility of the  transformed 'moving' images.
    """
    # <br><b>A</b> - Zoom-in on previous point in both images.
    # <br><b>D</b> - Zoom-in on next point in both images.


class WsiPrepShortcutsDialog(ShortcutsDialog):
    """Provide shortcuts information."""

    TITLE = "WSIPREP App Shortcuts"
    SHORTCUTS = """
    <br><b>F1</b> - Open documentation in the browser.
    <br>
    <br><b>Ctrl+T</b> - Open the IPython console.
    <br><b>Ctrl+C</b> - Import configuration file.
    <br>
    <br><b>Q</b> - Rotate images left.
    <br><b>E</b> - Rotate images right.
    <br><b>W</b> - Translate images up.
    <br><b>S</b> - Translate images down.
    <br><b>A</b> - Translate images left.
    <br><b>D</b> - Translate images right.
    <br><b>F</b> - Flip images left-right.
    <br><b>R</b> - Set currently selected image as the reference.
    <br><b>Z</b> - Accept image (use in registration).
    <br><b>X</b> - Reject image (don't use in registration).
    <br><b>L</b> - Lock image - rotation, translation, flipping will no longer be updated.
    <br><b>U</b> - Unlock image - rotation, translation, flipping will be updated.
    <br><b>N</b> - Go to the next group in the Group selection (skip 'All', 'None').
    <br><b>P</b> - Go to the previous group in the Group selection (skip 'All', 'None').
    <br><b>G</b> - Toggle between the overlay and group view.
    """
