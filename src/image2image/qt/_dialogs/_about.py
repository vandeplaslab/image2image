"""About dialog."""

from qtextra import helpers as hp
from qtextra.widgets.qt_dialog import QtFramelessPopup
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget


def open_about(parent: QWidget) -> None:
    """Open a dialog with information about the app."""
    dlg = AboutDialog(parent)
    dlg.show()


class AboutDialog(QtFramelessPopup):
    """About dialog."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)

    # noinspection PyAttributeOutsideInit
    def make_panel(self):
        """Make panel."""
        from koyo.utilities import get_version
        from qtextra.widgets.qt_svg import QtColoredSVGIcon

        from image2image import __version__
        from image2image.assets import ICON_SVG

        package_text = "<br>"
        for package in ["image2image", "image2image-io", "image2image-reg", "valis"]:
            package_text += f"<b>{package} version:</b> {get_version(package)}<br>"

        links = {
            "project": "https://github.com/vandeplaslab/image2image",
            "github": "https://github.com/lukasz-migas",
            "website": "https://lukasz-migas.com/",
        }

        text = f"""
        <p><h2><strong>image2image</strong></h2></p>
        <p><strong>Version:</strong> {__version__}</p>
        <p><strong>Author:</strong> Lukasz G. Migas</p>
        <p><strong>Email:</strong> {
            hp.parse_link_to_link_tag("mailto:l.g.migas@tudelft.nl", "l.g.migas@tudelft.nl")
        }</p>
        <p><strong>GitHub:</strong>&nbsp;{hp.parse_link_to_link_tag(links["project"])}</p>
        <p><strong>Project's GitHub:</strong>&nbsp;{hp.parse_link_to_link_tag(links["github"])}</p>
        <p><strong>Author's website:</strong>&nbsp;{hp.parse_link_to_link_tag(links["website"])}</p>
        <br>
        {package_text}
        <br>
        <p>Developed in the Van de Plas lab</p>
        """

        pix = QtColoredSVGIcon(ICON_SVG)
        self._image = hp.make_label(self, "")
        self._image.setPixmap(pix.pixmap(300, 300))

        # about label
        self.about_label = hp.make_label(self)
        self.about_label.setText(text)
        self.about_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # set layout
        vertical_layout = hp.make_v_layout()
        vertical_layout.addWidget(self._image, alignment=Qt.AlignmentFlag.AlignHCenter)
        vertical_layout.addWidget(self.about_label)
        return vertical_layout
