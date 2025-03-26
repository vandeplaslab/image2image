"""Network plot."""

from __future__ import annotations

import typing as ty

import qtextra.helpers as hp
from image2image_reg.enums import NetworkTypes
from loguru import logger
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextraplot._mpl.views import ViewMplLine
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout, QWidget

if ty.TYPE_CHECKING:
    from image2image_reg.workflows import ElastixReg


class NetworkViewer(QtFramelessTool):
    """Network viewer."""

    def __init__(self, parent: QWidget | None = None):
        self.__parent = parent
        super().__init__(parent)
        self.setMinimumSize(600, 600)
        self.on_plot()

    @property
    def registration_model(self) -> ElastixReg:
        """Registration model."""
        if hasattr(self, "_parent"):
            return self._parent.registration_model
        return None

    def on_plot(self) -> None:
        """Plot workflow."""
        from image2image_reg.elastix.visuals import draw_workflow

        # view_type = self.view_type.currentText()
        network_type = self.network_type.currentText()
        registration_model = self.registration_model
        if registration_model is None:
            logger.warning("No registration model found.")
            return

        self.view.clear()
        draw_workflow(registration_model, self.view.figure.ax, network_type)
        self.view.setup_interactivity(zoom_color=Qt.GlobalColor.white)
        self.view.repaint()
        self.view.figure.tight()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, handle_layout = self._make_close_handle()
        self._title_label.setText("Network viewer")

        self.view = ViewMplLine(self, x_label="", y_label="", axes_size=[0.0, 0.0, 1.0, 1.0], facecolor="black")

        # self.view_type = hp.make_combobox(
        #     self,
        #     ["Registration nodes", "Entire workflow"],
        #     tooltip="Select the type of network to display.",
        #     func=self.on_plot,
        # )
        self.network_type = hp.make_combobox(
            self,
            ty.get_args(NetworkTypes),
            tooltip="Select the type of network to display.",
            value="circular",
            func=self.on_plot,
        )

        layout = hp.make_form_layout()
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addRow(handle_layout)
        layout.addRow(hp.make_h_line(self))
        # layout.addRow("View type", self.view_type)
        layout.addRow("Network type", self.network_type)
        layout.addRow(hp.make_btn(self, "Refresh", func=self.on_plot))
        layout.addRow(self.view.figure)
        return layout


if __name__ == "__main__":  # pragma: no cover
    import sys

    from qtextra.utils.dev import apply_style, qapplication

    _ = qapplication()  # analysis:ignore
    # dlg = SpectrumPlotPopup(None, title="Spectrum viewer")
    dlg = NetworkViewer(None)
    apply_style(dlg)
    dlg.show()
    sys.exit(dlg.exec_())
