"""Close dialog with option to not ask again."""
import typing as ty

import qtextra.helpers as hp
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from image2image.config import CONFIG


class ConfirmCloseDialog(QDialog):
    """Confirm close dialog with an option to not ask again."""

    def __init__(self, parent: QWidget, attr: str, save_func: ty.Callable) -> None:
        super().__init__(parent)
        self.attr = attr
        self.save_func = save_func

        cancel_btn = hp.make_btn(self, "Cancel")
        save_btn = hp.make_qta_btn(self, "save", label="Save", standout=True)
        close_btn = hp.make_qta_btn(self, "warning", color="orange", label="Close", standout=True)

        icon_label = hp.make_qta_label(self, "warning", color="orange")
        icon_label.set_xxlarge()

        self.do_not_ask = hp.make_checkbox(self, "Do not ask in future")

        self.setWindowTitle("Close Application?")
        shortcut = QKeySequence("Ctrl+Q").toString(QKeySequence.SequenceFormat.NativeText)
        text = (
            f"Do you want to close the application? There might be some <b>unsaved</b> changes. ('{shortcut}' to"
            f" confirm)."
        )
        close_btn.setShortcut(QKeySequence("Ctrl+Q"))

        if callable(save_func):
            save_btn.clicked.connect(self.save_func)
        else:
            save_btn.hide()
        cancel_btn.clicked.connect(self.reject)
        close_btn.clicked.connect(self.accept)

        body_layout = QVBoxLayout()
        body_layout.addWidget(hp.make_label(self, text, enable_url=True))
        body_layout.addWidget(self.do_not_ask)

        icon_layout = QHBoxLayout()
        icon_layout.addWidget(icon_label)
        icon_layout.addLayout(body_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(icon_layout)
        layout.addLayout(btn_layout)

        # for test purposes because of the problem with shortcut testing:
        # https://github.com/pytest-dev/pytest-qt/issues/254
        self.close_btn = close_btn
        self.cancel_btn = cancel_btn

    def accept(self):
        """Accept."""
        if self.do_not_ask.isChecked():
            setattr(CONFIG, self.attr, False)
        super().accept()
