"""Open system info."""

from qtpy.QtWidgets import QWidget


def open_sysinfo(parent: "QWidget") -> None:
    """Open system info."""
    from qtextra.dialogs.qt_sysinfo import QtSystemInfo

    from image2image.utils.system import citation_text, get_system_info, title_text

    QtSystemInfo.show_sys_info(get_system_info(True), citation_text, title_text, parent=parent)
